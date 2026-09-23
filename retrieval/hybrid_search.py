"""Hybrid retrieval: semantic + BM25 -> RRF fusion -> cross-encoder reranking.

Stage contract:
    semantic_hits  : Qdrant search hits (existing schema)
    bm25_hits      : BM25 search hits (mirrors Qdrant hit schema)
    hybrid_hits    : fused + reranked hits (same schema, plus hybrid/rerank scores)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import (
    YTRAG_BM25_K,
    YTRAG_HYBRID_K,
    YTRAG_RERANKER_DEVICE,
    YTRAG_RERANKER_MODEL,
    YTRAG_RETRIEVAL_K,
    YTRAG_RRF_K,
    YTRAG_TOP_K,
)
from retrieval.bm25_search import BM25Retriever
from retrieval.filters import apply_score_filter, deduplicate_overlaps

logger = logging.getLogger(__name__)

# Stable key for deduplication across sources.
_HIT_KEY = "{video_id}::{chunk_id}"


def _hit_key(hit: Dict[str, Any]) -> str:
    return _HIT_KEY.format(
        video_id=str(hit.get("video_id", "")),
        chunk_id=str(hit.get("chunk_id", "")),
    )


def _normalize_scores(hits: List[Dict[str, Any]], score_key: str = "score") -> List[float]:
    """Min-max normalize scores to [0, 1]. Flat list returns zeros."""
    scores = [float(hit.get(score_key, 0.0) or 0.0) for hit in hits]
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s <= min_s:
        # Single-element or flat list: treat as equally relevant.
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def reciprocal_rank_fusion(
    semantic_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    Each hit keeps:
        - original metadata
        - semantic_score (normalized, or 0 if from BM25-only)
        - bm25_score (normalized, or 0 if from semantic-only)
        - hybrid_score (RRF score)
    """
    seen: Dict[str, Dict[str, Any]] = {}
    lists = [
        (semantic_hits, "semantic_score"),
        (bm25_hits, "bm25_score"),
    ]

    for hits, score_field in lists:
        norm = _normalize_scores(hits, "score")
        for rank, (hit, norm_score) in enumerate(zip(hits, norm), start=1):
            key = _hit_key(hit)
            if key not in seen:
                seen[key] = dict(hit)
                seen[key]["semantic_score"] = 0.0
                seen[key]["bm25_score"] = 0.0
                seen[key]["hybrid_score"] = 0.0
            seen[key][score_field] = max(seen[key][score_field], norm_score)
            seen[key]["hybrid_score"] += 1.0 / (k + rank)

    fused = list(seen.values())
    fused.sort(key=lambda h: float(h.get("hybrid_score", 0.0)), reverse=True)
    return fused


@dataclass
class Reranker:
    """Lazy-loaded cross-encoder reranker.

    Uses sentence-transformers CrossEncoder. GPU is used only when explicitly
    requested and available; otherwise falls back to CPU to avoid VRAM pressure.
    """

    model_name: str = YTRAG_RERANKER_MODEL
    device: str = YTRAG_RERANKER_DEVICE
    _model: Any = field(default=None, repr=False, compare=False)

    def _resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        resolved = self._resolve_device()
        logger.debug("Loading reranker model '%s' on '%s'", self.model_name, resolved)
        self._model = CrossEncoder(self.model_name, device=resolved)

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """Score (query, chunk_text) pairs and return hits sorted by rerank score."""
        if not hits:
            return []
        self._load()
        pairs = [(query, hit.get("text", "")) for hit in hits]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Reranker failed, returning original order: %s", exc)
            return hits[:top_n]

        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
        hits.sort(key=lambda h: float(h.get("rerank_score", 0.0)), reverse=True)
        return hits[: max(int(top_n), 1)]


class HybridRetriever:
    """Combine semantic search, BM25, RRF fusion, and cross-encoder reranking.

    Gracefully falls back to semantic-only retrieval if BM25 or the reranker
    fails, so the existing --ask pipeline never crashes.
    """

    def __init__(
        self,
        semantic_top_k: int = YTRAG_RETRIEVAL_K,
        bm25_top_k: int = YTRAG_BM25_K,
        hybrid_k: int = YTRAG_HYBRID_K,
        rrf_k: int = YTRAG_RRF_K,
        final_top_k: int = YTRAG_TOP_K,
        min_score: float = 0.25,
        dedup_overlap_ratio: float = 0.7,
    ):
        self.semantic_top_k = max(int(semantic_top_k), 1)
        self.bm25_top_k = max(int(bm25_top_k), 1)
        self.hybrid_k = max(int(hybrid_k), 1)
        self.rrf_k = max(int(rrf_k), 1)
        self.final_top_k = max(int(final_top_k), 1)
        self.min_score = float(min_score)
        self.dedup_overlap_ratio = float(dedup_overlap_ratio)

        # Lazy imports/deferred init to avoid heavy startup cost when not used.
        self._semantic_store = None
        self._bm25_retriever = None
        self._reranker = None

    def _get_semantic(self) -> Any:
        if self._semantic_store is None:
            from vector_store import QdrantVectorStore
            self._semantic_store = QdrantVectorStore()
        return self._semantic_store

    def _get_bm25(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever()
        return self._bm25_retriever

    def _get_reranker(self) -> Optional[Reranker]:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Run the full hybrid pipeline and return a structured result.

        Keys:
            query, semantic, bm25, hybrid, reranked, final,
            semantic_top_k, bm25_top_k, hybrid_k, final_top_k,
            fallback, error
        """
        result: Dict[str, Any] = {
            "query": query,
            "semantic": [],
            "bm25": [],
            "hybrid": [],
            "reranked": [],
            "final": [],
            "semantic_top_k": self.semantic_top_k,
            "bm25_top_k": self.bm25_top_k,
            "hybrid_k": self.hybrid_k,
            "final_top_k": self.final_top_k,
            "fallback": False,
            "error": None,
        }

        # Stage 1: semantic retrieval.
        try:
            semantic_hits = list(
                self._get_semantic().search(query, top_k=self.semantic_top_k)
            )
        except Exception as exc:
            logger.warning("Semantic retrieval failed: %s", exc)
            semantic_hits = []
        result["semantic"] = semantic_hits

        # Stage 2: BM25 retrieval (optional; failure does not block pipeline).
        bm25_hits: List[Dict[str, Any]] = []
        try:
            bm25_hits = self._get_bm25().search(query, k=self.bm25_top_k)
        except Exception as exc:
            logger.warning("BM25 retrieval failed, continuing without it: %s", exc)
        result["bm25"] = bm25_hits

        if not semantic_hits and not bm25_hits:
            result["fallback"] = True
            result["error"] = "Both semantic and BM25 retrieval returned no hits."
            return result

        # Stage 3: RRF fusion.
        try:
            hybrid = reciprocal_rank_fusion(semantic_hits, bm25_hits, k=self.rrf_k)
            hybrid = apply_score_filter(hybrid, self.min_score)
            hybrid = deduplicate_overlaps(hybrid, self.dedup_overlap_ratio)
            result["hybrid"] = hybrid[: self.hybrid_k]
        except Exception as exc:
            logger.warning("Hybrid fusion failed, using semantic fallback: %s", exc)
            result["hybrid"] = semantic_hits[: self.hybrid_k]
            result["fallback"] = True
            result["error"] = str(exc)

        # Stage 4: cross-encoder reranking (optional; failure keeps hybrid order).
        to_rerank = result["hybrid"][: self.hybrid_k]
        reranked = to_rerank
        try:
            reranker = self._get_reranker()
            if reranker is not None:
                reranked = reranker.rerank(query, to_rerank, top_n=self.final_top_k)
            else:
                reranked = to_rerank[: self.final_top_k]
        except Exception as exc:
            logger.warning("Reranking failed, using hybrid order: %s", exc)
            reranked = to_rerank[: self.final_top_k]
        result["reranked"] = reranked

        # Final trim.
        result["final"] = reranked[: self.final_top_k]
        return result

    @staticmethod
    def print_report(result: Dict[str, Any]) -> None:
        """Pretty-print the hybrid pipeline stages for --debug-search."""
        query = result.get("query", "")
        semantic = result.get("semantic", [])
        bm25 = result.get("bm25", [])
        hybrid = result.get("hybrid", [])
        reranked = result.get("reranked", [])
        final = result.get("final", [])

        print("\n" + "=" * 70)
        print("HYBRID RETRIEVAL REPORT")
        print("=" * 70)
        print(f"QUERY: {query}\n")

        print("Semantic results (Qdrant):")
        if not semantic:
            print("  (none)")
        for i, hit in enumerate(semantic, start=1):
            score = hit.get("score", 0.0)
            title = hit.get("title", "Unknown Title")
            start_sec = hit.get("start_sec", 0)
            ts = _format_timestamp(start_sec)
            print(f"  [{i:>2}] score={score:.4f}  title='{title}'  ts={ts}")

        print("\nBM25 results:")
        if not bm25:
            print("  (none)")
        for i, hit in enumerate(bm25, start=1):
            score = hit.get("score", 0.0)
            title = hit.get("title", "Unknown Title")
            start_sec = hit.get("start_sec", 0)
            ts = _format_timestamp(start_sec)
            print(f"  [{i:>2}] score={score:.4f}  title='{title}'  ts={ts}")

        print("\nHybrid results (RRF):")
        if not hybrid:
            print("  (none)")
        for i, hit in enumerate(hybrid, start=1):
            score = hit.get("hybrid_score", 0.0)
            title = hit.get("title", "Unknown Title")
            start_sec = hit.get("start_sec", 0)
            ts = _format_timestamp(start_sec)
            print(f"  [{i:>2}] score={score:.4f}  title='{title}'  ts={ts}")

        print("\nReranked results:")
        if not reranked:
            print("  (none)")
        for i, hit in enumerate(reranked, start=1):
            score = hit.get("rerank_score", 0.0)
            title = hit.get("title", "Unknown Title")
            start_sec = hit.get("start_sec", 0)
            ts = _format_timestamp(start_sec)
            print(f"  [{i:>2}] score={score:.4f}  title='{title}'  ts={ts}")

        print(f"\nFinal top-{len(final)}:")
        for i, hit in enumerate(final, start=1):
            score = hit.get("rerank_score") or hit.get("hybrid_score") or hit.get("score", 0.0)
            title = hit.get("title", "Unknown Title")
            start_sec = hit.get("start_sec", 0)
            ts = _format_timestamp(start_sec)
            text = (hit.get("text") or "").replace("\n", " ")[:120]
            print(f"  [{i}] score={score:.4f}  title='{title}'  ts={ts}")
            print(f"      text: {text}...")

        if result.get("fallback"):
            print(f"\n[FALLBACK] {result.get('error')}")
        print("=" * 70 + "\n")


def _format_timestamp(start_sec: Any) -> str:
    try:
        secs = float(start_sec)
        m, s = divmod(int(secs), 60)
        return f"{m:02d}:{s:02d}"
    except (TypeError, ValueError):
        return "00:00"
