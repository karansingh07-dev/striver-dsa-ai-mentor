"""Two-stage retrieval pipeline used by both the RAG engine and the evaluator.

Stages:
    1. Candidates      : Qdrant search for YTRAG_RETRIEVAL_K (default 15) hits
    2. Score gate      : drop hits below min_score (cosine; higher = better)
    3. Overlap dedup   : collapse >= 0.7-overlapping chunks from the same video
    4. Top-K trim      : keep the best YTRAG_TOP_K (default 5) chunks

The pipeline itself performs no RAG / LLM work and builds no answers — it only
answers "which chunks are the strongest evidence for this query?".
"""

from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_TOP_K,
    YTRAG_DEDUP_OVERLAP_RATIO,
    YTRAG_MIN_SCORE,
    YTRAG_RETRIEVAL_K,
)
from retrieval.filters import apply_score_filter, deduplicate_overlaps, overlap_ratio
from utils import format_seconds_to_timestamp
from vector_store import QdrantVectorStore

TEXT_PREVIEW_CHARS = 160


class RetrievalPipeline:
    """Retrieve -> score filter -> deduplicate -> top-K trim."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        retrieval_k: Optional[int] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        dedup_overlap_ratio: Optional[float] = None,
    ) -> None:
        self.vector_store = vector_store or QdrantVectorStore()
        # Stage-1 pool must never be smaller than the final top-k, otherwise the
        # later stages could never produce top_k chunks.
        self.retrieval_k = max(
            int(retrieval_k if retrieval_k is not None else YTRAG_RETRIEVAL_K),
            int(top_k if top_k is not None else DEFAULT_TOP_K),
        )
        self.top_k = max(int(top_k if top_k is not None else DEFAULT_TOP_K), 1)
        self.min_score = (
            float(min_score)
            if min_score is not None
            else float(YTRAG_MIN_SCORE)
        )
        self.dedup_overlap_ratio = float(
            dedup_overlap_ratio
            if dedup_overlap_ratio is not None
            else YTRAG_DEDUP_OVERLAP_RATIO
        )

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs all four stages and returns a structured result dict.

        Result keys:
            query, candidates, after_score_filter, after_dedup, final,
            retrieval_k, top_k, min_score, overlap_ratio
        """
        top_k_use = max(int(top_k if top_k is not None else self.top_k), 1)
        min_score_use = (
            float(min_score)
            if min_score is not None
            else float(self.min_score)
        )

        # Stage 1: candidate pool (broader recall than the final context).
        candidates: List[Dict[str, Any]] = list(
            self.vector_store.search(query, top_k=self.retrieval_k)
        )

        # Stage 2: score gate.
        after_score_filter: List[Dict[str, Any]] = apply_score_filter(
            candidates, min_score_use
        )

        # Stage 3: overlap deduplication (same video only).
        after_dedup: List[Dict[str, Any]] = deduplicate_overlaps(
            after_score_filter, self.dedup_overlap_ratio
        )

        # Stage 4: trim to the best top_k chunks (kept in score-descending order).
        final: List[Dict[str, Any]] = after_dedup[:top_k_use]

        return {
            "query": query,
            "candidates": candidates,
            "after_score_filter": after_score_filter,
            "after_dedup": after_dedup,
            "final": final,
            "retrieval_k": self.retrieval_k,
            "top_k": top_k_use,
            "min_score": min_score_use,
            "overlap_ratio": self.dedup_overlap_ratio,
        }
# ------------------------------------------------------------------
    # Debug / CLI report (used by --search and --ask --debug)
    # ------------------------------------------------------------------
    @staticmethod
    def _preview(text: str, limit: int = TEXT_PREVIEW_CHARS) -> str:
        text = (text or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ..."

    @staticmethod
    def _short(hit: Dict[str, Any]) -> str:
        ts = format_seconds_to_timestamp(hit.get("start_sec", 0))
        return (
            f"score={hit.get('score')}  title='{hit.get('title', 'Unknown Title')}'  "
            f"ts={ts} (from {hit.get('start_sec')}s)"
        )

    @staticmethod
    def print_report(result: Dict[str, Any]) -> None:
        """Prints the full retrieval report: Query -> candidates -> stage counts."""
        query = result.get("query", "")
        candidates: List[Dict[str, Any]] = result.get("candidates", [])
        after_filter: List[Dict[str, Any]] = result.get("after_score_filter", [])
        after_dedup: List[Dict[str, Any]] = result.get("after_dedup", [])
        final: List[Dict[str, Any]] = result.get("final", [])

        print("\n" + "=" * 70)
        print("RETRIEVAL PIPELINE REPORT")
        print("=" * 70)
        print(f"Query      : {query}")
        print(
            f"Stage sizes: candidates={len(candidates)} "
            f"(retrieval_k={result.get('retrieval_k')}) -> "
            f"score gate (min_score={result.get('min_score')}) -> "
            f"dedup (ratio={result.get('overlap_ratio')}) -> "
            f"final top_k={result.get('top_k')}"
        )

        print("\nStage 1 - Candidates:")
        if not candidates:
            print("  (none)")
        else:
            for i, hit in enumerate(candidates, start=1):
                print(f"  [{i:>2}] {RetrievalPipeline._short(hit)}")
                print(f"        {RetrievalPipeline._preview(hit.get('text', ''))}")

        print(f"\nAfter score filtering (cosine score >= {result.get('min_score')}): "
              f"{len(after_filter)} chunks")
        print(f"After deduplication (same-video overlap >= {result.get('overlap_ratio')}): "
              f"{len(after_dedup)} chunks")

        print(f"\nFinal context chunks (top {result.get('top_k')}):")
        if not final:
            print("  (none - no chunks passed the score gate)")
        else:
            for i, hit in enumerate(final, start=1):
                print(f"  [{i}] {RetrievalPipeline._short(hit)}")

        print("=" * 70)

    # Convenience static wrapper around the overlap helper (same-video only).
    @staticmethod
    def time_overlap_ratio(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> float:
        return overlap_ratio(chunk_a, chunk_b)