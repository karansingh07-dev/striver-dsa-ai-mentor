"""rag_engine.py - Grounded RAG answering pipeline for the Striver DSA AI Mentor.

Phase 8 pipeline:
    Question -> HybridRetriever (semantic + BM25 -> RRF -> reranker -> top-k)
    -> Context construction -> LLM -> Grounded answer + validated real sources

Fallback: if hybrid retrieval fails entirely, falls back to the legacy
RetrievalPipeline semantic search so --ask never crashes.

Responsibility split (kept out of main.py):
    retrieval/   - query embedding + Qdrant candidates, score filtering, overlap
                    deduplication, top-k trimming (RetrievalPipeline)
    retrieval/   - BM25 lexical search, RRF fusion, cross-encoder reranking (HybridRetriever)
    rag/         - context construction, source validation, prompts, LLM dispatch
    evaluation/  - keyword-based retrieval quality evaluation (no LLM judge)

Status semantics:
    retrieval_error -> Qdrant/embedding failure (REAL problem, not a refusal)
    insufficient    -> retrieval worked but nothing is relevant enough (same
                        phrase as when the LLM declares the context insufficient)
    llm_error       -> retrieval OK but the LLM call/config failed (REAL problem)
    grounded        -> answer produced strictly from the retrieved context
"""

from typing import Any, Dict, List, Optional

import time
from config import DEFAULT_TOP_K, YTRAG_MIN_SCORE
from rag.context import build_context, build_sources
from rag.generator import LLMGenerator
from rag.mentor_prompts import get_system_prompt, normalize_mode
from rag.prompt import (
    INSUFFICIENT_PHRASE,
    RAG_SYSTEM_PROMPT,
    STATUS_GROUNDED,
    STATUS_INSUFFICIENT,
    STATUS_LLM_ERROR,
    STATUS_RETRIEVAL_ERR,
    build_user_prompt,
)
from retrieval.hybrid_search import HybridRetriever
from retrieval.pipeline import RetrievalPipeline
from vector_store import QdrantVectorStore

# Backward-compatible module-level re-exports (existing imports still work).
__all__ = [
    "RAGPipeline",
    "RAG_SYSTEM_PROMPT",
    "INSUFFICIENT_PHRASE",
    "STATUS_GROUNDED",
    "STATUS_INSUFFICIENT",
    "STATUS_LLM_ERROR",
    "STATUS_RETRIEVAL_ERR",
]


class RAGPipeline:
    """Question -> retrieval/ -> rag/ -> grounded answer + validated sources.

    Response dict always contains:
        status   (str)  : one of the STATUS_* constants above
        answer   (str)  : human-readable answer or error message
        grounded (bool) : True only when status == STATUS_GROUNDED
        sources  (list) : real, validated Qdrant payload metadata, never invented
    """

    def __init__(self, vector_store: Optional[QdrantVectorStore] = None):
        self.vector_store = vector_store
        self.generator = LLMGenerator()
        self._hybrid: Optional[HybridRetriever] = None
        self._legacy: Optional[RetrievalPipeline] = None

    def _get_hybrid(self) -> HybridRetriever:
        if self._hybrid is None:
            self._hybrid = HybridRetriever()
        return self._hybrid

    def _get_legacy(self) -> RetrievalPipeline:
        if self._legacy is None:
            self._legacy = RetrievalPipeline(vector_store=self.vector_store)
        return self._legacy

    # ------------------------------------------------------------------
    # Main entry point.
    #   Retrieval exception     -> STATUS_RETRIEVAL_ERR (real error)
    #   No chunks above score   -> STATUS_INSUFFICIENT   (genuine refusal)
    #   LLM config / API error  -> STATUS_LLM_ERROR      (real error)
    #   LLM answered from hits  -> STATUS_GROUNDED
    #   Hybrid fallback         -> STATUS_RETRIEVAL_ERR with note if BOTH fail
    # ------------------------------------------------------------------
    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        debug: bool = False,
        mode: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return _insufficient_result("Please provide a non-empty question.")

        top_k_use = max(int(top_k if top_k is not None else DEFAULT_TOP_K), 1)
        min_score_use = (
            float(min_score) if min_score is not None else float(YTRAG_MIN_SCORE)
        )

        try:
            normalized_mode = normalize_mode(mode or "explain")
        except ValueError:
            normalized_mode = "explain"
        system_prompt = get_system_prompt(normalized_mode)

        # 1) Hybrid retrieval (semantic + BM25 + fusion + rerank).
        hybrid_error: Optional[str] = None
        hybrid_result: Optional[Dict[str, Any]] = None
        retrieval_start = time.time()
        try:
            hybrid_result = self._get_hybrid().retrieve(query)
        except Exception as exc:
            hybrid_error = str(exc)
        retrieval_time = time.time() - retrieval_start

        if hybrid_result and hybrid_result.get("final"):
            hits: List[Dict[str, Any]] = hybrid_result["final"]
            retrieval_meta = hybrid_result
        else:
            # 2) Graceful fallback to legacy semantic retrieval.
            try:
                legacy = self._get_legacy()
                legacy_result = legacy.retrieve(
                    query,
                    top_k=top_k_use,
                    min_score=min_score_use,
                )
                hits = legacy_result.get("final", [])
                retrieval_meta = legacy_result
            except Exception as exc:
                return {
                    "status": STATUS_RETRIEVAL_ERR,
                    "answer": (
                        "Retrieval error: hybrid retrieval failed"
                        + (f" ({hybrid_error})" if hybrid_error else "")
                        + f" and legacy semantic search also failed - {exc}"
                    ),
                    "grounded": False,
                    "sources": [],
                }

        if not hits:
            return _insufficient_result(INSUFFICIENT_PHRASE)

        # 3) Context + validated sources (built from the final top-k chunks).
        context = build_context(hits)
        sources = build_sources(hits)
        user_prompt = build_user_prompt(query, context, history=history)

        if debug:
            if hybrid_result:
                HybridRetriever.print_report(hybrid_result)
            else:
                RetrievalPipeline.print_report(retrieval_meta)
            print("\nCONTEXT SENT TO LLM:")
            print("-" * 60)
            print(context)
            print("-" * 60)

        # 4) LLM generation.
        llm_start = time.time()
        try:
            answer = self.generator.generate(user_prompt, system_prompt)
        except RuntimeError as exc:
            return {
                "status": STATUS_LLM_ERROR,
                "answer": f"LLM generation error: {exc}",
                "grounded": False,
                "sources": sources,
            }
        llm_time = time.time() - llm_start

        total_time = time.time() - retrieval_start

        # 5) Post-check: if the LLM itself decided the context was lacking, respect it.
        if INSUFFICIENT_PHRASE.lower() in answer.lower():
            return _insufficient_result(INSUFFICIENT_PHRASE)

        return {
            "status": STATUS_GROUNDED,
            "answer": answer,
            "grounded": True,
            "sources": sources,
            "timing": {
                "retrieval_ms": int(retrieval_time * 1000),
                "llm_ms": int(llm_time * 1000),
                "total_ms": int(total_time * 1000),
            },
        }


def _insufficient_result(message: str) -> Dict[str, Any]:
    """Shared builder for genuine refusals (retrieval worked, context missing)."""
    return {
        "status": STATUS_INSUFFICIENT,
        "answer": message,
        "grounded": False,
        "sources": [],
    }
