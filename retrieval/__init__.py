"""Retrieval package (Phase 6/8) — two-stage retrieval + hybrid search.

Responsibilities owned here (kept out of main.py and rag_engine.py):

    RetrievalPipeline.retrieve(query)
      1. Stage-1 candidates : query embedding -> Qdrant -> up to YTRAG_RETRIEVAL_K hits
      2. Score gate         : keep hits with cosine score >= min_score (higher = better)
      3. Overlap dedup      : drop near-duplicate time ranges from the same video
      4. Top-K trim         : keep the best YTRAG_TOP_K chunks for the LLM context

    HybridRetriever.retrieve(query)
      1. Semantic retrieval : Qdrant -> YTRAG_RETRIEVAL_K hits
      2. BM25 retrieval     : keyword search -> YTRAG_BM25_K hits
      3. Hybrid fusion      : RRF -> YTRAG_HYBRID_K hits
      4. Re-ranking         : cross-encoder -> YTRAG_TOP_K hits
"""

from retrieval.bm25_search import BM25Retriever, BM25Index  # noqa: F401
from retrieval.filters import (  # noqa: F401
    apply_score_filter,
    deduplicate_overlaps,
    overlap_ratio,
)
from retrieval.hybrid_search import HybridRetriever, Reranker, reciprocal_rank_fusion  # noqa: F401
from retrieval.pipeline import RetrievalPipeline  # noqa: F401

__all__ = [
    "apply_score_filter",
    "deduplicate_overlaps",
    "overlap_ratio",
    "BM25Index",
    "BM25Retriever",
    "HybridRetriever",
    "Reranker",
    "reciprocal_rank_fusion",
    "RetrievalPipeline",
]
