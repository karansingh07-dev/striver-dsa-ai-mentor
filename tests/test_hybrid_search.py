"""Phase 8 tests: BM25, RRF fusion, reranker, metadata preservation, fallback.

These tests use existing artifacts under ``data/chunks/`` and do not create
fake YouTube data.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pytest

# Ensure the project root is on the path when running pytest directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BM25_INDEX_PATH, CHUNKS_DIR, YTRAG_RRF_K
from retrieval.bm25_search import BM25Index, BM25Retriever, _tokenize
from retrieval.hybrid_search import (
    HybridRetriever,
    Reranker,
    reciprocal_rank_fusion,
    _hit_key,
    _normalize_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_hit(video_id: str, chunk_id: str, score: float = 1.0, **overrides) -> dict:
    hit = {
        "chunk_id": chunk_id,
        "video_id": video_id,
        "title": f"Lecture {video_id}",
        "start_sec": 0.0,
        "end_sec": 10.0,
        "start_time": "00:00:00",
        "end_time": "00:00:10",
        "text": "lower bound binary search BFS DFS O(log n)",
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t=0s",
        "score": score,
    }
    hit.update(overrides)
    return hit


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_lowercase(self):
        tokens = _tokenize("Binary Search Lower Bound")
        assert tokens == ["binary", "search", "lower", "bound"]

    def test_strips_punctuation(self):
        tokens = _tokenize("O(log n); BFS, DFS.")
        assert "o" in tokens
        assert "log" in tokens
        assert "bfs" in tokens
        assert "dfs" in tokens
        assert ";" not in tokens
        assert "(" not in tokens

    def test_empty(self):
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_basic_fusion(self):
        a = [_sample_hit("v1", "c1", score=0.9), _sample_hit("v1", "c2", score=0.8)]
        b = [_sample_hit("v1", "c2", score=0.7), _sample_hit("v1", "c3", score=0.6)]
        out = reciprocal_rank_fusion(a, b, k=60)
        keys = [_hit_key(h) for h in out]
        assert keys[0] == _hit_key(_sample_hit("v1", "c2"))
        assert keys[1] == _hit_key(_sample_hit("v1", "c1"))
        assert keys[2] == _hit_key(_sample_hit("v1", "c3"))

    def test_unique_chunks_preserved(self):
        a = [_sample_hit("v1", "c1")]
        b = [_sample_hit("v1", "c2")]
        out = reciprocal_rank_fusion(a, b, k=60)
        assert len(out) == 2

    def test_scores_preserved(self):
        a = [_sample_hit("v1", "c1", score=0.9)]
        b = [_sample_hit("v1", "c2", score=0.5)]
        out = reciprocal_rank_fusion(a, b, k=60)
        assert len(out) == 2
        # Normalized scores may be 0.0 for single-item lists, but keys must exist.
        assert "semantic_score" in out[0] or "bm25_score" in out[0]
        assert "hybrid_score" in out[0]

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], [], k=60) == []
        out = reciprocal_rank_fusion([_sample_hit("v1", "c1")], [], k=60)
        assert len(out) == 1
        assert out[0]["chunk_id"] == "c1"
        assert "hybrid_score" in out[0]


# ---------------------------------------------------------------------------
# BM25 index (uses real chunks)
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_build_and_search(self):
        retriever = BM25Retriever()
        retriever.reset()
        hits = retriever.search("binary search", k=5)
        assert isinstance(hits, list)
        if hits:
            assert "text" in hits[0]
            assert "video_id" in hits[0]
            assert "chunk_id" in hits[0]

    def test_persistent_index(self):
        retriever = BM25Retriever()
        retriever.reset()
        first = retriever.search("merge sort", k=3)
        # Second load should hit the persisted pickle.
        retriever2 = BM25Retriever()
        second = retriever2.search("merge sort", k=3)
        assert len(first) == len(second)

    def test_metadata_preserved(self):
        retriever = BM25Retriever()
        hits = retriever.search("BFS", k=1)
        if hits:
            hit = hits[0]
            assert "start_time" in hit
            assert "end_time" in hit
            assert "youtube_url" in hit
            assert "title" in hit


# ---------------------------------------------------------------------------
# Hybrid retriever (integration-style, no network beyond Qdrant)
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def test_hybrid_returns_final(self):
        retriever = HybridRetriever()
        result = retriever.retrieve("binary search lower bound")
        assert "final" in result
        assert "semantic" in result
        assert "bm25" in result
        assert "hybrid" in result
        assert "reranked" in result

    def test_final_chunks_have_metadata(self):
        retriever = HybridRetriever()
        result = retriever.retrieve("merge sort time complexity")
        for hit in result.get("final", []):
            assert "video_id" in hit
            assert "chunk_id" in hit
            assert "text" in hit
            assert "start_sec" in hit
            assert "end_sec" in hit
            assert "youtube_url" in hit
            assert "title" in hit

    def test_fallback_on_total_failure(self):
        retriever = HybridRetriever()
        # Empty query should produce no hits and fallback flag.
        result = retriever.retrieve("")
        assert result.get("fallback") is True or len(result.get("final", [])) == 0

    def test_duplicate_removal_across_sources(self):
        # Construct two identical hits from different sources.
        hit = _sample_hit("v1", "c1", score=0.9)
        out = reciprocal_rank_fusion([hit], [hit], k=60)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Reranker (smoke test; skips if model download is unavailable)
# ---------------------------------------------------------------------------

class TestReranker:
    def test_rerank_changes_order(self):
        try:
            reranker = Reranker()
        except Exception as exc:
            pytest.skip(f"Reranker unavailable: {exc}")
        hits = [
            _sample_hit("v1", "c1", text="irrelevant python tutorial"),
            _sample_hit("v1", "c2", text="binary search lower bound algorithm"),
        ]
        out = reranker.rerank("binary search lower bound", hits, top_n=2)
        assert out[0]["chunk_id"] == "c2"
