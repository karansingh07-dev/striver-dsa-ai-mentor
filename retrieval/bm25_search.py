"""BM25 lexical retrieval over cached transcript chunks.

Builds a BM25 index from all JSON files in ``data/chunks/`` and persists it to
``data/search/bm25_index.pkl`` so repeated queries do not rebuild the index.

The returned hit metadata mirrors Qdrant semantic-search hits where possible so
the rest of the pipeline can treat both sources uniformly.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import BM25_INDEX_PATH, CHUNKS_DIR
from utils import load_json


@dataclass
class BM25Chunk:
    """Lightweight view over one chunk payload with stable metadata."""

    chunk_id: str
    video_id: str
    title: str
    start_sec: float
    end_sec: float
    start_time: str
    end_time: str
    text: str
    youtube_url: str
    segment_count: int = 0
    bm25_score: float = 0.0

    def to_hit(self) -> Dict[str, Any]:
        """Return a dict compatible with the existing retrieval hit schema."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "title": self.title,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "youtube_url": self.youtube_url,
            "segment_count": self.segment_count,
            "score": self.bm25_score,
            "source": "bm25",
        }


def _tokenize(text: str) -> List[str]:
    """Whitespace/punctuation tokenizer suitable for code/transcript text."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [tok.lower() for tok in cleaned.split() if tok]


def _load_chunk_documents(
    chunks_dir: Path = CHUNKS_DIR,
) -> Tuple[List[List[str]], List[BM25Chunk]]:
    """Read all chunk JSON files and return tokenized docs plus chunk metadata."""
    tokenized: List[List[str]] = []
    chunks: List[BM25Chunk] = []

    for path in sorted(chunks_dir.glob("*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        for c in data.get("chunks", []):
            cid = c.get("chunk_id")
            text = (c.get("text") or "").strip()
            if not cid or not text:
                continue
            chunks.append(
                BM25Chunk(
                    chunk_id=cid,
                    video_id=c.get("video_id", path.stem),
                    title=c.get("title", path.stem),
                    start_sec=float(c.get("start_sec", 0.0)),
                    end_sec=float(c.get("end_sec", 0.0)),
                    start_time=c.get("start_time", ""),
                    end_time=c.get("end_time", ""),
                    text=text,
                    youtube_url=c.get("youtube_url", ""),
                    segment_count=int(c.get("segment_count", 0)),
                )
            )
            tokenized.append(_tokenize(text))

    return tokenized, chunks


class BM25Index:
    """Persistent BM25 index over transcript chunks."""

    def __init__(self, index_path: Path = BM25_INDEX_PATH, chunks_dir: Path = CHUNKS_DIR):
        self.index_path = Path(index_path)
        self.chunks_dir = Path(chunks_dir)
        self.bm25: Any = None
        self.chunks: List[BM25Chunk] = []

    def build(self) -> None:
        """Build BM25 from chunk files and persist to disk."""
        tokenized, chunks = _load_chunk_documents(self.chunks_dir)
        if not tokenized:
            raise RuntimeError(f"No chunks found in '{self.chunks_dir}'.")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install it with: pip install rank-bm25"
            ) from exc

        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self, force_rebuild: bool = False) -> None:
        """Load persisted BM25 index or rebuild if missing/stale/forced."""
        if not force_rebuild and self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
                if self.bm25 is not None and self.chunks:
                    return
            except Exception:
                pass
        self.build()

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """Run BM25 search and return metadata-bearing hit dicts."""
        if self.bm25 is None or not self.chunks:
            self.load()
        if not query.strip():
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        k = max(int(k), 1)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            chunk.bm25_score = score
            results.append(chunk.to_hit())
        return results


class BM25Retriever:
    """Lazy-loaded singleton-style BM25 retriever."""

    def __init__(self, index_path: Path = BM25_INDEX_PATH, chunks_dir: Path = CHUNKS_DIR):
        self.index_path = Path(index_path)
        self.chunks_dir = Path(chunks_dir)
        self._index: Optional[BM25Index] = None

    def _ensure(self) -> BM25Index:
        if self._index is None:
            idx = BM25Index(index_path=self.index_path, chunks_dir=self.chunks_dir)
            idx.load()
            self._index = idx
        return self._index

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        return self._ensure().search(query, k=k)

    def reset(self) -> None:
        """Drop cached index so the next search rebuilds from disk."""
        self._index = None
        try:
            self.index_path.unlink()
        except FileNotFoundError:
            pass
