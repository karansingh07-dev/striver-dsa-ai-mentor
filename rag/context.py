"""Context construction and source validation for grounded RAG.

Source validation rule (Phase 6):
    * video_id must be present and look like a YouTube ID
    * title must be a non-empty string
    * start_sec / end_sec must be valid, ordered, non-negative numbers
    * the youtube_url is ALWAYS regenerated from the stored metadata
      (video_id + start_sec) — it is never taken from the LLM.
    Chunks that fail validation are silently dropped from the source list.
"""

import re
from typing import Any, Dict, List, Optional

from utils import generate_youtube_timestamp_url

# YouTube video IDs are 11 characters ([A-Za-z0-9_-]); we accept a slightly
# wider 6-20 range so unusual-but-valid IDs are not rejected.
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


def build_context(hits: List[Dict[str, Any]]) -> str:
    """Builds the LLM context block from retrieved chunks (real transcript text)."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"Lecture: {hit['title']}\n"
            f"Timestamp: {hit['start_time']}\n"
            f"Text: {hit['text']}"
        )
    return "\n\n".join(blocks)


def validate_source(hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validates a retrieved chunk's source metadata; None if the chunk is unusable.

    Returns a normalized dict with video_id/title/start_sec/end_sec/chunk_id or
    None when any required field is missing/invalid.
    """
    video_id = str(hit.get("video_id", "") or "").strip()
    title = str(hit.get("title", "") or "").strip()

    if not video_id or not YOUTUBE_ID_RE.match(video_id):
        return None
    if not title:
        return None

    try:
        start_sec = int(hit.get("start_sec", -1))
        end_sec = int(hit.get("end_sec", -1))
    except (TypeError, ValueError):
        return None

    if start_sec < 0 or end_sec < 0 or end_sec < start_sec:
        return None

    return {
        "video_id": video_id,
        "title": title,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "chunk_id": str(hit.get("chunk_id", "") or ""),
    }


def build_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds validated source metadata, deduped by (video_id, start_sec, end_sec).

    The youtube_url is generated from the stored video_id + start_sec metadata.
    The LLM never supplies URLs; invalid chunks are dropped.
    """
    sources: List[Dict[str, Any]] = []
    seen = set()
    for hit in hits:
        src = validate_source(hit)
        if src is None:
            continue
        key = f"{src['video_id']}_{src['start_sec']}_{src['end_sec']}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": src["title"],
                "video_id": src["video_id"],
                "chunk_id": src["chunk_id"],
                "start_sec": src["start_sec"],
                "end_sec": src["end_sec"],
                "youtube_url": generate_youtube_timestamp_url(
                    src["video_id"], src["start_sec"]
                ),
            }
        )
    return sources