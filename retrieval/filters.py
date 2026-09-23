"""Filtering helpers for retrieved chunks.

Score semantics: Qdrant uses cosine similarity, so HIGHER score = MORE relevant.
We always keep hits with score >= min_score — never invert the comparison.
"""

from typing import Any, Dict, List

# A chunk is a dict produced by QdrantVectorStore.search():
#   {score, title, start_time, start_sec, end_sec, text, youtube_url,
#    video_id, chunk_id}
Chunk = Dict[str, Any]


def apply_score_filter(
    hits: List[Chunk], min_score: float
) -> List[Chunk]:
    """Keeps only chunks whose cosine score is >= min_score (higher = better).

    min_score <= 0 disables the gate entirely (pure recall mode).
    """
    if min_score is None or float(min_score) <= 0.0:
        return list(hits)
    threshold = float(min_score)
    return [
        hit for hit in hits
        if float(hit.get("score", 0.0)) >= threshold
    ]


def overlap_ratio(chunk_a: Chunk, chunk_b: Chunk) -> float:
    """Fraction of the shorter chunk's duration shared with the other chunk.

    Only chunks from the SAME video can overlap meaningfully (they share the
    same timeline); different videos always return 0.0.
    """
    if chunk_a.get("video_id") != chunk_b.get("video_id"):
        return 0.0
    try:
        start_a = float(chunk_a.get("start_sec", 0.0))
        end_a = float(chunk_a.get("end_sec", 0.0))
        start_b = float(chunk_b.get("start_sec", 0.0))
        end_b = float(chunk_b.get("end_sec", 0.0))
    except (TypeError, ValueError):
        return 0.0

    dur_a = max(end_a - start_a, 0.0)
    dur_b = max(end_b - start_b, 0.0)
    if dur_a <= 0.0 or dur_b <= 0.0:
        return 0.0

    overlap = min(end_a, end_b) - max(start_a, start_b)
    if overlap <= 0.0:
        return 0.0
    # Relative to the SHORTEST chunk so a small chunk fully inside a big one
    # counts as a near-duplicate of the small one.
    return overlap / min(dur_a, dur_b)


def deduplicate_overlaps(
    hits: List[Chunk], threshold: float = 0.7
) -> List[Chunk]:
    """Removes near-duplicate time ranges from the same video, keeping the best.

    Greedy sweep: chunks are processed highest-score first. A chunk is dropped
    if it overlaps (ratio >= threshold) an already-accepted chunk from the same
    video — the accepted one has a higher or equal score by construction.

    Time-aware chunking overlaps adjacent chunks by ~15s / 75s = 0.2, so those
    legitimately distinct chunks survive; only true near-duplicates (>= 0.7 of
    the shorter duration in common) collapse.
    """
    if not hits:
        return []

    threshold = float(threshold) if threshold is not None else 0.7
    ordered = sorted(
        hits,
        key=lambda h: float(h.get("score", 0.0)),
        reverse=True,
    )

    kept: List[Chunk] = []
    for hit in ordered:
        is_duplicate = False
        for accepted in kept:
            if hit.get("video_id") and hit.get("video_id") == accepted.get("video_id"):
                if overlap_ratio(hit, accepted) >= threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            kept.append(hit)

    # Preserve original rank order (by score) at the end.
    return kept