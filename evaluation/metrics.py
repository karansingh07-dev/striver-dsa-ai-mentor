"""Metric helpers for retrieval-quality evaluation (no LLM judge).

All metrics are computed from the ACTUAL retrieval pipeline results — nothing
is faked or assumed. Semantics:

    Recall@k        : fraction of questions whose expected_video_id appears in
                      the top-k retrieved results (k = final top-5 chunks for
                      Recall@5; k = 10 earliest candidates for Recall@10).
    Expected video  : same as Recall@k for k = 5 (the shipped default top-k),
                      reported as "X / N" alongside the percentage.
    Avg top score   : mean of each question's best (top-1 candidate) cosine score.
    Keywords found  : a question is a keyword hit only when EVERY expected
                      keyword appears in the concatenated top-5 chunk texts.
    Grounded        : retrieval-level grounding = expected video in top-5 AND
                      top score >= min_score AND all keywords found.
"""

from typing import Any, Dict, List


def video_in_top(
    hits: List[Dict[str, Any]], expected_video_id: str, k: int
) -> bool:
    """True if expected_video_id appears among the first k hits."""
    if k <= 0:
        return False
    video_ids = {
        str(hit.get("video_id", "")) for hit in hits[:k]
    }
    return expected_video_id in video_ids


def keywords_found_in_top(
    hits: List[Dict[str, Any]],
    keywords: List[str],
    k: int = 5,
) -> bool:
    """True when EVERY expected keyword occurs in the top-k chunk texts."""
    if not keywords:
        return True
    if not hits:
        return False
    texts = " ".join(str(hit.get("text", "")) for hit in hits[:k]).lower()
    lowered = set()
    for kw in keywords:
        lowered.add(kw)
    return all(kw.lower() in texts for kw in lowered)


def top_score(hits: List[Dict[str, Any]]) -> float:
    """Best (highest) cosine score across the retrieved hits, else 0.0."""
    if not hits:
        return 0.0
    return max(float(hit.get("score", 0.0)) for hit in hits)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-question rows into the final summary metrics."""
    n = len(rows)
    if n == 0:
        return {
            "questions": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "avg_top_score": 0.0,
            "keywords_found": 0.0,
            "grounded": 0.0,
        }

    retrieved_at_5 = sum(1 for r in rows if r["retrieved_top5"])
    retrieved_at_10 = sum(1 for r in rows if r["retrieved_top10"])
    keyword_hits = sum(1 for r in rows if r["keywords_found"])
    grounded_hits = sum(1 for r in rows if r["grounded"])

    scores = [r["top_score"] for r in rows if r["top_score"] > 0.0]
    avg_top = (sum(scores) / len(scores)) if scores else 0.0

    return {
        "questions": n,
        "expected_video_retrieved": retrieved_at_5,
        "recall_at_5": retrieved_at_5 / n,
        "recall_at_10": retrieved_at_10 / n,
        "avg_top_score": avg_top,
        "keywords_found": keyword_hits,
        "keywords_found_rate": keyword_hits / n,
        "grounded": grounded_hits,
        "grounded_rate": grounded_hits / n,
    }