"""evaluation package (Phase 6) — retrieval-quality + keyword answer evaluation.

    evaluation/questions.json - 14 ground-truth questions based ONLY on the
                               4 real indexed Striver lectures
    evaluation/metrics.py     - Recall@5 / Recall@10 / avg top score + helpers
    evaluation/runner.py      - runs --evaluate without an LLM judge
"""

from evaluation.metrics import (  # noqa: F401
    keywords_found_in_top,
    summarize_rows,
    top_score,
    video_in_top,
)
from evaluation.runner import load_questions, run_evaluation  # noqa: F401

__all__ = [
    "keywords_found_in_top",
    "summarize_rows",
    "top_score",
    "video_in_top",
    "load_questions",
    "run_evaluation",
]