"""Evaluation runner — retrieval quality + keyword-based answer evaluation.

Runs the two-stage RetrievalPipeline against the questions in
evaluation/questions.json and reports, per question:
    * top result title + score
    * expected video retrieved in top-5       (YES/NO)
    * expected keywords found in top-5        (YES/NO)
    * grounded at retrieval level             (YES/NO)

Summary metrics (computed from actual results, never faked):
    Questions                : N
    Expected video retrieved : X / N
    Recall@5, Recall@10      : rates
    Average top score        : mean of each question's best cosine score

No LLM is called and no LLM judge exists — this measures the RETRIEVAL stage,
which is exactly what we are improving in Phase 6.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DEFAULT_TOP_K, YTRAG_MIN_SCORE, YTRAG_RETRIEVAL_K
from evaluation.metrics import (
    keywords_found_in_top,
    summarize_rows,
    top_score,
    video_in_top,
)
from retrieval.pipeline import RetrievalPipeline
from utils import format_seconds_to_timestamp

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.json"


def load_questions(path: Path = QUESTIONS_FILE) -> List[Dict[str, Any]]:
    """Loads the evaluation dataset from questions.json (validates required keys)."""
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"questions.json must be a non-empty list. Check {path}")

    required = {"id", "question", "expected_topic", "expected_video_id", "expected_keywords"}
    for q in questions:
        missing = required - set(q.keys())
        if missing:
            raise ValueError(
                f"Question {q.get('id', '?')} is missing required fields: {sorted(missing)}"
            )
    return questions


def evaluate_question(
    question: Dict[str, Any],
    pipeline: RetrievalPipeline,
    top_k: int,
    min_score: float,
) -> Dict[str, Any]:
    """Runs one question through the pipeline and scores it against expectations."""
    retrieval = pipeline.retrieve(question["question"], top_k=top_k, min_score=min_score)
    candidates = retrieval.get("candidates", [])
    final = retrieval.get("final", [])
    expected_video = str(question.get("expected_video_id", ""))

    retrieved_top5 = video_in_top(final, expected_video, k=top_k)
    retrieved_top10 = video_in_top(candidates, expected_video, k=10)
    expected_keywords = list(question.get("expected_keywords", []))
    kw_found = keywords_found_in_top(final, expected_keywords, k=top_k)
    best_score = top_score(candidates)
    grounded = bool(
        best_score >= min_score and retrieved_top5 and kw_found
    )

    return {
        "id": question.get("id"),
        "question": question.get("question"),
        "expected_topic": question.get("expected_topic"),
        "expected_video_id": expected_video,
        "top_hit_title": (candidates[0].get("title", "(none)") if candidates else "(none)"),
        "top_score": best_score,
        "retrieved_top5": retrieved_top5,
        "retrieved_top10": retrieved_top10,
        "keywords_found": kw_found,
        "grounded": grounded,
        "final_count": len(final),
        "candidate_count": len(candidates),
        "retrieval": retrieval,
    }
def run_evaluation(
    limit: Optional[int] = None,
    debug: bool = False,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> None:
    """Runs the full keyword-based retrieval-quality evaluation and prints it."""
    questions = load_questions()
    if limit and limit > 0:
        questions = questions[:limit]

    pipeline = RetrievalPipeline(
        retrieval_k=YTRAG_RETRIEVAL_K,
        top_k=top_k or DEFAULT_TOP_K,
    )
    min_score_use = float(min_score if min_score is not None else YTRAG_MIN_SCORE)
    effective_top_k = int(top_k or DEFAULT_TOP_K)

    print("\n" + "=" * 78)
    print("RETRIEVAL QUALITY EVALUATION  (keyword-based, no LLM judge)")
    print("=" * 78)
    print(f"Dataset        : {QUESTIONS_FILE.name} ({len(questions)} questions)")
    print(f"Candidate pool : {pipeline.retrieval_k} (retrieval_k)")
    print(f"Final context  : {effective_top_k} chunks (top_k)")
    print(f"Score gate     : >= {min_score_use}")
    print(f"Overlap dedup  : >= {pipeline.dedup_overlap_ratio} (same video)")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    for i, question in enumerate(questions, start=1):
        row = evaluate_question(question, pipeline, effective_top_k, min_score_use)
        rows.append(row)
        _print_row(row, index=i)

        if debug:
            pipeline.print_report(row["retrieval"])
            print()

    summary = summarize_rows(rows)
    _print_summary(summary)
    print("=" * 78 + "\n")


def _print_row(row: Dict[str, Any], index: int) -> None:
    ts = ""
    top_hit = row["retrieval"]["candidates"][0] if row["retrieval"]["candidates"] else {}
    if top_hit:
        ts = format_seconds_to_timestamp(top_hit.get("start_sec", 0))
    yes = "YES " if row["retrieved_top5"] else "NO  "

    print()
    print(f"[{index:>2}] {row['id']}  {row['question']}")
    print(f"      Expected topic/video : {row['expected_topic']}  ({row['expected_video_id']})")
    print(
        f"      Top result           : '{row['top_hit_title']}'  "
        f"{row['top_score']:.4f}  ts={ts}"
    )
    print(
        f"      Expected video (top-5): {yes}  "
        f"|  Keywords found (top-5): {yes if row['keywords_found'] else 'NO  '}  "
        f"|  Grounded: {yes if row['grounded'] else 'NO  '}"
    )


def _print_summary(s: Dict[str, Any]) -> None:
    print()
    print("-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(f"Questions evaluated             : {s['questions']}")
    print(
        f"Expected video retrieved (top-5): {s['expected_video_retrieved']} / "
        f"{s['questions']}"
    )
    print(f"Recall@5                        : {s['recall_at_5']:.2%}")
    print(f"Recall@10                       : {s['recall_at_10']:.2%}")
    print(f"Average top-1 score             : {s['avg_top_score']:.4f}")
    print(
        f"Keywords found (top-5)          : {s['keywords_found']} / "
        f"{s['questions']}  ({s['keywords_found_rate']:.2%})"
    )
    print(
        f"Grounded (retrieval level)      : {s['grounded']} / "
        f"{s['questions']}  ({s['grounded_rate']:.2%})"
    )
    print("-" * 78)