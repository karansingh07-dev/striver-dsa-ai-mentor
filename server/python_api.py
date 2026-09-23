"""FastAPI wrapper around the existing Python RAG pipeline.

Reuses the existing RAGPipeline, retrieval, embedding, and reranker models.
Models are loaded once at startup, not per-request.

Run:
    uvicorn server.python_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the project root is importable when running uvicorn from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_TOP_K, YTRAG_MIN_SCORE  # noqa: E402
from rag.mentor_prompts import get_system_prompt, normalize_mode  # noqa: E402
from rag_engine import RAGPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PYTHON_API_HOST = os.getenv("PYTHON_API_HOST", "127.0.0.1")
PYTHON_API_PORT = int(os.getenv("PYTHON_API_PORT", "8000"))

# CORS: allow the Node backend and React dev server by default.
CORS_ORIGINS = os.getenv(
    "PYTHON_API_CORS_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("python_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    logger.info("Starting Python RAG API...")
    pipeline = RAGPipeline()
    app_state["rag_pipeline"] = pipeline
    logger.info("RAG pipeline initialized.")
    yield
    logger.info("Shutting down Python RAG API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Striver DSA AI Mentor - Python RAG API",
    description="Thin FastAPI wrapper around the existing Python RAG pipeline.",
    version="9.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    mode: Optional[str] = Field(
        "explain",
        description="Mentor mode: explain, hint, approach, code_review, quiz",
    )
    top_k: Optional[int] = Field(None, ge=1, le=20)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    conversation_id: Optional[str] = Field(None, description="Session identifier")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Recent conversation messages")


class SourceOut(BaseModel):
    title: str
    video_id: str
    start_sec: int
    timestamp: str
    url: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[SourceOut]
    conversation_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_timestamp(seconds: Any) -> str:
    try:
        secs = int(round(float(seconds)))
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"
    except (TypeError, ValueError):
        return "00:00"


def _transform_sources(raw_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in raw_sources:
        url = src.get("youtube_url") or ""
        if not url and src.get("video_id"):
            url = f"https://www.youtube.com/watch?v={src['video_id']}&t={int(src.get('start_sec', 0))}s"
        out.append(
            {
                "title": src.get("title", ""),
                "video_id": src.get("video_id", ""),
                "start_sec": int(src.get("start_sec", 0) or 0),
                "timestamp": _format_timestamp(src.get("start_sec", 0)),
                "url": url,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        logger.info("%s %s %d %dms", request.method, request.url.path, response.status_code, int(duration * 1000))
        return response
    except Exception as exc:
        duration = time.time() - start
        logger.error("%s %s 500 %dms error=%s", request.method, request.url.path, int(duration * 1000), str(exc))
        raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        normalized_mode = normalize_mode(req.mode or "explain")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mode. Allowed values: explain, hint, approach, code_review, quiz.")

    pipeline = app_state.get("rag_pipeline")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline is not initialized.")

    history = req.history or []
    try:
        result = pipeline.answer_question(
            query=query,
            top_k=req.top_k if req.top_k is not None else DEFAULT_TOP_K,
            min_score=req.min_score if req.min_score is not None else YTRAG_MIN_SCORE,
            debug=False,
            mode=normalized_mode,
            history=history,
        )
    except Exception as exc:
        logger.error("RAG pipeline error: %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline error. Please try again later.",
        ) from exc

    status = result.get("status")
    if status == "retrieval_error":
        raise HTTPException(
            status_code=500,
            detail="Retrieval error. Please try again later.",
        )
    if status == "llm_error":
        raise HTTPException(
            status_code=500,
            detail="LLM generation error. Please try again later.",
        )

    timing = result.get("timing") or {}
    if timing:
        logger.info(
            "RAG timings query='%s' retrieval_ms=%d llm_ms=%d total_ms=%d",
            query[:50],
            timing.get("retrieval_ms", 0),
            timing.get("llm_ms", 0),
            timing.get("total_ms", 0),
        )

    sources = _transform_sources(result.get("sources", []))
    return AskResponse(
        answer=result.get("answer", ""),
        grounded=bool(result.get("grounded", False)),
        sources=sources,
        conversation_id=req.conversation_id,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.python_api:app",
        host=PYTHON_API_HOST,
        port=PYTHON_API_PORT,
        reload=False,
        log_level="info",
    )
