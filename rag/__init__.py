"""rag package — grounded-RAG building blocks (context, prompts, LLM dispatch).

    rag/context.py   - context construction + source metadata validation
    rag/prompt.py    - system prompt, user-prompt builder, status constants
    rag/generator.py - Groq / Gemini / OpenAI generation dispatch
"""

from rag.context import build_context, build_sources, validate_source  # noqa: F401
from rag.generator import DEFAULT_MODELS, LLMGenerator  # noqa: F401
from rag.prompt import (  # noqa: F401
    INSUFFICIENT_PHRASE,
    RAG_SYSTEM_PROMPT,
    STATUS_GROUNDED,
    STATUS_INSUFFICIENT,
    STATUS_LLM_ERROR,
    STATUS_RETRIEVAL_ERR,
    build_user_prompt,
)

__all__ = [
    "build_context",
    "build_sources",
    "validate_source",
    "DEFAULT_MODELS",
    "LLMGenerator",
    "INSUFFICIENT_PHRASE",
    "RAG_SYSTEM_PROMPT",
    "STATUS_GROUNDED",
    "STATUS_INSUFFICIENT",
    "STATUS_LLM_ERROR",
    "STATUS_RETRIEVAL_ERR",
    "build_user_prompt",
]