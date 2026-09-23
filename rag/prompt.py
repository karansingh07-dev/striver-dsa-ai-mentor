"""Prompts and response status constants for grounded RAG."""

from typing import Any, Dict, List, Optional

# System prompt: the grounding contract given to the LLM.
RAG_SYSTEM_PROMPT = (
    "You are an expert Data Structures & Algorithms (DSA) tutor based on "
    "Striver's A2Z DSA Course.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Answer ONLY using the Striver lecture transcript context provided below.\n"
    "2. Do NOT use outside general knowledge to fill gaps not covered in the context.\n"
    "3. If the context lacks enough information to answer, respond EXACTLY with:\n"
    "   \"I couldn't find enough information in the indexed Striver lectures to answer this question.\"\n"
    "4. Do NOT hallucinate, invent explanations, or attribute statements to Striver "
    "unless they are present in the context.\n"
    "5. Explain clearly and concisely in English or Hinglish, matching the user's language.\n"
    "6. Preserve DSA terminology: lower bound, upper bound, time complexity, recursion, "
    "base case, binary search, etc.\n"
    "7. Use examples or intuition ONLY from the retrieved context - never invented ones.\n"
    "8. NEVER invent YouTube URLs, timestamps, or source titles.\n"
)

# The exact phrase the model must use when the context is insufficient.
INSUFFICIENT_PHRASE = (
    "I couldn't find enough information in the indexed Striver lectures to answer this question."
)

# Response status constants.
STATUS_GROUNDED      = "grounded"         # retrieval + LLM succeeded, answer grounded
STATUS_INSUFFICIENT  = "insufficient"     # no relevant chunks (below score threshold)
STATUS_LLM_ERROR     = "llm_error"        # retrieval OK, but LLM call failed
STATUS_RETRIEVAL_ERR = "retrieval_error"  # Qdrant search itself failed


def build_user_prompt(query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Builds the user prompt = retrieved context + question + grounding instruction."""
    parts: List[str] = []
    if history:
        parts.append("PREVIOUS CONVERSATION:\n")
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        parts.append("\nEND OF CONVERSATION.\n")
    parts.append(
        "CONTEXT (Striver A2Z DSA lecture transcript chunks):\n\n"
        f"{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Answer the QUESTION using ONLY the CONTEXT above."
    )
    return "\n".join(parts)