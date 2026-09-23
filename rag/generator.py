"""LLM generation dispatch for the grounded RAG pipeline.

Supports groq | gemini | openai, selected via LLM_PROVIDER. The default model
per provider comes from DEFAULT_MODELS (LLM_MODEL env var overrides it).

Contract:
    * raises RuntimeError on misconfiguration or API failure
      (a real llm_error — never \"insufficient context\")
    * never silently falls back to another provider
    * never prints or logs the API key
"""

import os

from config import GEMINI_API_KEY, GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY

# Default LLM model per provider (used when LLM_MODEL env var is empty).
# Groq previously shipped 'llama-3.3-70b-versatile' but retired it (model_not_found).
# Pick a current model from your Groq account: e.g. openai/gpt-oss-20b,
# qwen/qwen3.8-27b, groq/compound-mini.
DEFAULT_MODELS = {
    "groq":   "openai/gpt-oss-20b",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}


class LLMGenerator:
    """Sends (user_prompt, system_prompt) to the configured LLM provider."""

    def __init__(self, provider: str = "", model: str = "") -> None:
        self.provider = (provider or LLM_PROVIDER).strip().lower()
        self.model = (
            (model or LLM_MODEL)
            or DEFAULT_MODELS.get(self.provider, "")
        ).strip()

    def generate(self, user_prompt: str, system_prompt: str) -> str:
        """Returns the assistant's answer text; raises RuntimeError on any failure."""
        if not self.model:
            raise RuntimeError(
                f"LLM provider: {self.provider}\n"
                f"Error: LLM_MODEL is not configured for provider '{self.provider}'.\n"
                f"Set LLM_MODEL in your .env file or update DEFAULT_MODELS."
            )

        if self.provider == "groq":
            return self._call_groq(system_prompt, user_prompt, self.model)
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt, self.model)
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, self.model)

        raise RuntimeError(
            f"LLM provider: {self.provider}\n"
            f"Error: Unknown LLM_PROVIDER '{self.provider}'. "
            f"Supported values: groq, gemini, openai."
        )

    # ---- Groq ---------------------------------------------------------------
    def _call_groq(self, system_prompt: str, user_prompt: str, model: str) -> str:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "LLM provider: groq\n"
                "Error: GROQ_API_KEY is not configured.\n"
                "Add  GROQ_API_KEY=<your-key>  to your .env file.\n"
                "Free keys available at https://console.groq.com/keys"
            )
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"LLM provider: groq\nError: Groq API call failed - {e}")

    # ---- Gemini --------------------------------------------------------------
    def _call_gemini(self, system_prompt: str, user_prompt: str, model: str) -> str:
        api_key = (
            GEMINI_API_KEY
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )
        if not api_key:
            raise RuntimeError(
                "LLM provider: gemini\n"
                "Error: GEMINI_API_KEY is not configured.\n"
                "Add  GEMINI_API_KEY=<your-key>  to your .env file.\n"
                "Free keys available at https://aistudio.google.com/apikey"
            )
        try:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            raise RuntimeError(
                f"LLM provider: gemini\nError: Gemini API call failed - {e}"
            )

    # ---- OpenAI --------------------------------------------------------------
    def _call_openai(self, system_prompt: str, user_prompt: str, model: str) -> str:
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "LLM provider: openai\n"
                "Error: OPENAI_API_KEY is not configured.\n"
                "Add  OPENAI_API_KEY=<your-key>  to your .env file.\n"
                "Keys available at https://platform.openai.com/api-keys"
            )
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(
                f"LLM provider: openai\nError: OpenAI API call failed - {e}"
            )