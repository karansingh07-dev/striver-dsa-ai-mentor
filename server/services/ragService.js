const PYTHON_RAG_URL = process.env.PYTHON_RAG_URL || "http://127.0.0.1:8000";
const REQUEST_TIMEOUT = Number(process.env.PYTHON_REQUEST_TIMEOUT || 120_000);

const ALLOWED_MODES = new Set(["explain", "hint", "approach", "code_review", "quiz"]);
const MAX_MESSAGE_LENGTH = 2000;

function validateChatRequest(body) {
  const message = (body && body.message) || "";
  const mode = (body && body.mode) || "explain";

  if (!message || !message.trim()) {
    return { error: "Message must not be empty.", status: 400 };
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    return { error: "Message is too long. Please shorten your question.", status: 400 };
  }
  const normalizedMode = mode.trim().toLowerCase();
  if (!ALLOWED_MODES.has(normalizedMode)) {
    return { error: `Invalid mode '${mode}'. Allowed values: ${Array.from(ALLOWED_MODES).join(", ")}.`, status: 400 };
  }
  return { message: message.trim(), mode: normalizedMode };
}

function validateRagResponse(data) {
  if (!data || typeof data !== "object") {
    return { error: "Invalid response from RAG service.", status: 502 };
  }
  if (typeof data.answer !== "string") {
    return { error: "Invalid response from RAG service: missing answer.", status: 502 };
  }
  if (!Array.isArray(data.sources)) {
    return { error: "Invalid response from RAG service: sources must be an array.", status: 502 };
  }
  return null;
}

export async function callPythonRag(query, mode = "explain", options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  try {
    const body = {
      query,
      mode,
      conversation_id: options.conversationId || null,
      history: options.history || [],
    };

    const res = await fetch(`${PYTHON_RAG_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      let detail = text;
      try {
        const parsed = JSON.parse(text);
        detail = parsed.detail || detail;
      } catch {
        // keep raw text
      }
      throw new Error(detail || `Python API returned ${res.status}`);
    }

    const data = await res.json();
    const validationError = validateRagResponse(data);
    if (validationError) {
      throw new Error(validationError.error);
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

export { validateChatRequest, ALLOWED_MODES, MAX_MESSAGE_LENGTH };
