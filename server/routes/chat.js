import express from "express";
import { callPythonRag, validateChatRequest } from "../services/ragService.js";
import {
  getOrCreateConversation,
  updateConversation,
  buildHistory,
} from "../services/conversationStore.js";

const router = express.Router();

function sanitizeForLog(message) {
  return message.length > 80 ? message.slice(0, 77) + "..." : message;
}

function updateConversationState(conversation, userMessage, assistantResponse) {
  const mode = conversation.mode;
  const state = { ...conversation.state };

  if (!Array.isArray(conversation.messages)) {
    conversation.messages = [];
  }

  const newMessages = [
    ...conversation.messages,
    { role: "user", content: userMessage, mode },
    { role: "assistant", content: assistantResponse },
  ];

  if (mode === "hint") {
    state.hintCount = (state.hintCount || 0) + 1;
    state.topic = state.topic || userMessage;
  } else if (mode === "quiz") {
    state.topic = state.topic || userMessage.replace(/quiz me on /i, "").trim();
    state.questionNumber = (state.questionNumber || 0) + 1;
    if (assistantResponse && assistantResponse.answer) {
      const answerText = assistantResponse.answer.toLowerCase();
      if (answerText.includes("correct")) {
        state.score = (state.score || 0) + 1;
      }
    }
  } else if (mode === "code_review") {
    if (userMessage.includes("```") || userMessage.includes("int ") || userMessage.includes("void ")) {
      state.code = userMessage;
    }
  } else if (mode === "explain") {
    state.topic = state.topic || userMessage;
  }

  return updateConversation(conversation.id, {
    messages: newMessages,
    state,
  });
}

router.post("/", async (req, res) => {
  const start = Date.now();
  const validation = validateChatRequest(req.body);
  if (validation.error) {
    console.log("[NodeBackend] Bad request:", validation.error);
    return res.status(validation.status).json({ error: validation.error });
  }

  const { message, mode } = validation;
  let conversationId = req.body.conversationId;
  const conversation = getOrCreateConversation(conversationId, mode);
  conversationId = conversation.id;

  console.log("[NodeBackend] Request received mode=%s message=%s conversation=%s", mode, sanitizeForLog(message), conversationId);

  const history = buildHistory(conversation);

  try {
    const data = await callPythonRag(message, mode, {
      conversationId,
      history,
    });

    updateConversationState(conversation, message, data);

    const duration = Date.now() - start;
    console.log(
      "[NodeBackend] Request completed mode=%s duration_ms=%d grounded=%s sources=%d conversation=%s",
      mode,
      duration,
      data.grounded ? "yes" : "no",
      Array.isArray(data.sources) ? data.sources.length : 0,
      conversationId
    );

    res.json({ ...data, conversationId });
  } catch (error) {
    const duration = Date.now() - start;
    console.error("[NodeBackend] Request failed mode=%s duration_ms=%d error=%s conversation=%s", mode, duration, JSON.stringify(error).slice(0, 200), conversationId);
    res.status(500).json({
      error: "RAG service unavailable. Please try again later.",
    });
  }
});

export default router;
