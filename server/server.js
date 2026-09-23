import "dotenv/config";
import express from "express";
import cors from "cors";
import chatRouter from "./routes/chat.js";
import {
  createConversation,
  deleteConversation,
  getConversation,
} from "./services/conversationStore.js";

const app = express();
const PORT = process.env.PORT || 5000;

const allowedOrigins = (process.env.FRONTEND_URL || "http://localhost:5173")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error("Not allowed by CORS"));
      }
    },
    credentials: true,
  })
);

app.use(express.json());

app.get("/api/health", async (req, res) => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const pythonRes = await fetch(`${process.env.PYTHON_RAG_URL || "http://127.0.0.1:8000"}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (pythonRes.ok) {
      res.json({ status: "ok", python: { reachable: true } });
    } else {
      res.status(500).json({ status: "degraded", python: { reachable: false, status: pythonRes.status } });
    }
  } catch (error) {
    res.status(500).json({ status: "degraded", python: { reachable: false, error: "Python RAG service unreachable" } });
  }
});

app.post("/api/conversations", (req, res) => {
  try {
    const mode = (req.body && req.body.mode) || "explain";
    const convo = createConversation(mode);
    res.status(201).json(convo);
  } catch (error) {
    console.error("[NodeBackend] Failed to create conversation:", error);
    res.status(500).json({ error: "Failed to create conversation." });
  }
});

app.get("/api/conversations/:id", (req, res) => {
  const convo = getConversation(req.params.id);
  if (!convo) {
    return res.status(404).json({ error: "Conversation not found." });
  }
  res.json(convo);
});

app.delete("/api/conversations/:id", (req, res) => {
  deleteConversation(req.params.id);
  res.status(204).send();
});

app.use("/api/chat", chatRouter);

app.listen(PORT, () => {
  console.log(`Node backend listening on http://localhost:${PORT}`);
});
