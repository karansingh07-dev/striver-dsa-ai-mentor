import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_BASE = "/api";
const MENTOR_MODES = [
  { value: "explain", label: "Explain" },
  { value: "hint", label: "Hint" },
  { value: "approach", label: "Approach" },
  { value: "code_review", label: "Code Review" },
  { value: "quiz", label: "Quiz" },
];

function generateId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("explain");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [conversationId, setConversationId] = useState(() => generateId());
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function startNewChat() {
    const newId = generateId();
    setConversationId(newId);
    setMessages([]);
    setError("");
    setInput("");
  }

  async function sendMessage(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const userMessage = { role: "user", content: text, mode };
    setMessages((m) => [...m, userMessage]);
    setInput("");
    setLoading(true);
    setError("");

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 180_000);

      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, mode, conversationId }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      if (data.conversationId) {
        setConversationId(data.conversationId);
      }
      setMessages((m) => [...m, { role: "assistant", content: data }]);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-top">
          <h1>🧠 Striver DSA AI Mentor</h1>
          <button className="new-chat-button" onClick={startNewChat} disabled={loading}>
            + New Chat
          </button>
        </div>
        <div className="mode-bar">
          <label htmlFor="mentor-mode">Mode:</label>
          <select
            id="mentor-mode"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            disabled={loading}
          >
            {MENTOR_MODES.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask any DSA question from the Striver A2Z course.</p>
            <div className="examples">
              <span>Try:</span>
              <button
                onClick={() => setInput("binary search lower bound explain karo")}
              >
                binary search lower bound
              </button>
              <button
                onClick={() =>
                  setInput("merge sort ki time complexity kya hai?")
                }
              >
                merge sort time complexity
              </button>
              <button
                onClick={() => setInput("sliding window kya hota hai?")}
              >
                sliding window
              </button>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="bubble">
              <strong>{msg.role === "user" ? "You" : "AI"}:</strong>
              {msg.role === "user" ? (
                <p>{msg.content}</p>
              ) : (
                <div className="ai-content">
                  <div className="answer">
                    <ReactMarkdown>{msg.content.answer}</ReactMarkdown>
                  </div>
                  {msg.content.grounded && (
                    <div className="sources">
                      <h4>Sources</h4>
                      <ol>
                        {msg.content.sources.map((src, i) => (
                          <li key={i}>
                            <span className="source-title">
                              {src.title}
                            </span>
                            <span className="source-meta">
                              <span className="timestamp">{src.timestamp}</span>
                              <a
                                href={src.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="watch-link"
                              >
                                Watch Lecture
                              </a>
                            </span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                  {!msg.content.grounded && (
                    <div className="ungrounded">
                      Answer may not be grounded in the course material.
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="bubble loading">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}

        {error && <div className="error">{error}</div>}
        <div ref={bottomRef} />
      </main>

      <form className="input-bar" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your DSA question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
