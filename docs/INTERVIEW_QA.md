# Technical Interview Cheat Sheet — Striver DSA AI Mentor

This document contains deep technical interview questions and detailed answers explaining the architecture, algorithms, trade-offs, and engineering decisions made in this project.

---

## 1. Core RAG & Vector Architecture

### Q1: Why RAG instead of Fine-Tuning or Prompt Engineering alone?
**Answer:**
Fine-tuning an LLM updates model parameters with knowledge but cannot easily cite specific YouTube video timestamps, is expensive to retrain when new videos are published, and still suffers from hallucinations on edge-case syntax. RAG (Retrieval-Augmented Generation) keeps the data layer dynamic and searchable, allowing us to ground answers in verbatim transcript passages from Striver's lectures and generate clickable YouTube links (`https://youtube.com/watch?v=...&t=120s`) for verifiable learning.

### Q2: Why Qdrant Cloud for Vector Storage?
**Answer:**
Qdrant Cloud provides a managed, high-performance HNSW (Hierarchical Navigable Small World) vector index. It natively supports payload filtering (filtering by `video_id`, `module`, etc.), offers low-latency REST/gRPC interfaces, and supports deterministic point IDs (UUIDv5) allowing idempotent upserts during ingestion runs.

### Q3: Why Embeddings and why `sentence-transformers/all-MiniLM-L6-v2`?
**Answer:**
Dense embeddings convert text into high-dimensional numerical vectors capturing semantic intent. We selected `all-MiniLM-L6-v2` because it generates 384-dimensional vectors with fast inference speed (ideal for real-time CPU/GPU deployment) while maintaining strong benchmark performance on sentence similarity and semantic retrieval tasks.

### Q4: Why BM25 Keyword Search alongside Dense Vector Search?
**Answer:**
Dense vector search excels at conceptual intent ("How do I reverse a linked list?"), but struggles with exact technical syntax, C++ STL function names, or math terms (e.g. `lower_bound`, `Kadane's algorithm`, `unordered_map`, `O(N log N)`). BM25 uses Term Frequency-Inverse Document Frequency (TF-IDF) scoring with length normalization ($k_1=1.2, b=0.75$) to guarantee exact term recall.

### Q5: Why Hybrid Search (Dense + Sparse)?
**Answer:**
Neither dense vector search nor BM25 keyword search is optimal alone. Hybrid search combines dense semantic recall with sparse keyword precision, preventing retrieval failures on syntax-heavy computer science queries.

### Q6: What is Reciprocal Rank Fusion (RRF) and why use it?
**Answer:**
RRF is a score normalization technique used to combine rank lists from heterogeneous search algorithms (Qdrant similarity scores and BM25 scores) without needing to calibrate their raw numeric scales:
$$RRF\_Score(d) = \frac{1}{k + r_{\text{dense}}(d)} + \frac{1}{k + r_{\text{sparse}}(d)} \quad (k=60)$$
A smoothing constant of $k=60$ balances high-ranking items from both search engines, producing a single unified candidate list.

---

## 2. Retrieval, Reranking & Ingestion

### Q7: Why Two-Stage / Three-Stage Retrieval and Reranking?
**Answer:**
Single-stage vector search returns items based on coarse similarity. Our 3-stage pipeline:
1. **Stage 1 (Retrieval)**: Pulls top 15 candidate chunks from Qdrant Cloud (dense).
2. **Stage 2 (Keyword & Fusion)**: Pulls top 15 candidates from BM25 (sparse) and merges them via RRF ($k=60$) into top 10 candidates.
3. **Stage 3 (Cross-Encoder Reranking)**: Re-scores the top 10 fused candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 5 highest-relevance context chunks for the LLM.

### Q8: Why Cross-Encoder Reranking over Vector Similarity alone?
**Answer:**
Bi-encoders (like `all-MiniLM-L6-v2`) encode query and document separately into static vectors, missing fine-grained token-to-token cross-attention. A Cross-Encoder processes the query and document chunk **jointly** through all self-attention transformer layers, accurately scoring exact relevance before LLM prompt assembly.

### Q9: How does the Ingestion Pipeline work?
**Answer:**
1. **Playlist Fetching**: Extract video IDs, titles, and sequence metadata via `playlist_fetcher.py`.
2. **Audio Download**: Download `m4a` audio via `audio_downloader.py` using `yt-dlp` with cookie auth support (`chrome`/`cookies.txt`).
3. **Transcription**: Transcribe audio via `Faster-Whisper` (`base` model, CUDA float16 with CPU int8 fallback).
4. **Time-Aware Chunking**: Group segment timestamps into 75-second windows with 15-second overlap (`chunker.py`).
5. **Vector Indexing**: Generate 384-dim embeddings (`embedder.py`) and perform deterministic UUIDv5 upserts into Qdrant Cloud (`vector_store.py`).
6. **BM25 Indexing**: Build and pickle rank-bm25 index at `data/search/bm25_index.pkl`.

### Q10: How are Timestamps Preserved during Chunking?
**Answer:**
`chunker.py` uses fixed time duration windows (75s with 15s overlap) instead of character/token bounds. Each chunk tracks `start_sec` and `end_sec`. When constructing LLM prompts and UI source cards, `start_sec` generates exact playable YouTube links (e.g. `https://www.youtube.com/watch?v=VIDEO_ID&t=180s`).

### Q11: How are Failed Videos or YouTube Bot-Checks Handled?
**Answer:**
In `audio_downloader.py`, `yt-dlp` supports automated browser cookie extraction (`--cookies-from-browser chrome`) and `cookies.txt` loading. Ingestion status per video is persisted in `data/status/`. Interrupted runs resume without repeating previously completed videos.


---

## 3. Operations, Reliability & Architecture

### Q12: How does Caching work in the system?
**Answer:**
- **In-Memory ML Models**: `RAGEngine` lazily initializes the embedding model, cross-encoder reranker, and BM25 index once during startup, caching them across requests.
- **BM25 Persistence**: The tokenized BM25 index is pickled to `data/search/bm25_index.pkl` (14.49 MB) to prevent expensive re-tokenization of 7,601 chunks on server restarts.
- **Session Memory**: Node.js middleware retains multi-turn chat history (`conversationStore.js`) in-memory for active sessions.

### Q13: How does the LLM receive Context?
**Answer:**
The top 5 reranked transcript chunks are formatted with metadata headers (Video Title, Start Time, End Time, YouTube URL, Segment Text) and injected into the prompt alongside the user question and mode-specific instructions (Explain, Hint, Approach, Code Review, Quiz).

### Q14: How do you reduce Hallucinations?
**Answer:**
1. **Grounding Cosine Similarity Floor (`YTRAG_MIN_SCORE = 0.25`)**: Retained chunks scoring below 0.25 are discarded. If zero chunks meet the score, the system rejects generation and returns a fallback message.
2. **System Prompt Rules**: Prompts explicitly forbid generating facts outside the retrieved transcripts and require inline markdown timestamp citations `[Title](url)`.

### Q15: What happens if Retrieval fails?
**Answer:**
If Qdrant Cloud or BM25 fails to retrieve matching documents, `RAGEngine` catches the exception, flags `grounded = False`, and returns a safe fallback message informing the user that insufficient context exists in the Striver DSA dataset for that question.

### Q16: What happens if the LLM API fails?
**Answer:**
`rag_engine.py` supports provider switching (Groq, Gemini, OpenAI). If an API call fails due to rate limits or network issues, FastAPI captures the error, returns a structured HTTP 500 JSON error payload, and Node.js proxies the friendly error message to the React UI without crashing the service.

### Q17: How are API keys protected?
**Answer:**
All API keys (`GROQ_API_KEY`, `QDRANT_API_KEY`, etc.) are loaded exclusively via environment variables (`.env`). `.env`, `cookies.txt`, and server environment configs are strictly included in `.gitignore` and omitted from version control. Sample templates (`.env.example`) contain placeholders only.

### Q18: Why React + Node.js + FastAPI (3-Tier Microservices)?
**Answer:**
- **React UI**: High-performance single-page app for real-time Markdown rendering, code highlighting, and interactive source cards.
- **Node.js/Express Proxy**: Handles CORS, session conversation state management, and request routing.
- **Python/FastAPI Engine**: Leverages Python's native ML ecosystem (PyTorch, Sentence-Transformers, CTranslate2, rank-bm25, Qdrant Client) while exposing async endpoints with FastAPI lifespan events.

### Q19: How would you scale this application for production?
**Answer:**
1. **Vector DB Scaling**: Qdrant Cloud handles horizontal scaling with distributed collections and HNSW indexing.
2. **Stateless API Workers**: Deploy FastAPI behind Gunicorn/Uvicorn workers on AWS ECS / Kubernetes.
3. **Session Persistence**: Replace Node's in-memory conversation store with Redis.
4. **Model Serving**: Host `all-MiniLM-L6-v2` and Cross-Encoder model inference on dedicated GPU microservices (Triton Inference Server or TEI).

