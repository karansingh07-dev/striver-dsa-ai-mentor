# Resume Bullet Points — Striver DSA AI Mentor

These bullet points are formatted for software engineering, AI/ML engineering, and full-stack/backend resume submissions. Select the variation that best fits your Target Role.

---

## 🚀 Tailored Role Bullet Points

### Option 1: AI / ML Engineer (Focus on RAG, Search Architecture, & LLMs)
- **Built an End-to-End Multimodal RAG Mentor** indexing **316 Striver A2Z DSA YouTube lectures** (~7,601 vector embeddings in Qdrant Cloud), enabling instant natural language search with timestamp-precise lecture video citations.
- **Architected a 3-Stage Hybrid Retrieval Engine** combining **384-dim dense vector search** (Sentence-Transformers) and **BM25 sparse keyword search** merged via **Reciprocal Rank Fusion (RRF, k=60)** and re-scored with a **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) reranker**, increasing retrieval accuracy for DSA domain terms.
- **Engineered Time-Aware Transcript Chunking** (75s windows with 15s overlap) paired with **Faster-Whisper (CUDA float16 / CPU int8)**, reducing audio-to-text processing time while maintaining exact video timestamp alignment for deep-linking.
- **Designed 5 Custom Prompt Mentorship Modes** (Explain, Hint, Approach, Code Review, Quiz) enforced with strict score grounding thresholds ($\ge 0.25$ cosine similarity floor) to eliminate hallucinated answers and mandate source citations.

---

### Option 2: Full-Stack / Backend Engineer (Focus on System Design, Performance, & APIs)
- **Developed a Microservices-Based RAG Application** with a **FastAPI Python backend** for AI retrieval, a **Node.js/Express middleware** handling session state and API proxying, and a **React 18 + Vite** frontend.
- **Optimized Model Inference & Pipeline Performance** by implementing **lazy single-instance model loading** and **GPU-accelerated Whisper transcription** with automated runtime CPU fallback, preventing memory leaks and ensuring high service availability.
- **Implemented Resilient Data Ingestion & Authentication** handling 316 videos using `yt-dlp` with automated cookie extraction (`chrome`/`cookies.txt`), checkpoint state persistence, and idempotent Qdrant deterministic UUIDv5 vector upserts.
- **Built Production-Grade REST APIs & Web Interface** complete with FastAPI lifespan events, CORS management, request validation, structured error reporting, and interactive source attribution cards in React.

---

### Option 3: Concise Core Summary (For 1-2 Line Resume Sections)
- **Striver DSA AI Mentor**: Engineered a full-stack RAG system over **316 DSA YouTube lectures** (**7,601 Qdrant vectors**, 14.5MB BM25 index) using **Faster-Whisper CUDA transcription**, **RRF Hybrid Retrieval**, **Cross-Encoder Reranking**, and **FastAPI / Express / React**.
