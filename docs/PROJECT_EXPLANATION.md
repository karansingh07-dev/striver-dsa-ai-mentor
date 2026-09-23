# 2-Minute Technical Interview Explanation — Striver DSA AI Mentor

> **How to use this guide:** Speak this response naturally during interview rounds when asked *"Tell me about your project"* or *"Walk me through the architecture of your RAG system"*.

---

## 🎙️ Spoken Script (2 Minutes)

"I built **Striver DSA AI Mentor**, a full-stack Retrieval-Augmented Generation (RAG) system that transforms **316 YouTube lectures** from Striver’s A2Z DSA playlist into an interactive, grounded AI tutor.

### 1. The Problem & Why RAG
Learning Data Structures & Algorithms from video lectures is passive. Students can't easily search across 300+ hours of video for specific syntax or concepts, and raw LLMs often hallucinate DSA code solutions. RAG solves this by grounding the AI in exact lecture transcripts while providing direct YouTube timestamp links.

### 2. Data Ingestion & Time-Aware Chunking
For data ingestion, I extracted audio using `yt-dlp` with cookie authentication to bypass YouTube bot blocks, then transcribed the audio using **Faster-Whisper** with CUDA GPU acceleration and CPU fallback. 

Instead of arbitrary token counts, I engineered **Time-Aware Chunking**—grouping transcripts into **75-second time windows with 15-second overlaps**. This preserves temporal continuity and guarantees that every retrieved chunk maps directly to a start and end second in the original video.

### 3. Embeddings, Storage & Hybrid Retrieval
I embedded these chunks into **7,601 vector points** using `all-MiniLM-L6-v2` and stored them in **Qdrant Cloud**.

Because DSA queries often involve exact syntax—like `lower_bound` or `unordered_map`—dense vector search alone isn't enough. So I built a **3-stage Hybrid Retrieval Engine**:
1. **Dense Semantic Search**: Retrieves top 15 candidates from Qdrant Cloud.
2. **Sparse Keyword Search**: Retrieves top 15 candidates from a pre-built **BM25 index** (14.5 MB).
3. **Reciprocal Rank Fusion (RRF, k=60)**: Merges both candidate pools based on relative rank positions.

### 4. Reranking, Grounding & Mentorship
To maximize precision, I pass the top 10 fused results into a **Cross-Encoder reranker** (`ms-marco-MiniLM-L-6-v2`) which jointly processes the query and text chunk to pick the top 5 most relevant context passages.

I enforced a **0.25 cosine similarity grounding floor**—if no retrieved chunk meets this threshold, the system rejects the answer to prevent hallucinations. The prompt is then passed to an LLM (Groq / Gemini / OpenAI) across **5 distinct mentorship modes**: Explain, Hint, Approach, Code Review, and Quiz.

### 5. Full-Stack Architecture
The system is built with a **3-tier microservice architecture**:
- **React + Vite UI**: Displays interactive chat history, Markdown code blocks, and playable YouTube timestamp citation cards.
- **Node.js / Express Backend**: Handles API proxying and manages in-memory conversation state.
- **Python / FastAPI RAG Service**: Runs the hybrid retrieval pipeline, model inference, and LLM generation."

---

## 🎯 Quick Elevator Summary (30 Seconds)

"Striver DSA AI Mentor is a RAG system indexing 316 YouTube lectures (7,601 Qdrant vectors) using Faster-Whisper, 75-second time-aware chunking, and a 3-stage retrieval pipeline combining Qdrant dense vector search, BM25 keyword matching, Reciprocal Rank Fusion, and Cross-Encoder reranking. It delivers grounded answers with clickable YouTube timestamp citations across 5 mentorship modes via FastAPI, Express, and React."
