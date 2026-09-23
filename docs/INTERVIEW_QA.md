# Technical Interview Q&A — Striver DSA AI Mentor

This document contains deep technical interview questions and detailed answers explaining the architecture, algorithms, trade-offs, and engineering decisions made in this project.

---

## Q1: Why did you use Hybrid Search (Dense + Sparse) instead of Pure Dense Vector Search?

**Answer:**
Pure dense vector search (using cosine similarity on embeddings) is great for capture of semantic intent and conceptual queries, such as "How do I find the middle of a linked list?". However, DSA domain queries often contain exact keyword syntax, function names, and technical terms (e.g., `lower_bound`, `Kadane's algorithm`, `unordered_map`, `O(N log N)`). Standard embedding models often smooth out or miss these specific tokens.

To solve this, we implemented a **3-stage retrieval pipeline**:
1. **Semantic Search**: Queries Qdrant Cloud for top 15 candidates using 384-dim `all-MiniLM-L6-v2` dense vectors.
2. **Sparse Keyword Search**: Queries a pre-indexed BM25 index (14.49 MB) for top 15 candidates based on exact term frequencies.
3. **Reciprocal Rank Fusion (RRF)**: Merges the candidates using:
   $$RRF\_Score(d) = \sum_{m \in \{Semantic, BM25\}} \frac{1}{k + r_m(d)} \quad (k=60)$$
This guarantees that exact syntax queries hit BM25 matches while concept queries hit semantic matches.

---

## Q2: Why is Cross-Encoder Reranking necessary after RRF Fusion?

**Answer:**
RRF effectively combines rank orders from two heterogeneous search methods, but RRF scores are based solely on rank positions, not deep cross-attention between query terms and transcript passages.

By passing the top 10 RRF candidates into a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), the query and document chunk are processed **jointly** through all transformer layers. This computes an exact relevance score between query and text, placing the top 5 most relevant context chunks directly into the LLM context prompt.

---

## Q3: How does Time-Aware Chunking differ from Standard Token Chunking?

**Answer:**
Standard text RAG systems chunk documents by token count (e.g. 512 tokens with 50 token overlap). In video lecture RAG, chunking by tokens breaks the temporal continuity of timestamps.

We engineered **Time-Aware Chunking** (`chunker.py`):
- Transcripts are grouped into **75-second time windows with a 15-second overlap**.
- Each chunk preserves `start_sec` and `end_sec` timestamps.
- When generating citations, `start_sec` converts directly into a playable YouTube deep link (e.g., `https://www.youtube.com/watch?v=VIDEO_ID&t=120s`).

---

## Q4: How do you handle YouTube Authentication & Bot Check Errors during Ingestion?

**Answer:**
YouTube frequently flags `yt-dlp` requests with "Sign in to confirm you're not a bot" errors.

We implemented a robust multi-layered auth solution in `audio_downloader.py`:
1. **Cookie Configuration**: Supports reading browser cookies directly via `--cookies-from-browser` (Chrome, Edge, Firefox) or loading Netscape-format `cookies.txt` via `YTDLP_COOKIES_FILE`.
2. **Automated Error Handling**: If a configured cookie file or locked browser database fails, the pipeline logs an unambiguous warning and attempts an anonymous fallback before recording state.
3. **Resilient Ingestion Checkpoints**: Each video's stage (download, transcription, chunking, embedding, vector storage) is saved to `data/status/`. Interrupted ingestions resume from the exact point of failure without repeating successful stages.
