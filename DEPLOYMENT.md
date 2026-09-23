# Striver DSA AI Mentor — Deployment Guide

## Production Architecture

```
React (Vercel/Netlify/Cloudflare Pages)
    ↓ HTTPS
Node.js/Express (Render/Railway/Fly.io)
    ↓ HTTPS/HTTP
Python FastAPI (Render/Railway/Fly.io)
    ↓ in-process
Hybrid Retrieval (Qdrant Cloud + BM25 + RRF + Reranker)
    ↓ context
LLM (Groq API)
```

## Prerequisites

Before deployment, ensure you have:

1. **GitHub/GitLab repository** with your code pushed
2. **Qdrant Cloud** account with cluster created
3. **Groq** API key (or Gemini/OpenAI)
4. **Hosting accounts** (choose one provider per service):
   - Frontend: Vercel / Netlify / Cloudflare Pages
   - Node API: Render / Railway / Fly.io
   - Python API: Render / Railway / Fly.io

## Step 1: Qdrant Cloud Setup

1. Sign up at https://cloud.qdrant.io
2. Create a new cluster (free tier is sufficient)
3. Note your:
   - `QDRANT_URL` (e.g., `https://xxx.cloud.qdrant.io`)
   - `QDRANT_API_KEY`
4. The existing 7601 vectors are in Qdrant Cloud already if you configured it during development.
5. If vectors are local, migrate using the Qdrant dashboard or snapshot import.

## Step 2: Environment Variables

### Root `.env` (Python service)

```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=striver_dsa_chunks
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
YTRAG_RETRIEVAL_K=15
YTRAG_BM25_K=15
YTRAG_HYBRID_K=10
YTRAG_RRF_K=60
YTRAG_TOP_K=5
YTRAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
YTRAG_RERANKER_DEVICE=auto
PYTHON_API_HOST=0.0.0.0
PYTHON_API_PORT=8000
PYTHON_API_CORS_ORIGINS=https://your-frontend.vercel.app,https://your-node-api.onrender.com
```

### Server `.env` (Node service)

```env
PORT=5000
PYTHON_RAG_URL=https://your-python-api.onrender.com
FRONTEND_URL=https://your-frontend.vercel.app
PYTHON_REQUEST_TIMEOUT=120000
```

### Client `.env` (React — optional)

```env
VITE_API_URL=https://your-node-api.onrender.com/api
```

## Step 3: Deploy Python FastAPI

### Option A: Render

1. Create new **Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Root directory:** `./` (repo root)
   - **Runtime:** Docker
   - **Dockerfile path:** `Dockerfile.python`
   - **Port:** 8000
4. Environment variables: Add all from Root `.env`
5. Deploy

### Option B: Railway

1. Create new project → Deploy from GitHub
2. Add Dockerfile
3. Set environment variables
4. Deploy

### Option C: Fly.io

```bash
fly launch
fly deploy
```

## Step 4: Deploy Node.js Backend

Same as Step 3, but use `Dockerfile.node` and port 5000.

Environment variables from Server `.env`.

## Step 5: Build and Deploy React

### Option A: Vercel

```bash
npm install -g vercel
vercel --prod
```

Or connect GitHub repo in Vercel dashboard:
- Framework preset: Vite
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL=https://your-node-api.onrender.com/api`

### Option B: Netlify

1. Connect GitHub repo
2. Build command: `npm run build`
3. Publish directory: `dist`
4. Environment variable: `VITE_API_URL`

### Option C: Cloudflare Pages

1. Connect GitHub repo
2. Build command: `npm run build`
3. Output directory: `dist`
4. Environment variable: `VITE_API_URL`

## Step 6: Verify Deployment

### Health Checks

```bash
# Python API
curl https://your-python-api.onrender.com/health

# Node API
curl https://your-node-api.onrender.com/api/health

# Expected response:
# {"status":"ok","python":{"reachable":true}}
```

### Test Query

```bash
curl -X POST https://your-node-api.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain binary search","mode":"explain"}'
```

## Step 7: CORS Configuration

Production CORS is controlled by:

- **Python:** `PYTHON_API_CORS_ORIGINS` env var
- **Node:** `FRONTEND_URL` env var

Set these to your actual frontend domain (e.g., `https://your-app.vercel.app`).

## Step 8: BM25 Index

The BM25 index (`data/search/bm25_index.pkl`) is built during Python service startup if missing.

**Production options:**

A. **Include in Docker image** (recommended for small index):
   - Copy `data/search/bm25_index.pkl` into the image
   - Faster startup, no rebuild needed

B. **Rebuild on startup**:
   - Exclude `data/` from Docker
   - Index rebuilds from `data/chunks/` on first request
   - 14.49 MB is small; rebuild takes ~5–10 seconds

**Current recommendation:** Include the pickle in the Docker image or mount it as a volume.

## Step 9: Security Checklist

- [ ] `.env` is not committed to Git
- [ ] `.env.example` contains placeholders only
- [ ] No API keys in source code
- [ ] No API keys in frontend bundle
- [ ] CORS restricted to production frontend domain
- [ ] Node backend does not expose Python errors to frontend
- [ ] HTTPS enabled on all services
- [ ] Environment variables set in hosting platform (not in code)

## Step 10: Performance Expectations

| Metric | Expected Value |
|---|---|
| Python startup (cold) | 10–30 seconds (model loading) |
| Python startup (warm) | < 2 seconds |
| Retrieval latency | 1–2 seconds |
| Reranking latency | 0.5–1 second |
| LLM latency | 2–5 seconds |
| Total response time | 4–8 seconds |

## Troubleshooting

### Python service won't start
- Check `QDRANT_URL` and `QDRANT_API_KEY` are set
- Verify Qdrant cluster is reachable
- Check logs for model loading errors

### Node service returns 500
- Verify `PYTHON_RAG_URL` points to the deployed Python service
- Check Python `/health` endpoint
- Ensure CORS origins match

### Frontend can't connect
- Verify `VITE_API_URL` points to Node API
- Check CORS configuration
- Ensure Node API is running

### BM25 not found
- The index will rebuild automatically from `data/chunks/`
- Ensure `data/chunks/` exists in the Python service

## Cost Estimates

### Free Tier (Student)

| Component | Service | Cost |
|---|---|---|
| Frontend | Vercel/Netlify | $0 |
| Node API | Render free tier | $0 |
| Python API | Render free tier | $0 |
| Qdrant | Qdrant Cloud free tier | $0 |
| LLM | Groq free tier | $0 |
| **Total** | | **$0/month** |

**Limitations:** Cold starts, limited hours, no custom domains.

### Production (Paid)

| Component | Service | Cost |
|---|---|---|
| Frontend | Vercel Pro | $20/month |
| Node API | Render Starter | $7/month |
| Python API | Render Starter | $7/month |
| Qdrant | Qdrant Cloud | $25/month |
| LLM | Groq API | Pay-as-you-go |
| **Total** | | **~$60–100/month** |

## Next Steps

1. Push code to GitHub (ensure `.env` is not committed)
2. Set up Qdrant Cloud
3. Deploy Python API first
4. Deploy Node API
5. Build and deploy React
6. Run health checks
7. Test end-to-end
