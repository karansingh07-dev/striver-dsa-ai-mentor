import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress non-critical HuggingFace Hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# Data Directories
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
QDRANT_LOCAL_DIR = DATA_DIR / "qdrant_db"
SEARCH_DIR = DATA_DIR / "search"

# Phase 7 — Playlist Ingestion state (per-video checkpoints + machine-readable report)
STATUS_DIR = DATA_DIR / "status"
INGESTION_REPORT = DATA_DIR / "ingestion_report.json"

# Create directories if they do not exist
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

# Ensure FFmpeg is available in system PATH (via imageio_ffmpeg fallback)
try:
    import imageio_ffmpeg
    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = str(ffmpeg_path.parent)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

# Whisper Settings
DEFAULT_WHISPER_MODEL = "base"
# Default to NVIDIA GPU (CUDA, float16). Set WHISPER_DEVICE=cpu (and/or
# WHISPER_COMPUTE_TYPE=int8) in .env to force CPU; transcriber.py also falls
# back to cpu/int8 automatically if CUDA initialisation fails at load time.
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "cuda").strip().lower()
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip().lower()

# If the nvidia-cublas-cu12 package is installed (pip install nvidia-cublas-cu12),
# expose its bin directory on PATH: CTranslate2 loads "cublas64_12.dll" by bare
# name on Windows, and the standard DLL search order includes PATH.
try:
    import importlib.util
    _cublas_spec = importlib.util.find_spec("nvidia.cublas")
    if _cublas_spec is not None:
        if getattr(_cublas_spec, "origin", None):
            _cublas_root = Path(_cublas_spec.origin).parent
        elif _cublas_spec.submodule_search_locations:
            _cublas_root = Path(list(_cublas_spec.submodule_search_locations)[0])
        else:
            _cublas_root = None
        if _cublas_root is not None:
            _cublas_bin = _cublas_root / "bin"
            if _cublas_bin.is_dir() and str(_cublas_bin) not in os.environ["PATH"]:
                os.environ["PATH"] = str(_cublas_bin) + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

# Supported Audio Format
AUDIO_FORMAT = "m4a"

# yt-dlp / YouTube Authentication (Phase 7.1)
# ------------------------------------------------
# Fix for YouTube "Sign in to confirm you're not a bot" errors.
# Set YTDLP_COOKIES_FROM_BROWSER to a browser yt-dlp supports (chrome, edge,
# firefox, brave, opera, ...) to authenticate downloads with that browser's
# cookies (equivalent to `yt-dlp --cookies-from-browser <browser>`).
# Alternatively export a Netscape-format cookies.txt and point
# YTDLP_COOKIES_FILE at it (equivalent to `yt-dlp --cookies <file>`).
# Never hardcode cookies/API keys in source code.
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip().lower()
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "").strip()

# Chunking Settings (Time-Aware Chunking in Seconds)
CHUNK_SECONDS = float(os.getenv("YTRAG_CHUNK_SECONDS", "75"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("YTRAG_CHUNK_OVERLAP", "15"))

# Embedding Settings (Local Sentence Transformers)
DEFAULT_EMBED_MODEL = os.getenv("YTRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_EMBED_BATCH_SIZE = int(os.getenv("YTRAG_EMBED_BATCH_SIZE", "32"))

# Qdrant Vector DB Settings
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "striver_dsa_chunks").strip()
QDRANT_BATCH_SIZE = int(os.getenv("YTRAG_QDRANT_BATCH_SIZE", "64"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "300"))
DEFAULT_TOP_K = int(os.getenv("YTRAG_TOP_K", "5"))
YTRAG_TOP_K = DEFAULT_TOP_K


def _env_int(key: str, default: int) -> int:
    """Reads an integer env var, tolerating empty/unparseable values."""
    val = os.getenv(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"[WARNING] Invalid integer for {key}='{val}'; using default {default}.")
        return default


def _env_float(key: str, default: float) -> float:
    """Reads a float env var, tolerating empty/unparseable values."""
    val = os.getenv(key, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"[WARNING] Invalid number for {key}='{val}'; using default {default}.")
        return default


# Grounding Threshold (Minimum Cosine Similarity Score required for context relevance).
# Cosine similarity: HIGHER score = MORE relevant (never invert this comparison).
# Leave YTRAG_MIN_SCORE empty to use the default 0.25 below.
YTRAG_MIN_SCORE = _env_float("YTRAG_MIN_SCORE", 0.25)

# Three Retrieval Quality Settings (Phase 6):
#   RETRIEVAL_K            - Stage-1 candidate pool pulled from Qdrant (broader recall).
#   YTRAG_TOP_K (DEFAULT_TOP_K) - Stage-3 final context chunks fed to the LLM.
#   DEDUP_OVERLAP_RATIO    - Two chunks from the SAME video with >= this fraction of
#                            overlapping time-range are treated as duplicates; the
#                            higher-scoring one wins.
YTRAG_RETRIEVAL_K = _env_int("YTRAG_RETRIEVAL_K", 15)
YTRAG_DEDUP_OVERLAP_RATIO = _env_float("YTRAG_DEDUP_OVERLAP_RATIO", 0.7)

# Phase 8 — Hybrid Search + Re-ranking
YTRAG_BM25_K = _env_int("YTRAG_BM25_K", 15)
YTRAG_HYBRID_K = _env_int("YTRAG_HYBRID_K", 10)
YTRAG_RRF_K = _env_int("YTRAG_RRF_K", 60)
YTRAG_RERANKER_MODEL = os.getenv("YTRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
YTRAG_RERANKER_DEVICE = os.getenv("YTRAG_RERANKER_DEVICE", "auto").strip().lower()
SEARCH_DIR = DATA_DIR / "search"
BM25_INDEX_PATH = SEARCH_DIR / "bm25_index.pkl"

# LLM Settings (Supports Groq, Gemini, OpenAI)
# Default provider is groq. Override with LLM_PROVIDER in .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()






