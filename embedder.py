from pathlib import Path
from typing import Dict, Any, List, Optional
import os

from config import (
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_BATCH_SIZE,
)
from utils import save_json, load_json

# Process-level model cache keyed by model name. QdrantVectorStore.search()
# creates a NEW EmbeddingGenerator for every query, so without this cache the
# SentenceTransformer weights are re-read from disk once per query (e.g. once
# per evaluation question). The cache keeps one loaded model per process.
_MODEL_CACHE: Dict[str, Any] = {}


class EmbeddingGenerator:
    """
    Local embedding generator using Sentence Transformers.
    Converts timestamp-aware transcript chunks into dense vector embeddings
    while preserving all original metadata and segment boundaries.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        output_dir: Path = EMBEDDINGS_DIR,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def model(self):
        """Lazy load SentenceTransformer model when needed (cached per process)."""
        if self._model is None:
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is not None:
                self._model = cached
            else:
                print(f"[EMBEDDER] Loading embedding model '{self.model_name}'...")
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
                    _MODEL_CACHE[self.model_name] = self._model
                    print(f"[EMBEDDER] Model loaded successfully. Dimension: {self.dimension}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load SentenceTransformer model '{self.model_name}': {e}")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector embedding dimension of the loaded model."""
        m = self.model
        if hasattr(m, "get_embedding_dimension"):
            return m.get_embedding_dimension()
        return m.get_sentence_embedding_dimension()

    def generate_embeddings_for_file(self, chunk_file: Path) -> Dict[str, Any]:
        """
        Generates vector embeddings for all chunks in a single chunk JSON file.
        Attaches normalized embeddings while preserving all chunk metadata.
        """
        chunk_data = load_json(chunk_file)
        video_id = chunk_data.get("video_id", chunk_file.stem)
        title = chunk_data.get("title", video_id)
        chunks = chunk_data.get("chunks", [])

        if not chunks:
            out_payload = {
                "video_id": video_id,
                "title": title,
                "embedding_model": self.model_name,
                "embedding_dimension": self.dimension,
                "chunk_count": 0,
                "chunks": []
            }
            out_file = self.output_dir / f"{video_id}.json"
            save_json(out_payload, out_file)
            return out_payload

        # Extract texts for batch encoding
        texts = [c.get("text", "").strip() for c in chunks]

        # Batch encode with normalization (L2 norm = 1.0 for Cosine Similarity)
        raw_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embedded_chunks = []
        for idx, chunk_item in enumerate(chunks):
            c_dict = dict(chunk_item)
            emb_vector = [float(val) for val in raw_embeddings[idx]]
            c_dict["embedding"] = emb_vector
            embedded_chunks.append(c_dict)

        out_payload = {
            "video_id": video_id,
            "title": title,
            "embedding_model": self.model_name,
            "embedding_dimension": len(embedded_chunks[0]["embedding"]),
            "chunk_count": len(embedded_chunks),
            "chunks": embedded_chunks
        }

        out_file = self.output_dir / f"{video_id}.json"
        save_json(out_payload, out_file)
        return out_payload

    def process_all_chunks(
        self,
        chunks_dir: Path = CHUNKS_DIR,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Scans data/chunks/ directory, embeds all chunk files, and saves to data/embeddings/.
        Skips already processed embedding files unless force=True.
        """
        chunk_files = list(Path(chunks_dir).glob("*.json"))

        if not chunk_files:
            print(f"[EMBEDDER] No chunk files found in '{chunks_dir}'. Run 'python main.py --chunk' first.")
            return {"files_processed": 0, "chunks_embedded": 0, "failed": 0}

        print(f"Found {len(chunk_files)} chunk files.\n")

        processed_files = 0
        total_chunks_embedded = 0
        failed_count = 0

        for idx, c_file in enumerate(chunk_files, 1):
            vid_id = c_file.stem
            emb_file = self.output_dir / f"{vid_id}.json"

            try:
                c_data = load_json(c_file)
            except Exception as e:
                print(f"[{idx}/{len(chunk_files)}] Failed to load '{c_file.name}': {e}")
                failed_count += 1
                continue

            title = c_data.get("title", vid_id)
            chunk_count = len(c_data.get("chunks", []))

            # Cache check
            if not force and emb_file.exists() and emb_file.stat().st_size > 0:
                try:
                    cached_emb_data = load_json(emb_file)
                    cached_chunks = cached_emb_data.get("chunks", [])
                    if len(cached_chunks) == chunk_count and (not cached_chunks or "embedding" in cached_chunks[0]):
                        total_chunks_embedded += chunk_count
                        processed_files += 1
                        print(f"[{idx}/{len(chunk_files)}] {title}")
                        print(f"       {chunk_count} chunks")
                        print(f"       [CACHE HIT] Loaded embeddings from cache\n")
                        continue
                except Exception:
                    pass  # Corrupted cache file, re-generate

            try:
                res = self.generate_embeddings_for_file(c_file)
                num_chunks = res["chunk_count"]
                total_chunks_embedded += num_chunks
                processed_files += 1
                print(f"[{idx}/{len(chunk_files)}] {title}")
                print(f"       {num_chunks} chunks")
                print(f"       Embeddings generated\n")
            except Exception as e:
                print(f"[{idx}/{len(chunk_files)}] ERROR embedding '{title}': {e}\n")
                failed_count += 1

        print("=" * 50)
        print("EMBEDDING GENERATION COMPLETE")
        print(f"Files processed: {processed_files}")
        print(f"Chunks embedded: {total_chunks_embedded}")
        print(f"Failed:          {failed_count}")
        print("=" * 50)

        return {
            "files_processed": processed_files,
            "chunks_embedded": total_chunks_embedded,
            "failed": failed_count
        }

    def print_info(self, chunks_dir: Path = CHUNKS_DIR, embeddings_dir: Path = EMBEDDINGS_DIR):
        """Prints embedding model information and corpus stats."""
        emb_dim = self.dimension
        chunk_files = list(Path(chunks_dir).glob("*.json"))
        total_chunks = 0

        for cf in chunk_files:
            try:
                d = load_json(cf)
                total_chunks += len(d.get("chunks", []))
            except Exception:
                pass

        print("\n" + "=" * 40)
        print(" EMBEDDING PIPELINE INFO")
        print("=" * 40)
        print(f"Embedding model:\n{self.model_name}\n")
        print(f"Embedding dimension:\n{emb_dim}\n")
        print(f"Number of files:\n{len(chunk_files)}\n")
        print(f"Number of chunks:\n{total_chunks}")
        print("=" * 40 + "\n")
