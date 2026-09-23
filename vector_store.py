import json
import time
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    EMBEDDINGS_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_BATCH_SIZE,
    QDRANT_TIMEOUT,
    QDRANT_LOCAL_DIR,
    DEFAULT_TOP_K,
)
from embedder import EmbeddingGenerator


def generate_deterministic_id(video_id: str, chunk_id: str) -> str:
    """
    Generates a deterministic UUID v5 from video_id and chunk_id.
    This guarantees idempotency: upserting the exact same chunk multiple times
    updates the existing point in Qdrant instead of inserting duplicates.
    """
    key = f"{video_id}_{chunk_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class QdrantVectorStore:
    """
    Manages vector indexing and semantic similarity search in Qdrant vector database.
    Supports both Qdrant Cloud (via QDRANT_URL & QDRANT_API_KEY) and local disk fallback.
    """

    def __init__(self, collection_name: str = QDRANT_COLLECTION):
        self.collection_name = collection_name
        self.client = self._init_client()

    def _init_client(self) -> QdrantClient:
        """Initializes Qdrant client depending on environment configuration."""
        if QDRANT_URL:
            api_key = QDRANT_API_KEY if QDRANT_API_KEY else None
            try:
                return QdrantClient(
                    url=QDRANT_URL,
                    api_key=api_key,
                    timeout=QDRANT_TIMEOUT,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to Qdrant Cloud at {QDRANT_URL}: {e}"
                )
        else:
            # Fallback to persistent local disk storage
            return QdrantClient(path=str(QDRANT_LOCAL_DIR))

    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        """
        Creates or safely recreates the Qdrant collection using Cosine distance.
        Vector dimension is dynamically set based on detected embedding data.
        """
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception:
            exists = False

        if recreate and exists:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def validate_chunk_point(self, chunk: Dict[str, Any], file_name: str) -> bool:
        """
        Validates that a chunk contains all required vector embedding & payload fields.
        """
        required_fields = [
            "chunk_id",
            "video_id",
            "title",
            "start_sec",
            "end_sec",
            "text",
            "youtube_url",
            "embedding",
        ]
        for field in required_fields:
            if field not in chunk or chunk[field] is None:
                print(
                    f"[WARNING] Skipping chunk in {file_name}: missing required field '{field}'"
                )
                return False

        if not isinstance(chunk["embedding"], list) or len(chunk["embedding"]) == 0:
            print(
                f"[WARNING] Skipping chunk '{chunk.get('chunk_id')}' in {file_name}: invalid or empty embedding array"
            )
            return False

        return True
    def index_all_embeddings(
        self,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        reindex: bool = False,
        batch_size: int = QDRANT_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Reads cached vector embedding JSON files from embeddings_dir, validates payloads,
        detects vector dimension, and batch-upserts points to Qdrant with deterministic IDs.
        """
        if not embeddings_dir.exists():
            raise FileNotFoundError(
                f"Embeddings directory '{embeddings_dir}' does not exist. Run 'python main.py --embed' first."
            )

        json_files = list(embeddings_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No embedding JSON files found in '{embeddings_dir}'. Run 'python main.py --embed' first."
            )

        print(f"\nFound {len(json_files)} embedding file(s).")
        print(f"Collection: {self.collection_name}")

        # Detect vector dimension from first valid chunk
        detected_dim = None
        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                chunks = data.get("chunks", [])
                for c in chunks:
                    if "embedding" in c and isinstance(c["embedding"], list) and len(c["embedding"]) > 0:
                        detected_dim = len(c["embedding"])
                        break
                if detected_dim:
                    break
            except Exception:
                continue

        if not detected_dim:
            raise ValueError(
                "Could not detect valid embedding dimension from cached embedding files."
            )

        print(f"Embedding dimension: {detected_dim}\n")

        # Initialize/Reset collection
        self.ensure_collection(dimension=detected_dim, recreate=reindex)

        total_files = len(json_files)
        files_processed = 0
        total_vectors_indexed = 0
        failed_files = 0

        for idx, json_file in enumerate(json_files, start=1):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                video_title = data.get("title", json_file.stem)
                chunks = data.get("chunks", [])

                points_to_upsert = []
                for chunk in chunks:
                    if not self.validate_chunk_point(chunk, json_file.name):
                        continue

                    embedding_vec = [float(x) for x in chunk["embedding"]]
                    if len(embedding_vec) != detected_dim:
                        print(
                            f"[WARNING] Skipping chunk '{chunk['chunk_id']}': dimension mismatch ({len(embedding_vec)} vs expected {detected_dim})"
                        )
                        continue

                    point_id = generate_deterministic_id(
                        chunk["video_id"], chunk["chunk_id"]
                    )
                    payload = {
                        "video_id": chunk["video_id"],
                        "title": chunk["title"],
                        "chunk_id": chunk["chunk_id"],
                        "start_sec": chunk["start_sec"],
                        "end_sec": chunk["end_sec"],
                        "start_time": chunk.get("start_time", ""),
                        "end_time": chunk.get("end_time", ""),
                        "text": chunk["text"],
                        "youtube_url": chunk["youtube_url"],
                    }

                    points_to_upsert.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embedding_vec,
                            payload=payload,
                        )
                    )

                # Batch Upsert
                file_vectors_count = len(points_to_upsert)
                for b_start in range(0, file_vectors_count, batch_size):
                    batch = points_to_upsert[b_start : b_start + batch_size]
                    self._upsert_with_retry(batch)

                files_processed += 1
                total_vectors_indexed += file_vectors_count

                print(f"[{idx}/{total_files}] {video_title}")
                print(f"      {file_vectors_count} chunks indexed\n")

            except Exception as e:
                failed_files += 1
                print(
                    f"[{idx}/{total_files}] [FAILED] Failed to index {json_file.name}: {e}\n",
                    file=sys.stderr,
                )

        print("=" * 42)
        print("QDRANT INDEXING COMPLETE")
        print("=" * 42)
        print(f"Files processed: {files_processed}")
        print(f"Vectors indexed: {total_vectors_indexed}")
        print(f"Failed:          {failed_files}")
        print("=" * 42)

        return {
            "files_processed": files_processed,
            "vectors_indexed": total_vectors_indexed,
            "failed_files": failed_files,
        }

    def _upsert_with_retry(
        self, batch: List[models.PointStruct], attempts: int = 3
    ) -> None:
        """
        Upserts a batch of points with exponential-backoff retries.

        Transient network timeouts (common on Qdrant Cloud free tier) are
        retried before the file is declared failed. The final exception is
        re-raised so the per-file error handling in index_all_embeddings
        reports it accurately.
        """
        for attempt in range(1, attempts + 1):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                return
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(2 * attempt)  # 2s, 4s backoff

    def index_embedding_file(
        self,
        embedding_file: Path,
        batch_size: int = QDRANT_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Indexes a SINGLE cached embedding JSON file into Qdrant.

        Point IDs are deterministic UUIDv5 values derived from (video_id, chunk_id),
        so re-running this function for the same file is an idempotent UPSERT:
        existing points are updated in place and NO duplicates are ever created.

        The collection is (re)used as-is — no second collection is ever created.
        """
        embedding_file = Path(embedding_file)
        data = json.loads(embedding_file.read_text(encoding="utf-8"))

        video_id = data.get("video_id", embedding_file.stem)
        title = data.get("title", video_id)
        chunks = data.get("chunks", [])

        if not chunks:
            raise ValueError(f"No chunks found in '{embedding_file.name}'.")

        seen_chunk_ids: set = set()
        unique_chunks = []
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            if cid in seen_chunk_ids:
                print(
                    f"[WARNING] Skipping duplicate chunk '{cid}' in "
                    f"'{embedding_file.name}': duplicate chunk_id detected."
                )
                continue
            seen_chunk_ids.add(cid)
            unique_chunks.append(chunk)

        detected_dim = None
        points = []

        for chunk in unique_chunks:
            if not self.validate_chunk_point(chunk, embedding_file.name):
                continue

            embedding_vec = [float(x) for x in chunk["embedding"]]
            if detected_dim is None:
                detected_dim = len(embedding_vec)
            elif len(embedding_vec) != detected_dim:
                print(
                    f"[WARNING] Skipping chunk '{chunk['chunk_id']}': dimension mismatch "
                    f"({len(embedding_vec)} vs {detected_dim})"
                )
                continue

            point_id = generate_deterministic_id(chunk["video_id"], chunk["chunk_id"])
            payload = {
                "video_id": chunk["video_id"],
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "start_time": chunk.get("start_time", ""),
                "end_time": chunk.get("end_time", ""),
                "text": chunk["text"],
                "youtube_url": chunk["youtube_url"],
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding_vec,
                    payload=payload,
                )
            )

        if detected_dim is None:
            raise ValueError(f"No valid embedded chunks found in '{embedding_file.name}'.")

        # Reuse the existing collection with the correct dimension (no-op if present).
        self.ensure_collection(dimension=detected_dim, recreate=False)

        # Batch upsert (idempotent).
        for b_start in range(0, len(points), batch_size):
            batch = points[b_start : b_start + batch_size]
            self._upsert_with_retry(batch)

        return {
            "video_id": video_id,
            "title": title,
            "chunks_total": len(unique_chunks),
            "vectors_indexed": len(points),
        }

    def count_existing_points(self, point_ids: List[str], video_id: Optional[str] = None) -> int:
        """
        Returns how many of the given point IDs already exist in the collection.

        Prefers a payload filter count when video_id is available, but falls back
        to retrieve() if the collection lacks the required payload index.
        """
        if not point_ids:
            return 0
        try:
            if video_id:
                try:
                    result = self.client.count(
                        collection_name=self.collection_name,
                        count_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="video_id",
                                    match=models.MatchValue(value=video_id),
                                )
                            ]
                        ),
                        exact=True,
                    )
                    return int(result.count) if result else 0
                except Exception:
                    pass
            found = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            return len(found) if found else 0
        except Exception:
            return 0

    def count_all_points(self) -> int:
        """Returns the total number of vectors currently in the collection."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                exact=False,
            )
            return int(result.count) if result else 0
        except Exception:
            return 0

    def print_info(self) -> None:
        """Displays status and information about the Qdrant vector database collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            vectors_config = collection_info.config.params.vectors
            dim = getattr(vectors_config, "size", "N/A")
            distance = getattr(vectors_config, "distance", "Cosine")

            print("\n" + "=" * 40)
            print(" QDRANT DATABASE INFO")
            print("=" * 40)
            print(f"Collection:       {self.collection_name}")
            print(f"Vectors Indexed:  {points_count}")
            print(f"Vector Dimension: {dim}")
            print(f"Distance Metric:  {distance}")
            print("=" * 40 + "\n")
        except Exception as e:
            print(f"\n[ERROR] Unable to retrieve Qdrant info for collection '{self.collection_name}': {e}\n")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Converts query string to embedding using SentenceTransformers and retrieves top_k nearest vectors from Qdrant.
        """
        if not query or not query.strip():
            print("[ERROR] Search query cannot be empty.")
            return []

        # Load embedding generator (same model used during indexing)
        embedder = EmbeddingGenerator()
        query_vector = embedder.model.encode(query, normalize_embeddings=True).tolist()

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                )
        except Exception as e:
            print(f"\n[ERROR] Qdrant search failed: {e}")
            return []

        hits = []
        for hit in results:
            payload = hit.payload or {}
            hits.append(
                {
                    "score": round(float(hit.score), 4),
                    "title": payload.get("title", "Unknown Title"),
                    "start_time": payload.get("start_time", "00:00:00"),
                    "start_sec": int(payload.get("start_sec", 0)),
                    "end_sec": int(payload.get("end_sec", 0)),
                    "text": payload.get("text", ""),
                    "youtube_url": payload.get("youtube_url", ""),
                    "video_id": payload.get("video_id", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                }
            )

        return hits
