"""
Phase 7 — Production-Style Full Playlist Ingestion.

End-to-end, resumable, cache-driven pipeline:

    Playlist discovery
            ↓
    Transcription  (data/downloads/  audio cache  -> data/transcripts/)
            ↓
    Chunking       (data/transcripts/ -> data/chunks/)
            ↓
    Embedding      (data/chunks/ -> data/embeddings/)
            ↓
    Qdrant Indexing (data/embeddings/ -> existing collection, e.g. "striver_dsa_chunks")

Design guarantees:
  * Each stage consumes cached artifacts from the previous stage and never
    repeats expensive work: valid transcripts/chunks/embeddings/Qdrant points
    are skipped.
  * Per-video checkpoint files in data/status/<video_id>.json make the pipeline
    resumable — an interrupted run continues from the first incomplete video.
  * A stage is only marked "completed" after its OUTPUT is actually validated
    (timestamps, text, chunk ids, embedding dimension, Qdrant point presence) —
    never merely because a file exists.
  * A failure in one video (id/title/url/stage/error recorded) stops only that
    video; processing continues. All failures are written to
    data/ingestion_report.json for `--retry-failed`.
  * Qdrant point IDs are deterministic UUIDv5 values derived from
    (video_id, chunk_id), so upserts are idempotent and duplicates are impossible.
  * Processing is strictly sequential to keep RAM/VRAM usage modest.
"""

import datetime
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from audio_downloader import AudioDownloader
from chunker import TranscriptChunker
from config import (
    DEFAULT_WHISPER_MODEL,
    INGESTION_REPORT,
    QDRANT_COLLECTION,
    STATUS_DIR,
)
from embedder import EmbeddingGenerator
from playlist_fetcher import PlaylistFetcher
from transcriber import AudioTranscriber
from utils import extract_playlist_id, load_json, save_json
from vector_store import QdrantVectorStore, generate_deterministic_id

STAGES: Tuple[str, ...] = ("transcription", "chunking", "embedding", "indexing")

_STAGE_ALIASES = {
    "transcription": "transcription",
    "transcript": "transcription",
    "chunking": "chunking",
    "chunk": "chunking",
    "embedding": "embedding",
    "embed": "embedding",
    "indexing": "indexing",
    "index": "indexing",
}


# ---------------------------------------------------------------------------
# Stage-output validators. A stage is only marked "completed" when these pass.
# ---------------------------------------------------------------------------

def _validate_transcript(data: Any) -> Tuple[bool, str]:
    """A valid transcript must be JSON with video_id and timestamped, non-empty segments."""
    if not isinstance(data, dict):
        return False, "transcript is not a JSON object"
    if not data.get("video_id"):
        return False, "missing 'video_id'"
    if not data.get("title"):
        return False, "missing 'title'"

    segments = data.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        return False, "no timestamped 'segments' found"
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"segment {i} is not an object"
        text = seg.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return False, f"segment {i} has no text"
        for key in ("start", "end", "start_time", "end_time"):
            if key not in seg:
                return False, f"segment {i} missing '{key}'"
    return True, "ok"


def _validate_chunks(data: Any) -> Tuple[bool, str]:
    """A valid chunk file must contain chunks with real text and timestamps."""
    if not isinstance(data, dict):
        return False, "chunk file is not a JSON object"
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or len(chunks) == 0:
        return False, "no chunks found"
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            return False, f"chunk {i} is not an object"
        for key in (
            "chunk_id",
            "video_id",
            "text",
            "start_sec",
            "end_sec",
            "start_time",
            "end_time",
            "youtube_url",
        ):
            if key not in chunk:
                return False, f"chunk {i} missing '{key}'"
        text = chunk.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return False, f"chunk {i} has empty text"
    return True, "ok"


def _validate_embeddings(data: Any) -> Tuple[bool, str]:
    """A valid embedding file must contain chunks that ALL carry a consistent-dimension vector."""
    if not isinstance(data, dict):
        return False, "embedding file is not a JSON object"
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or len(chunks) == 0:
        return False, "no embedded chunks found"
    dims = set()
    for i, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or len(embedding) == 0:
            return False, f"chunk {i} has no embedding vector"
        try:
            dims.add(len([float(v) for v in embedding]))
        except (TypeError, ValueError):
            return False, f"chunk {i} embedding is not numeric"
    if len(dims) != 1:
        return False, f"inconsistent embedding dimensions: {sorted(dims)}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Per-video checkpoint store (data/status/<video_id>.json)
# ---------------------------------------------------------------------------

def _default_status(video: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "video_id": video["video_id"],
        "title": video.get("title", "Unknown Title"),
        "url": video.get("url", f"https://www.youtube.com/watch?v={video['video_id']}"),
        "playlist_id": video.get("playlist_id", ""),
        "playlist_url": video.get("playlist_url", ""),
        "index": video.get("index", 0),
        "transcription": "pending",
        "chunking": "pending",
        "embedding": "pending",
        "indexing": "pending",
        "failed_stage": None,
        "error": None,
        "availability": "available",
    }


class StatusStore:
    """Reads/writes one lightweight JSON checkpoint per video."""

    def __init__(self, status_dir: Path = STATUS_DIR):
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def load(self, video: Dict[str, Any]) -> Dict[str, Any]:
        path = self.status_dir / f"{video['video_id']}.json"
        status = _default_status(video)
        if not path.exists():
            return status
        try:
            data = load_json(path)
        except Exception:
            return status
        for key in status:
            if key in data:
                status[key] = data[key]
        return status

    def save(self, status: Dict[str, Any]) -> None:
        save_json(status, self.status_dir / f"{status['video_id']}.json")

    def mark(
        self,
        status: Dict[str, Any],
        stage: str,
        state: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert stage in STAGES, f"unknown stage {stage}"
        assert state in ("pending", "completed", "failed"), f"unknown state {state}"
        status[stage] = state
        status["failed_stage"] = stage if state == "failed" else None
        status["error"] = error if state == "failed" else None
        self.save(status)
        return status


# ---------------------------------------------------------------------------
# Playlist ingestion orchestrator
# ---------------------------------------------------------------------------

class PlaylistIngestor:
    """
    Sequential, resumable, 4-stage playlist ingestion.

    - playlist_url / videos : discovery source. When `videos` is provided the
      provided list is used as-is (used by --retry-failed).
    - limit                 : only applies to freshly discovered playlists.
    - force_flags           : dict of {stage_name: True} forcing that stage to
      re-run. A re-run automatically forces all downstream stages fresh, so no
      stale cache is ever blindly reused.
    """

    def __init__(
        self,
        playlist_url: Optional[str] = None,
        videos: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        start_index: Optional[int] = None,
        force_flags: Optional[Dict[str, bool]] = None,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str] = None,
    ):
        self.playlist_url = playlist_url
        self.videos = videos
        self.limit = limit
        self.start_index = start_index
        self.discovered_total: int = len(videos) if videos else 0
        self.playlist_title: str = ""
        self.playlist_id: str = extract_playlist_id(playlist_url) if playlist_url else ""

        # Normalize force flags to canonical stage names.
        self.force_flags: Dict[str, bool] = {}
        for key, value in (force_flags or {}).items():
            canonical = _STAGE_ALIASES.get(str(key).lower())
            if canonical and value:
                self.force_flags[canonical] = True

        self.downloader = AudioDownloader(
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.chunker = TranscriptChunker()
        self.embedder = EmbeddingGenerator()
        self._status = StatusStore()
        self._store: Optional[QdrantVectorStore] = None

    # ------------------------------------------------------------------ utils

    def _ensure_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore()
        return self._store

    def _load_existing_report(self) -> Dict[str, Any]:
        if not INGESTION_REPORT.exists():
            return {}
        try:
            return load_json(INGESTION_REPORT)
        except Exception:
            return {}

    def _cached_data(self, path: Path, validator) -> Optional[Dict[str, Any]]:
        """Loads + validates an artifact; returns it only if it validates."""
        if not path.exists():
            return None
        try:
            data = load_json(path)
        except Exception:
            return None
        ok, _reason = validator(data)
        return data if ok else None

    def _count_indexed(self, video_id: str, embedding_path: Path) -> int:
        """Counts how many of this video's chunks are indexed in Qdrant."""
        try:
            emb = load_json(embedding_path)
        except Exception:
            return 0
        seen: set = set()
        unique_chunk_ids = []
        for c in emb.get("chunks", []):
            cid = c.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            unique_chunk_ids.append(cid)
        if not unique_chunk_ids:
            return 0
        point_ids = [
            generate_deterministic_id(video_id, cid) for cid in unique_chunk_ids
        ]
        try:
            return self._ensure_store().count_existing_points(point_ids, video_id=video_id)
        except Exception:
            return 0

    def _expected_chunk_count(self, embedding_path: Path) -> Optional[int]:
        try:
            data = load_json(embedding_path)
        except Exception:
            return None
        chunks = data.get("chunks", [])
        seen: set = set()
        valid_unique = 0
        for c in chunks:
            cid = c.get("chunk_id")
            if not cid:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            valid_unique += 1
        return valid_unique if valid_unique else None


# ------------------------------------------------------------- running

    def run(self, mode: str = "full") -> Dict[str, Any]:
        t0 = time.time()
        old_report = self._load_existing_report()

        if self.videos is None:
            fetcher = PlaylistFetcher()
            self.videos = fetcher.fetch_playlist_videos(
                self.playlist_url, limit=self.limit, start_index=self.start_index
            )
            self.discovered_total = (
                fetcher.last_total if fetcher.last_total > 0 else len(self.videos)
            )
            self.playlist_title = fetcher.last_playlist_title
            self.playlist_id = (
                fetcher.last_playlist_id or extract_playlist_id(self.playlist_url) or ""
            )
            self.playlist_url = self.playlist_url or ""
        else:
            old_playlist = old_report.get("playlist", {}) or {}
            self.discovered_total = (
                self.discovered_total
                or old_playlist.get("discovered_total")
                or len(self.videos)
            )
            self.playlist_title = self.playlist_title or old_playlist.get(
                "title", "YouTube Playlist"
            )
            self.playlist_id = self.playlist_id or old_playlist.get("id", "")
            self.playlist_url = self.playlist_url or old_playlist.get("url", "")

        if not self.videos:
            raise RuntimeError(
                "No videos to process: playlist discovery returned an empty list."
            )

        self._print_banner(mode)

        stats = {"successful": 0, "failed": 0, "skipped": 0}
        interrupted = False

        try:
            for i, video in enumerate(self.videos, 1):
                print("\n" + "-" * 42)
                print(
                    f"[{i:03d}/{len(self.videos)}] {video.get('title', video['video_id'])}"
                )

                try:
                    result = self._process_video(video)
                except Exception as e:  # last resort: a video can NEVER kill the run
                    result = self._fail_video_hard(video, e)
                    print(f"  [VIDEO-ERROR] {type(e).__name__}: {str(e)[:160]}")

                if result["outcome"] == "success":
                    stats["successful"] += 1
                    if result["all_skipped"]:
                        stats["skipped"] += 1
                else:
                    stats["failed"] += 1

                for stage in STAGES:
                    if stage not in result["stages"]:
                        continue
                    outcome, message = result["stages"][stage]
                    suffix = f" ({message})" if message else ""
                    print(f"  {stage:13s} : {outcome}{suffix}")
        except KeyboardInterrupt:
            # Per-video checkpoints are saved stage-by-stage on disk, so an
            # interrupted run resumes exactly from the first incomplete video.
            interrupted = True
            print(
                "\n[INTERRUPTED] Progress was saved. Re-run the same command "
                "to resume from the first incomplete video."
            )

        report = self._finalize_report(mode, stats, time.time() - t0)
        save_json(report, INGESTION_REPORT)
        self._print_summary(mode, stats, report, time.time() - t0)
        if interrupted:
            sys.exit(130)
        return report

    def _process_video(self, video: Dict[str, Any]) -> Dict[str, Any]:
        status = self._status.load(video)
        if status.get("availability") == "private":
            return {
                "stages": {s: ("skipped", "unavailable video") for s in STAGES},
                "outcome": "failed",
                "all_skipped": False,
            }
        force_next: set = set()
        stages: Dict[str, Tuple[str, str]] = {}
        outcome = "success"

        for stage in STAGES:
            force = self.force_flags.get(stage, False) or stage in force_next
            try:
                stage_outcome, message = self._run_stage(stage, video, status, force)
            except Exception as e:  # safety net: a stage never kills the playlist
                try:
                    self._status.mark(status, stage, "failed", error=str(e))
                except Exception:
                    pass  # status file unwritable; _fail_video_hard covers run level
                stage_outcome, message = "failed", str(e)

            stages[stage] = (stage_outcome, message)

            if stage_outcome == "failed":
                outcome = "failed"
                break
            if stage_outcome == "completed" and stage != STAGES[-1]:
                # Upstream changed -> downstream artifacts must be regenerated.
                force_next.add(STAGES[STAGES.index(stage) + 1])

        all_skipped = bool(stages) and all(s[0] == "skipped" for s in stages.values())
        return {"stages": stages, "outcome": outcome, "all_skipped": all_skipped}

    def _fail_video_hard(self, video: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """
        Last-resort failure isolation when _process_video itself raises (instead of
        returning a normal failed result). Records the failure in the per-video
        checkpoint (best-effort) and returns a synthetic failed result so the
        ingest loop can continue with the next video. Used for unusual failures
        (e.g. a status-file write crash) on top of the normal per-stage catches.
        """
        st = self._status.load(video)
        failed_stage = next(
            (s for s in STAGES if st.get(s) != "completed"), "transcription"
        )
        try:
            self._status.mark(st, failed_stage, "failed", error=str(error))
        except Exception:
            pass  # status disk is broken; the report still records what it can
        stages: Dict[str, Tuple[str, str]] = {
            s: (st.get(s, "pending"), "") for s in STAGES
        }
        stages[failed_stage] = ("failed", str(error))
        return {"stages": stages, "outcome": "failed", "all_skipped": False}

    def _run_stage(
        self, stage: str, video: Dict[str, Any], status: Dict[str, Any], force: bool
    ) -> Tuple[str, str]:
        if stage == "transcription":
            return self._stage_transcription(video, status, force)
        if stage == "chunking":
            return self._stage_chunking(video, status, force)
        if stage == "embedding":
            return self._stage_embedding(video, status, force)
        if stage == "indexing":
            return self._stage_indexing(video, status, force)
        raise ValueError(f"Unknown stage: {stage}")


# ----------------------------------------------------- stage: transcription

    def _stage_transcription(
        self, video: Dict[str, Any], status: Dict[str, Any], force: bool
    ) -> Tuple[str, str]:
        video_id = video["video_id"]
        transcript_path = self.transcriber.output_dir / f"{video_id}.json"

        if not force:
            if status.get("transcription") == "completed" and self._cached_data(
                transcript_path, _validate_transcript
            ):
                return "skipped", "cached transcript validated"
            cached = self._cached_data(transcript_path, _validate_transcript)
            if cached is not None:
                self._status.mark(status, "transcription", "completed")
                return (
                    "skipped",
                    f"adopted existing transcript ({len(cached.get('segments', []))} segments)",
                )

        try:
            audio_meta = self.downloader.download_audio(video_id)
            audio_meta["title"] = video.get(
                "title", audio_meta.get("title", "Unknown Title")
            )
            audio_meta["url"] = video.get(
                "url",
                audio_meta.get("url", f"https://www.youtube.com/watch?v={video_id}"),
            )
            # Fresh path: bypass the transcriber's own file-cache, otherwise a
            # corrupt cached transcript (which failed validation) would be
            # returned again and again forever.
            transcript = self.transcriber.transcribe(
                audio_meta, force_retranscribe=True
            )

            ok, reason = _validate_transcript(transcript)
            if not ok:
                raise RuntimeError(f"transcript failed validation: {reason}")

            self._status.mark(status, "transcription", "completed")
            return "completed", f"{len(transcript.get('segments', []))} segments"
        except Exception as e:
            error_msg = str(e)
            if any(
                m in error_msg.lower()
                for m in ["private video", "members only", "age-restricted"]
            ):
                status["availability"] = "private"
            self._status.mark(status, "transcription", "failed", error=error_msg)
            return "failed", error_msg

    # ----------------------------------------------------------- stage: chunking

    def _stage_chunking(
        self, video: Dict[str, Any], status: Dict[str, Any], force: bool
    ) -> Tuple[str, str]:
        video_id = video["video_id"]
        chunk_path = self.chunker.output_dir / f"{video_id}.json"
        transcript_path = self.transcriber.output_dir / f"{video_id}.json"

        if not force:
            if status.get("chunking") == "completed" and self._cached_data(
                chunk_path, _validate_chunks
            ):
                return "skipped", "cached chunks validated"
            cached = self._cached_data(chunk_path, _validate_chunks)
            if cached is not None:
                self._status.mark(status, "chunking", "completed")
                return (
                    "skipped",
                    f"adopted existing chunks ({len(cached.get('chunks', []))})",
                )

        try:
            if not transcript_path.exists():
                raise FileNotFoundError(f"transcript missing: {transcript_path.name}")
            transcript = load_json(transcript_path)
            ok, reason = _validate_transcript(transcript)
            if not ok:
                raise RuntimeError(f"transcript failed validation: {reason}")

            result = self.chunker.chunk_transcript(transcript)
            ok2, reason2 = _validate_chunks(result)
            if not ok2:
                raise RuntimeError(f"chunking failed validation: {reason2}")

            self._status.mark(status, "chunking", "completed")
            return "completed", f"{result['chunk_count']} chunks"
        except Exception as e:
            self._status.mark(status, "chunking", "failed", error=str(e))
            return "failed", str(e)


# ---------------------------------------------------------- stage: embedding

    def _stage_embedding(
        self, video: Dict[str, Any], status: Dict[str, Any], force: bool
    ) -> Tuple[str, str]:
        video_id = video["video_id"]
        embedding_path = self.embedder.output_dir / f"{video_id}.json"
        chunk_path = self.chunker.output_dir / f"{video_id}.json"

        if not force:
            if status.get("embedding") == "completed" and self._cached_data(
                embedding_path, _validate_embeddings
            ):
                return "skipped", "cached embeddings validated"
            cached = self._cached_data(embedding_path, _validate_embeddings)
            if cached is not None:
                self._status.mark(status, "embedding", "completed")
                return (
                    "skipped",
                    f"adopted existing embeddings ({len(cached.get('chunks', []))})",
                )

        try:
            if not chunk_path.exists():
                raise FileNotFoundError(f"chunk file missing: {chunk_path.name}")
            chunk_data = load_json(chunk_path)
            ok, reason = _validate_chunks(chunk_data)
            if not ok:
                raise RuntimeError(f"chunk data failed validation: {reason}")

            embedded = self.embedder.generate_embeddings_for_file(chunk_path)
            ok2, reason2 = _validate_embeddings(embedded)
            if not ok2:
                raise RuntimeError(f"embedding validation failed: {reason2}")

            self._status.mark(status, "embedding", "completed")
            return (
                "completed",
                f"{embedded['chunk_count']} vectors (dim {embedded['embedding_dimension']})",
            )
        except Exception as e:
            self._status.mark(status, "embedding", "failed", error=str(e))
            return "failed", str(e)

    # ---------------------------------------------------------- stage: indexing

    def _stage_indexing(
        self, video: Dict[str, Any], status: Dict[str, Any], force: bool
    ) -> Tuple[str, str]:
        video_id = video["video_id"]
        embedding_path = self.embedder.output_dir / f"{video_id}.json"

        if not force:
            if status.get("indexing") == "completed":
                existing = self._count_indexed(video_id, embedding_path)
                expected = self._expected_chunk_count(embedding_path)
                if expected is not None and existing >= expected:
                    return "skipped", f"Qdrant already has {existing}/{expected} points"
            cached = self._cached_data(embedding_path, _validate_embeddings)
            if cached is not None:
                existing = self._count_indexed(video_id, embedding_path)
                expected = self._expected_chunk_count(embedding_path)
                if expected is not None and existing >= expected:
                    self._status.mark(status, "indexing", "completed")
                    return (
                        "skipped",
                        f"adopted: Qdrant already has {existing}/{expected} points",
                    )

        try:
            if not embedding_path.exists():
                raise FileNotFoundError(
                    f"embedding file missing: {embedding_path.name}"
                )
            embedded = load_json(embedding_path)
            ok, reason = _validate_embeddings(embedded)
            if not ok:
                raise RuntimeError(f"embedding data failed validation: {reason}")

            store = self._ensure_store()
            result = store.index_embedding_file(embedding_path)
            after = self._count_indexed(video_id, embedding_path)
            expected = self._expected_chunk_count(embedding_path)
            if after < expected:
                raise RuntimeError(
                    f"expected {expected} Qdrant points but only {after} confirmed"
                )

            self._status.mark(status, "indexing", "completed")
            return (
                "completed",
                f"upserted {result['vectors_indexed']} vectors (verified {after})",
            )
        except Exception as e:
            self._status.mark(status, "indexing", "failed", error=str(e))
            return "failed", str(e)


# ------------------------------------------------------------- report

    def _video_public(self, video: Dict[str, Any]) -> Dict[str, Any]:
        """Authoritative per-video summary merged from the live checkpoint."""
        st = self._status.load(video)
        return {
            "video_id": video["video_id"],
            "title": st.get("title") or video.get("title", ""),
            "url": st.get("url") or video.get("url", ""),
            "playlist_id": st.get("playlist_id") or video.get("playlist_id", ""),
            "playlist_url": st.get("playlist_url") or video.get("playlist_url", ""),
            "index": video.get("index") or st.get("index") or 0,
            "transcription": st.get("transcription", "pending"),
            "chunking": st.get("chunking", "pending"),
            "embedding": st.get("embedding", "pending"),
            "indexing": st.get("indexing", "pending"),
            "availability": st.get("availability", "available"),
        }

    def _finalize_report(
        self, mode: str, stats: Dict[str, int], elapsed: float
    ) -> Dict[str, Any]:
        old = self._load_existing_report()
        old_playlist = old.get("playlist", {}) or {}
        old_videos: Dict[str, Any] = {
            v.get("video_id"): v for v in old.get("videos", []) if v.get("video_id")
        }

        # Union of previously-known and current videos (enables retry-merge).
        known: Dict[str, Any] = dict(old_videos)
        for video in self.videos:
            known[video["video_id"]] = self._video_public(video)

        videos_list = [
            known[k]
            for k in sorted(known, key=lambda vid: known[vid].get("index") or 0)
        ]

        failed_videos: List[Dict[str, Any]] = []
        unavailable_videos: List[Dict[str, Any]] = []
        for v in videos_list:
            st = self._status.load(v)
            if st.get("failed_stage"):
                entry = {
                    "video_id": v["video_id"],
                    "title": st.get("title") or v.get("title", ""),
                    "url": st.get("url") or v.get("url", ""),
                    "index": v.get("index") or 0,
                    "failed_stage": st.get("failed_stage"),
                    "error": st.get("error") or "",
                }
                if st.get("availability") == "private":
                    entry["availability"] = "private"
                    unavailable_videos.append(entry)
                else:
                    failed_videos.append(entry)

        counts = self._dataset_counts()

        playlist = {
            "url": self.playlist_url or old_playlist.get("url", ""),
            "id": self.playlist_id or old_playlist.get("id", ""),
            "title": self.playlist_title or old_playlist.get("title", ""),
            "discovered_total": (
                self.discovered_total
                or old_playlist.get("discovered_total")
                or len(videos_list)
            ),
            "limit": self.limit,
        }

        return {
            "playlist": playlist,
            "last_run": {
                "mode": mode,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed, 2),
                "processed_this_run": len(self.videos),
                "successful_this_run": stats["successful"],
                "failed_this_run": stats["failed"],
                "skipped_this_run": stats["skipped"],
            },
            "counts": counts,
            "failed_videos": failed_videos,
            "unavailable_videos": unavailable_videos,
            "videos": videos_list,
        }

    def _dataset_counts(self) -> Dict[str, int]:
        """Counts validated artifacts on disk + total Qdrant vectors."""
        transcripts = sum(
            1
            for f in self.transcriber.output_dir.glob("*.json")
            if self._cached_data(f, _validate_transcript) is not None
        )
        chunks = sum(
            1
            for f in self.chunker.output_dir.glob("*.json")
            if self._cached_data(f, _validate_chunks) is not None
        )
        embeddings = sum(
            1
            for f in self.embedder.output_dir.glob("*.json")
            if self._cached_data(f, _validate_embeddings) is not None
        )
        try:
            vectors = self._ensure_store().count_all_points()
        except Exception:
            vectors = 0
        return {
            "transcripts": transcripts,
            "chunks": chunks,
            "embeddings": embeddings,
            "qdrant_vectors": vectors,
        }


# ------------------------------------------------------------- printing

    def _print_banner(self, mode: str) -> None:
        print("\n" + "=" * 42)
        print(" STRIVER A2Z INGESTION")
        print("=" * 42)
        print(f"Playlist URL     : {self.playlist_url or '(from prior report)'}")
        print(f"Playlist ID      : {self.playlist_id or '(unknown)'}")
        print(f"Playlist title   : {self.playlist_title or '(unknown)'}")
        print(f"Videos in list   : {self.discovered_total}")
        if self.limit and self.limit > 0:
            print(f"Limit            : processing first {self.limit}")
        print(f"Mode             : {mode}")
        print("Resume behavior  : completed stages are skipped (validated)")
        print(
            f"Force            : "
            f"{', '.join(sorted(self.force_flags)) if self.force_flags else 'none'}"
        )
        print(
            f"Whisper model    : {self.transcriber.model_size} "
            f"({self.transcriber.device}/{self.transcriber.compute_type})"
        )
        print(
            f"Qdrant collection: "
            f"{self._store.collection_name if self._store else QDRANT_COLLECTION}"
        )
        cookie_src = self.downloader.cookies_file or self.downloader.cookies_from_browser
        print(
            f"YouTube auth      : "
            f"{f'browser cookies ({cookie_src})' if cookie_src else 'no cookies (anonymous)'}"
        )

    def _print_summary(
        self, mode: str, stats: Dict[str, int], report: Dict[str, Any], elapsed: float
    ) -> None:
        counts = report.get("counts", {})
        failed = report.get("failed_videos", [])

        print("\n" + "=" * 42)
        print(" PLAYLIST INGESTION COMPLETE")
        print("=" * 42)
        if mode == "retry":
            print(f"Retried videos   : {len(self.videos)}")
        print(f"Total videos     : {len(self.videos)}")
        print(f"Successful       : {stats['successful']}")
        print(f"Failed           : {stats['failed']}")
        print(f"Skipped (cached) : {stats['skipped']}")
        print(f"Transcripts      : {counts.get('transcripts', 0)}")
        print(f"Chunks           : {counts.get('chunks', 0)}")
        print(f"Embeddings       : {counts.get('embeddings', 0)}")
        print(f"Qdrant vectors   : {counts.get('qdrant_vectors', 0)}")
        print(f"Elapsed          : {elapsed:.1f}s")

        if failed:
            print("\nFailed videos:")
            for i, f in enumerate(failed, 1):
                print(f"  {i}. {f.get('title')} ({f.get('video_id')})")
                print(
                    f"     stage: {f.get('failed_stage')} | "
                    f"error: {str(f.get('error', ''))[:160]}"
                )
        else:
            print("Failed videos    : none")

        print(f"\nReport saved     : {INGESTION_REPORT.resolve()}")

    # ------------------------------------------------------------- retry

    def retry_failed(self) -> Dict[str, Any]:
        """
        Reads data/ingestion_report.json and retries ONLY videos whose status is
        failed. Successfully processed videos and unavailable videos are never
        touched again.
        """
        old = self._load_existing_report()
        failed = old.get("failed_videos", [])

        unavailable_ids = set()
        for v in old.get("videos", []):
            video_id = v.get("video_id")
            if video_id:
                st = self._status.load(v)
                if st.get("availability") == "private":
                    unavailable_ids.add(video_id)
        for u in old.get("unavailable_videos", []):
            vid = u.get("video_id")
            if vid:
                unavailable_ids.add(vid)

        failed = [f for f in failed if f.get("video_id") not in unavailable_ids]

        if not failed:
            print(
                "\n[RETRY] No failed videos found in "
                "data/ingestion_report.json. Nothing to retry."
            )
            return {}

        print(f"\n[RETRY] {len(failed)} failed video(s) will be retried:")
        for f in failed:
            print(
                f"  - {f.get('title')} ({f.get('video_id')}) "
                f"— failed at {f.get('failed_stage')}"
            )

        known: Dict[str, Any] = {
            v.get("video_id"): v for v in old.get("videos", []) if v.get("video_id")
        }
        playlist_ctx = old.get("playlist", {}) or {}

        videos: List[Dict[str, Any]] = []
        for f in failed:
            video_id = f.get("video_id")
            k = known.get(video_id, {})
            videos.append(
                {
                    "video_id": video_id,
                    "title": f.get("title") or k.get("title") or video_id,
                    "url": f.get("url")
                    or k.get("url")
                    or f"https://www.youtube.com/watch?v={video_id}",
                    "playlist_id": k.get("playlist_id") or playlist_ctx.get("id", ""),
                    "playlist_url": k.get("playlist_url")
                    or k.get("playlist_url")
                    or playlist_ctx.get("url", ""),
                    "index": k.get("index") or f.get("index") or 0,
                }
            )

        self.videos = videos
        self.limit = None  # A discovery limit must never be re-applied on retry.
        return self.run(mode="retry")


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _count_valid_in_dir(directory: Path, validator) -> int:
    """Counts JSON files in `directory` whose content passes `validator`."""
    if not Path(directory).exists():
        return 0
    count = 0
    for f in sorted(Path(directory).glob("*.json")):
        try:
            data = load_json(f)
        except Exception:
            continue
        if validator(data)[0]:
            count += 1
    return count


def dataset_info() -> Dict[str, Any]:
    """
    Human-readable summary of the current dataset:
    playlist metadata, validated artifact counts, Qdrant vectors,
    failed videos, and approximate storage sizes.
    """
    from config import CHUNKS_DIR, DOWNLOADS_DIR, EMBEDDINGS_DIR, TRANSCRIPTS_DIR

    print("\n" + "=" * 42)
    print(" DATASET INFO (Phase 7)")
    print("=" * 42)

    report: Dict[str, Any] = {}
    if INGESTION_REPORT.exists():
        try:
            report = load_json(INGESTION_REPORT)
        except Exception:
            report = {}

    playlist = report.get("playlist", {}) or {}
    print(f"Playlist title    : {playlist.get('title', '(not ingested yet via Phase 7)')}")
    print(f"Playlist URL      : {playlist.get('url', '')}")
    print(f"Playlist ID       : {playlist.get('id', '')}")
    print(
        f"Videos discovered : "
        f"{playlist.get('discovered_total', '(unknown - run --ingest-playlist)')}"
    )
    limit = playlist.get("limit")
    print(f"Limit applied     : {limit if limit else 'none (full playlist)'}")

    print("\n--- Validated artifacts ---")
    transcripts = _count_valid_in_dir(TRANSCRIPTS_DIR, _validate_transcript)
    chunks = _count_valid_in_dir(CHUNKS_DIR, _validate_chunks)
    embeddings = _count_valid_in_dir(EMBEDDINGS_DIR, _validate_embeddings)
    try:
        vectors = QdrantVectorStore().count_all_points()
    except Exception:
        vectors = 0

    failed_videos = report.get("failed_videos", []) or []
    print(f"Transcripts (valid) : {transcripts}")
    print(f"Chunks (valid)      : {chunks}")
    print(f"Embeddings (valid)  : {embeddings}")
    print(f"Qdrant vectors      : {vectors}")
    print(f"Failed videos       : {len(failed_videos)}")

    last_run = report.get("last_run", {}) or {}
    if last_run:
        print(
            f"Last ingestion      : "
            f"{last_run.get('timestamp')} (mode: {last_run.get('mode')})"
        )

    print("\n--- Approximate storage ---")
    for name, directory in (
        ("data/downloads/", DOWNLOADS_DIR),
        ("data/transcripts/", TRANSCRIPTS_DIR),
        ("data/chunks/", CHUNKS_DIR),
        ("data/embeddings/", EMBEDDINGS_DIR),
    ):
        size_bytes = sum(
            f.stat().st_size for f in Path(directory).iterdir() if f.is_file()
        )
        print(f"  {name:20s}: {size_bytes / (1024 * 1024):.2f} MB")

    if failed_videos:
        print("\n--- Failed videos ---")
        for i, f in enumerate(failed_videos[:20], 1):
            print(
                f"  {i}. {f.get('title')} ({f.get('video_id')}) - "
                f"{f.get('failed_stage')}: {str(f.get('error', ''))[:120]}"
            )
        if len(failed_videos) > 20:
            print(f"  ... and {len(failed_videos) - 20} more")

    print("=" * 42 + "\n")
    return report