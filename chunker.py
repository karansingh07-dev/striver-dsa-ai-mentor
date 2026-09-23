from pathlib import Path
from typing import Dict, Any, List, Optional
import os

from config import (
    TRANSCRIPTS_DIR,
    CHUNKS_DIR,
    CHUNK_SECONDS,
    CHUNK_OVERLAP_SECONDS,
)
from utils import (
    format_seconds_to_timestamp,
    generate_youtube_timestamp_url,
    save_json,
    load_json,
)


class TranscriptChunker:
    """
    Time-aware chunker that groups contiguous Whisper transcript segments
    into time windows (default ~75s with ~15s overlap) while preserving
    exact segment boundaries, metadata, and clickable YouTube timestamp URLs.
    """

    def __init__(
        self,
        chunk_seconds: float = CHUNK_SECONDS,
        overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
        output_dir: Path = CHUNKS_DIR,
    ):
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a transcript dictionary and builds time-bounded chunks.
        Preserves video_id, title, start_sec, end_sec, text, and youtube_url.
        """
        video_id = transcript["video_id"]
        title = transcript.get("title", "Unknown Title")
        segments = transcript.get("segments", [])

        if not segments:
            return {
                "video_id": video_id,
                "title": title,
                "chunk_count": 0,
                "chunk_seconds": self.chunk_seconds,
                "chunk_overlap_seconds": self.overlap_seconds,
                "chunks": []
            }

        chunks = []
        n = len(segments)
        i = 0

        while i < n:
            start_sec = round(segments[i]["start"], 2)
            chunk_segs = []
            curr_idx = i

            while curr_idx < n:
                seg = segments[curr_idx]
                chunk_segs.append(seg)
                curr_end = round(seg["end"], 2)

                # Stop chunk when reaching target duration
                if (curr_end - start_sec) >= self.chunk_seconds and len(chunk_segs) > 0:
                    break

                curr_idx += 1

            if not chunk_segs:
                break

            end_sec = round(chunk_segs[-1]["end"], 2)
            chunk_text = " ".join(s["text"].strip() for s in chunk_segs if s["text"].strip())
            # Normalize whitespace
            chunk_text = " ".join(chunk_text.split())

            if chunk_text:
                chunk_id = f"{video_id}_{int(start_sec)}_{int(end_sec)}"
                yt_url = generate_youtube_timestamp_url(video_id, start_sec)

                chunk_payload = {
                    "chunk_id": chunk_id,
                    "video_id": video_id,
                    "title": title,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "start_time": format_seconds_to_timestamp(start_sec),
                    "end_time": format_seconds_to_timestamp(end_sec),
                    "text": chunk_text,
                    "youtube_url": yt_url,
                    "segment_count": len(chunk_segs)
                }
                chunks.append(chunk_payload)

            # Calculate next start segment index based on overlap
            next_i = curr_idx + 1
            for j in range(i + 1, len(chunk_segs) + i):
                if j < n and (end_sec - round(segments[j]["start"], 2)) <= self.overlap_seconds:
                    next_i = j
                    break

            if next_i <= i:
                next_i = i + 1

            i = next_i

        result = {
            "video_id": video_id,
            "title": title,
            "chunk_count": len(chunks),
            "chunk_seconds": self.chunk_seconds,
            "chunk_overlap_seconds": self.overlap_seconds,
            "chunks": chunks
        }

        out_file = self.output_dir / f"{video_id}.json"
        save_json(result, out_file)
        return result

    def process_all_transcripts(
        self,
        transcripts_dir: Path = TRANSCRIPTS_DIR,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Scans data/transcripts/ directory and chunks all JSON files.
        Skips already chunked files unless force=True.
        """
        transcript_files = list(Path(transcripts_dir).glob("*.json"))

        if not transcript_files:
            print("[CHUNKER] No transcript files found in data/transcripts/.")
            return {"transcripts_processed": 0, "total_chunks_created": 0}

        print(f"Found {len(transcript_files)} transcripts in '{transcripts_dir}'.\n")

        processed_count = 0
        total_chunks = 0

        for idx, t_file in enumerate(transcript_files, 1):
            vid_id = t_file.stem
            chunk_file = self.output_dir / f"{vid_id}.json"

            try:
                t_data = load_json(t_file)
            except Exception as e:
                print(f"[{idx}/{len(transcript_files)}] Failed to load '{t_file.name}': {e}")
                continue

            title = t_data.get("title", vid_id)
            segment_count = len(t_data.get("segments", []))

            # Cache check
            if not force and chunk_file.exists() and chunk_file.stat().st_size > 0:
                cached_chunk_data = load_json(chunk_file)
                c_count = cached_chunk_data.get("chunk_count", 0)
                total_chunks += c_count
                processed_count += 1
                print(f"[{idx}/{len(transcript_files)}] {title} ({vid_id})")
                print(f"   ↳ [CACHE HIT] {segment_count} segments → {c_count} chunks (Loaded from cache)\n")
                continue

            res = self.chunk_transcript(t_data)
            c_count = res["chunk_count"]
            total_chunks += c_count
            processed_count += 1

            print(f"[{idx}/{len(transcript_files)}] {title} ({vid_id})")
            print(f"   ↳ {segment_count} segments → {c_count} chunks\n")

        print("=" * 50)
        print("CHUNKING COMPLETE")
        print(f"Transcripts:    {processed_count}")
        print(f"Chunks created: {total_chunks}")
        print("=" * 50)

        return {
            "transcripts_processed": processed_count,
            "total_chunks_created": total_chunks
        }
