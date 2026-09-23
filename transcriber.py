import time
from pathlib import Path
from typing import Dict, Any
from faster_whisper import WhisperModel

from config import (
    TRANSCRIPTS_DIR,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_COMPUTE_TYPE,
)
from utils import (
    format_seconds_to_timestamp,
    generate_youtube_timestamp_url,
    save_json,
    load_json,
)


class AudioTranscriber:
    """
    Transcribes audio files into timestamped segments using faster-whisper.
    Saves and loads JSON transcripts with caching.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        output_dir: Path = TRANSCRIPTS_DIR,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _get_model(self) -> WhisperModel:
        """Lazy loader for Whisper model.

        Guarantees a SINGLE model instance per process: it is loaded on first
        use and reused for every subsequent video (saves RAM/VRAM + load time).
        If CUDA initialisation fails, automatically falls back to CPU (int8)
        so ingestion still makes progress without any signed-in configuration.
        """
        if self._model is None:
            try:
                print(f"[TRANSCRIPTION] Loading Whisper model '{self.model_size}' ({self.device}/{self.compute_type})...")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
            except Exception as exc:
                if self.device != "cuda":
                    raise
                print(
                    f"[TRANSCRIPTION] CUDA unavailable ({exc.__class__.__name__}: {exc}); "
                    "falling back to CPU (int8) for this run."
                )
                self.device = "cpu"
                self.compute_type = "int8"
                print(f"[TRANSCRIPTION] Loading Whisper model '{self.model_size}' ({self.device}/{self.compute_type})...")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
        return self._model

    @staticmethod
    def _is_cuda_library_error(exc: Exception) -> bool:
        """True when the error means a CUDA library could not be loaded
        (e.g. a missing cuBLAS DLL surfaced at first GPU compute)."""
        msg = str(exc).lower()
        return any(
            token in msg
            for token in ("cublas", "cudnn", "cudart", "cannot be loaded", "not found")
        )

    def _run_transcription(self, video_id: str, audio_path: str):
        """Run inference for one audio file; return (formatted_segments, info).

        If the CUDA stack fails the FIRST GPU compute (e.g. cuBLAS DLL missing),
        falls back to CPU (int8) exactly once: the CPU model is memoised, so the
        retry — and every later video — uses CPU for the rest of the process.
        """
        model = self._get_model()
        try:
            segments_generator, info = model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True  # Voice Activity Detection removes silent sections
            )
            segments = self._collect_segments(video_id, segments_generator)
            return segments, info
        except Exception as exc:
            if not (self.device == "cuda" and self._is_cuda_library_error(exc)):
                raise
            print(
                f"[TRANSCRIPTION] CUDA library error ({exc.__class__.__name__}: {str(exc)[:120]}); "
                "falling back to CPU (int8) and retrying this video once."
            )
            self.device = "cpu"
            self.compute_type = "int8"
            self._model = None  # drop any partially initialised CUDA model
            model = self._get_model()
            segments_generator, info = model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True
            )
            segments = self._collect_segments(video_id, segments_generator)
            return segments, info

    def _collect_segments(self, video_id: str, segments_generator):
        """Format raw Whisper segments into the timestamped transcript schema."""
        formatted_segments = []
        for idx, segment in enumerate(segments_generator):
            clean_text = segment.text.strip()
            if not clean_text:
                continue

            start_sec = round(segment.start, 2)
            end_sec = round(segment.end, 2)
            yt_timestamp_url = generate_youtube_timestamp_url(video_id, start_sec)

            segment_data = {
                "segment_id": idx,
                "start": start_sec,
                "end": end_sec,
                "start_time": format_seconds_to_timestamp(start_sec),
                "end_time": format_seconds_to_timestamp(end_sec),
                "text": clean_text,
                "youtube_url": yt_timestamp_url
            }
            formatted_segments.append(segment_data)
        return formatted_segments

    def transcribe(self, meta: Dict[str, Any], force_retranscribe: bool = False) -> Dict[str, Any]:
        """
        Transcribes the audio file specified in `meta["audio_path"]`.
        Checks if JSON transcript already exists in cache.
        """
        video_id = meta["video_id"]
        json_file = self.output_dir / f"{video_id}.json"

        # Check Cache
        if not force_retranscribe and json_file.exists() and json_file.stat().st_size > 0:
            print(f"[CACHE] Loaded transcript from cache: {json_file.name}")
            return load_json(json_file)

        audio_path = meta.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found for transcription: '{audio_path}'")

        print(f"[TRANSCRIPTION] Transcribing audio: {audio_path} ...")
        start_clock = time.time()

        model = self._get_model()

        formatted_segments, info = self._run_transcription(video_id, audio_path)

        elapsed_time = round(time.time() - start_clock, 2)

        transcript_result = {
            "video_id": video_id,
            "title": meta.get("title", "Unknown Title"),
            "url": meta.get("url", f"https://www.youtube.com/watch?v={video_id}"),
            "duration": meta.get("duration", 0),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "transcription_time_seconds": elapsed_time,
            "total_segments": len(formatted_segments),
            "segments": formatted_segments
        }

        # Save to JSON cache
        save_json(transcript_result, json_file)
        print(f"[SUCCESS] Transcribed {len(formatted_segments)} segments in {elapsed_time}s. Saved: {json_file.name}")

        return transcript_result
