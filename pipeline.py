from pathlib import Path
from typing import Dict, Any, Optional, List

import yt_dlp

from audio_downloader import AudioDownloader
from transcriber import AudioTranscriber
from config import DEFAULT_WHISPER_MODEL


class LecturePipeline:
    """
    High-level pipeline for:
    YouTube video -> Audio -> Whisper -> Timestamped JSON
    """

    def __init__(self, whisper_model: str = DEFAULT_WHISPER_MODEL):
        self.downloader = AudioDownloader()
        self.transcriber = AudioTranscriber(model_size=whisper_model)

    def process_url(
        self,
        youtube_url_or_id: str,
        force_retranscribe: bool = False
    ) -> Dict[str, Any]:

        print("\n==========================================")
        print(f" Processing Lecture: {youtube_url_or_id}")
        print("==========================================")

        # Step 1: Download audio
        audio_meta = self.downloader.download_audio(youtube_url_or_id)

        # Step 2: Transcribe / load cached transcript
        transcript_data = self.transcriber.transcribe(
            audio_meta,
            force_retranscribe=force_retranscribe
        )

        return transcript_data

    def get_playlist_videos(
        self,
        playlist_url: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract video IDs and basic information from a YouTube playlist.

        Does NOT download videos or audio.
        """

        print("\n==========================================")
        print(" Fetching YouTube Playlist")
        print("==========================================")
        print(f"Playlist: {playlist_url}")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "ignoreerrors": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    playlist_url,
                    download=False
                )

            entries = info.get("entries", [])

            videos = []

            for entry in entries:
                if not entry:
                    continue

                video_id = entry.get("id")

                if not video_id:
                    continue

                title = entry.get("title", "Unknown Title")

                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })

                if limit and len(videos) >= limit:
                    break

            print(f"\n[PLAYLIST] Found {len(videos)} videos to process.")

            for index, video in enumerate(videos, start=1):
                print(
                    f"{index}. {video['title']} "
                    f"({video['video_id']})"
                )

            return videos

        except Exception as e:
            raise RuntimeError(
                f"Failed to read playlist: {e}"
            ) from e

    def process_playlist(
        self,
        playlist_url: str,
        limit: Optional[int] = None,
        force_retranscribe: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple videos from a YouTube playlist.

        Each video is processed independently.
        If one video fails, the remaining videos continue.
        """

        videos = self.get_playlist_videos(
            playlist_url,
            limit=limit
        )

        if not videos:
            raise RuntimeError(
                "No videos were found in the playlist."
            )

        successful = []
        failed = []

        print("\n==========================================")
        print(" STARTING PLAYLIST INGESTION")
        print("==========================================")

        for index, video in enumerate(videos, start=1):

            print("\n")
            print("#" * 60)
            print(f"VIDEO {index}/{len(videos)}")
            print(f"TITLE: {video['title']}")
            print(f"ID: {video['video_id']}")
            print("#" * 60)

            try:
                result = self.process_url(
                    video["url"],
                    force_retranscribe=force_retranscribe
                )

                successful.append({
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "result": result
                })

                print(
                    f"\n[SUCCESS] "
                    f"{video['title']}"
                )

            except Exception as e:

                failed.append({
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "error": str(e)
                })

                print(
                    f"\n[FAILED] "
                    f"{video['title']}"
                )
                print(f"Reason: {e}")

                # Continue with next video
                continue

        print("\n")
        print("=" * 60)
        print(" PLAYLIST INGESTION SUMMARY")
        print("=" * 60)

        print(f"Total videos: {len(videos)}")
        print(f"Successful:   {len(successful)}")
        print(f"Failed:       {len(failed)}")

        if failed:
            print("\n--- Failed Videos ---")

            for item in failed:
                print(
                    f"- {item['title']} "
                    f"({item['video_id']})"
                )
                print(f"  Error: {item['error']}")

        print("\nPlaylist processing completed.")

        return {
            "total": len(videos),
            "successful": successful,
            "failed": failed
        }