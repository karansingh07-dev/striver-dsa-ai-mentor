import yt_dlp
from typing import List, Dict, Any, Optional
from audio_downloader import get_ffmpeg_path
from utils import extract_playlist_id


class PlaylistFetcher:
    """
    Extracts video metadata (ID, title, URL) from a YouTube playlist URL.

    Each returned video entry additionally carries playlist-level metadata
    (playlist_id, playlist_url, playlist_title) and its 1-based position
    inside the playlist (`index`) so ingestion can preserve ordering.
    """

    def __init__(self) -> None:
        # Total number of videos discovered before any limit was applied.
        self.last_total: int = 0
        self.last_playlist_title: str = "YouTube Playlist"
        self.last_playlist_id: str = ""

    def fetch_playlist_videos(
        self, playlist_url: str, limit: Optional[int] = None, start_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all video items from a YouTube playlist.
        Optionally skips the first ``start_index`` entries and then limits
        the number of returned videos to ``limit``.
        """
        print(f"[PLAYLIST] Extracting video list from playlist...")

        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
        }

        ffmpeg_bin = get_ffmpeg_path()
        if ffmpeg_bin:
            ydl_opts['ffmpeg_location'] = ffmpeg_bin

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                if not info:
                    raise ValueError(f"Could not extract information from playlist: '{playlist_url}'")

                entries = info.get("entries", [])
                if not entries:
                    raise ValueError("No video entries found in the specified playlist.")

                playlist_title = info.get("title", "YouTube Playlist")
                playlist_id = info.get("id") or extract_playlist_id(playlist_url) or ""
                self.last_playlist_title = playlist_title
                self.last_playlist_id = playlist_id

                videos = []

                for entry in entries:
                    if not entry:
                        continue

                    video_id = entry.get("id")
                    title = entry.get("title", "Untitled Video")
                    url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"

                    if video_id:
                        videos.append({
                            "video_id": video_id,
                            "title": title,
                            "url": url,
                            "playlist_title": playlist_title,
                            "playlist_id": playlist_id,
                            "playlist_url": playlist_url,
                            "index": len(videos) + 1,
                        })

                self.last_total = len(videos)

                if start_index and start_index > 0:
                    videos = videos[start_index:]

                if limit and limit > 0:
                    print(f"[PLAYLIST] Limit applied: Processing {len(videos)} videos starting from offset {start_index or 0}.")
                    videos = videos[:limit]

                print(f"[PLAYLIST] Successfully fetched {len(videos)} video entries.")
                return videos

        except Exception as e:
            raise RuntimeError(f"Failed to extract playlist metadata: {str(e)}") from e
