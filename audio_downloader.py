import os
from pathlib import Path
from typing import Any, Dict, Optional
import yt_dlp

from config import (
    AUDIO_FORMAT,
    DOWNLOADS_DIR,
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_FROM_BROWSER,
)
from utils import extract_youtube_id, generate_youtube_timestamp_url

# Browsers recognized by yt-dlp's --cookies-from-browser (lowercase names).
# Passed through as-is; yt-dlp reports a clear error for anything it cannot read.
SUPPORTED_COOKIE_BROWSERS = {
    "brave",
    "chrome",
    "chromium",
    "edge",
    "firefox",
    "opera",
    "safari",
    "vivaldi",
    "whale",
}

# Substrings matching yt-dlp failures caused by the cookie SOURCE itself being
# unreadable (e.g. Chrome's cookie DB is exclusively locked by a running Chrome
# - see yt-dlp issue #7271 - or the cookies file cannot be read). These are
# NOT YouTube auth problems: the correct action is to retry anonymously.
_COOKIE_UNUSABLE_MARKERS = (
    "could not copy",
    "could not find",
    "permission denied",
    "is in use",
    "locked",
    "database could not",
)


def _is_cookie_unusable_error(err_text: str) -> bool:
    """True when the error means the cookie source could not be read at all."""
    lowered = (err_text or "").lower()
    return "cookie" in lowered and any(
        marker in lowered for marker in _COOKIE_UNUSABLE_MARKERS
    )

# Helper to get ffmpeg path if imageio_ffmpeg is installed
def get_ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class AudioDownloader:
    """
    Downloads audio from YouTube videos using yt-dlp with caching support.
    """

    def __init__(
        self,
        output_dir: Path = DOWNLOADS_DIR,
        audio_format: str = AUDIO_FORMAT,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cookie authentication is fully configurable (env vars in .env or via
        # constructor / CLI flags). Nothing is ever hardcoded in source code.
        self.cookies_from_browser = (
            (cookies_from_browser or YTDLP_COOKIES_FROM_BROWSER or "").strip().lower()
        )
        self.cookies_file = (cookies_file or YTDLP_COOKIES_FILE or "").strip()

        # Remains True while the configured cookie source is usable. Once an
        # attempt proves the source unreadable (locked browser DB, missing
        # file, ...), downloads fall back to anonymous for the rest of the run
        # so we never re-attempt the broken cookie read per video.
        self._cookies_ok = bool(self.cookies_file or self.cookies_from_browser)

        self._log_cookie_status()

    # ------------------------------------------------------------- cookie auth

    def _log_cookie_status(self) -> None:
        """Clear, unambiguous log of whether browser cookies are being used."""
        if self.cookies_from_browser and self.cookies_file:
            print(
                "[COOKIES] WARNING: both YTDLP_COOKIES_FROM_BROWSER and "
                "YTDLP_COOKIES_FILE are set - the cookies FILE wins."
            )
        if self.cookies_file:
            print(
                f"[COOKIES] YouTube auth via cookies file: {self.cookies_file} "
                "(yt-dlp --cookies)"
            )
        elif self.cookies_from_browser:
            if self.cookies_from_browser not in SUPPORTED_COOKIE_BROWSERS:
                print(
                    f"[COOKIES] WARNING: browser '{self.cookies_from_browser}' is not "
                    f"in the known set {sorted(SUPPORTED_COOKIE_BROWSERS)}; passing it "
                    "to yt-dlp anyway (yt-dlp will report a clear error if unsupported)."
                )
            print(
                f"[COOKIES] YouTube auth via browser cookies: "
                f"{self.cookies_from_browser} "
                f"(yt-dlp --cookies-from-browser {self.cookies_from_browser})"
            )
        else:
            print(
                "[COOKIES] No browser cookies configured (anonymous downloads). "
                "If YouTube reports 'Sign in to confirm you're not a bot', set "
                "YTDLP_COOKIES_FROM_BROWSER=chrome in .env (or run with "
                "--yt-cookies-browser chrome) and retry with --retry-failed."
            )

    def _build_ydl_opts(
        self,
        extra: Optional[Dict[str, Any]] = None,
        include_cookies: bool = True,
    ) -> Dict[str, Any]:
        """Base yt-dlp options + optional cookie auth + extras.

        `include_cookies=False` produces a fully anonymous option set used by
        the automatic fallback when the configured cookie source is unreadable.
        """
        opts: Dict[str, Any] = {"quiet": True, "no_warnings": True}
        if extra:
            opts.update(extra)

        if include_cookies:
            # A cookies FILE takes precedence over browser cookies when both are set.
            if self.cookies_file:
                opts["cookiefile"] = self.cookies_file
            elif self.cookies_from_browser:
                opts["cookiesfrombrowser"] = (self.cookies_from_browser, None, None, None)
        return opts

    def _auth_hint(self, msg: str) -> str:
        """Actionable troubleshooting text for known yt-dlp failure modes."""
        lowered = msg.lower()
        if "sign in to confirm" in lowered or ("bot" in lowered and "cookie" in lowered):
            return (
                " YouTube bot-check blocked this download. Fix options: "
                "1) set YTDLP_COOKIES_FROM_BROWSER=chrome in .env, "
                "2) close Chrome (so its cookie DB is unlocked) and re-run "
                "`python main.py --retry-failed`, or "
                "3) export cookies.txt (Netscape format) and set "
                "YTDLP_COOKIES_FILE=/path/to/cookies.txt, then re-run "
                "`python main.py --retry-failed`."
            )
        if "cookie" in lowered and (
            "database" in lowered
            or "cannot access" in lowered
            or "encrypted" in lowered
            or "locked" in lowered
        ):
            return (
                " Browser cookie database could not be read. Close the browser "
                "(yt-dlp needs the cookie DB unlocked) and retry; or export "
                "cookies.txt and set YTDLP_COOKIES_FILE."
            )
        if "private video" in lowered or "members only" in lowered or "age-restricted" in lowered:
            return (
                " Video is private / members-only / age-restricted and cannot be "
                "downloaded without authorized (signed-in) cookies."
            )
        return ""

    def download_audio(self, youtube_url_or_id: str) -> Dict[str, Any]:
        """
        Downloads audio for a given YouTube URL or Video ID.
        If the audio file already exists in cache, downloads are skipped.

        Returns metadata dict containing video_id, title, duration, url, and audio_path.

        Cookie resilience: when the configured cookie source is unreadable
        (e.g. Chrome's cookie DB is exclusively locked by a running Chrome,
        yt-dlp #7271) the download is retried once anonymously. The broken
        cookie source is remembered for the rest of the process lifetime so we
        never re-attempt the unreadable read per video.
        """
        video_id = extract_youtube_id(youtube_url_or_id)
        expected_audio_file = self.output_dir / f"{video_id}.{self.audio_format}"

        # Standard video URL
        full_url = f"https://www.youtube.com/watch?v={video_id}"

        # Check cache
        if expected_audio_file.exists() and expected_audio_file.stat().st_size > 0:
            print(f"[CACHE] Audio file found in cache: {expected_audio_file.name}")
            meta = self._fetch_metadata_only(full_url)
            meta["audio_path"] = str(expected_audio_file.resolve())
            return meta

        print(f"[DOWNLOAD] Downloading audio for Video ID: {video_id} ...")

        ffmpeg_bin = get_ffmpeg_path()
        attempts = (True, False) if self._cookies_ok else (False,)
        last_cookie_error: Optional[Exception] = None

        for use_cookies in attempts:
            # yt-dlp configuration options (+ optional cookie authentication)
            ydl_opts = self._build_ydl_opts(
                {
                    'format': 'm4a/bestaudio/best',
                    'outtmpl': str(self.output_dir / f"{video_id}.%(ext)s"),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': self.audio_format,
                        'preferredquality': '192',
                    }],
                },
                include_cookies=use_cookies,
            )
            if ffmpeg_bin:
                ydl_opts['ffmpeg_location'] = ffmpeg_bin

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(full_url, download=True)
                    title = info.get("title", "Unknown Title")
                    duration = info.get("duration", 0)

                actual_audio_file = self.output_dir / f"{video_id}.{self.audio_format}"
                if not actual_audio_file.exists():
                    for ext in ["m4a", "webm", "mp3", "opus"]:
                        candidate = self.output_dir / f"{video_id}.{ext}"
                        if candidate.exists():
                            actual_audio_file = candidate
                            break

                if not actual_audio_file.exists():
                    raise FileNotFoundError(
                        f"Downloaded audio file not found for video ID {video_id}"
                    )

                return {
                    "video_id": video_id,
                    "title": title,
                    "duration": duration,
                    "url": full_url,
                    "audio_path": str(actual_audio_file.resolve())
                }

            except yt_dlp.utils.DownloadError as e:
                if use_cookies and _is_cookie_unusable_error(str(e)):
                    self._cookies_ok = False
                    last_cookie_error = e
                    print(
                        "[AUTH] configured cookie source is unreadable "
                        "(yt-dlp #7271): retrying this video anonymously. "
                        "Close the browser (or export cookies.txt) and re-run "
                        "to use signed-in cookies."
                    )
                    continue
                hint = self._auth_hint(str(e))
                raise RuntimeError(
                    f"yt-dlp download error for video '{video_id}': {str(e)}{hint}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download audio for '{youtube_url_or_id}': {str(e)}"
                ) from e

        # Cookie attempt failed from a broken cookie source and the anonymous
        # fallback also failed: report the (typed) failure with remediation.
        hint = self._auth_hint(str(last_cookie_error or ""))
        raise RuntimeError(
            f"yt-dlp download error for video '{video_id}': "
            f"{last_cookie_error}{hint}"
        ) from last_cookie_error

    def _fetch_metadata_only(self, full_url: str) -> Dict[str, Any]:
        """Fetches video title and duration without downloading audio stream.

        Like `download_audio`, falls back to anonymous when the configured
        cookie source is unreadable, and always returns a minimal dict rather
        than raising (used for cache-hit metadata refresh).
        """
        ffmpeg_bin = get_ffmpeg_path()
        attempts = (True, False) if self._cookies_ok else (False,)

        for use_cookies in attempts:
            ydl_opts = self._build_ydl_opts(
                {'skip_download': True}, include_cookies=use_cookies
            )
            if ffmpeg_bin:
                ydl_opts['ffmpeg_location'] = ffmpeg_bin

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(full_url, download=False)
                    return {
                        "video_id": info.get("id"),
                        "title": info.get("title", "Unknown Title"),
                        "duration": info.get("duration", 0),
                        "url": full_url
                    }
            except yt_dlp.utils.DownloadError as e:
                if use_cookies and _is_cookie_unusable_error(str(e)):
                    self._cookies_ok = False
                    print(
                        "[AUTH] configured cookie source is unreadable "
                        "(yt-dlp #7271): falling back to anonymous metadata."
                    )
                    continue
            except Exception:
                pass

        return {
            "video_id": extract_youtube_id(full_url),
            "title": "Cached Lecture Video",
            "duration": 0,
            "url": full_url
        }

