import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def extract_youtube_id(url_or_id: str) -> str:
    """
    Extracts 11-character YouTube video ID from standard URLs, shortened links,
    or returns raw ID if already provided.
    
    Supported formats:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ
    - https://www.youtube.com/embed/dQw4w9WgXcQ
    - dQw4w9WgXcQ
    """
    url_or_id = url_or_id.strip()
    
    # Direct Video ID pattern (11 characters)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    # Regular expressions for YouTube URLs
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a valid YouTube Video ID from input: '{url_or_id}'")


def format_seconds_to_timestamp(seconds: float) -> str:
    """
    Converts float seconds (e.g. 125.4) into readable HH:MM:SS format (e.g. '00:02:05').
    """
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_playlist_id(url: str) -> str | None:
    """
    Extracts the playlist ID / list parameter from a YouTube playlist URL.

    Forms supported:
    - https://www.youtube.com/playlist?list=PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz
    - https://www.youtube.com/watch?v=...&list=PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz
    - https://youtube.com/playlist/PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz
    """
    if not url:
        return None

    match = re.search(r"[?&]list=([0-9A-Za-z_-]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"(?:playlist|list)[/=]([0-9A-Za-z_-]+)", url)
    if match:
        return match.group(1)

    return None


def generate_youtube_timestamp_url(video_id: str, seconds: float) -> str:
    """
    Generates clickable YouTube video URL at exact starting timestamp in seconds.
    """
    timestamp_seconds = int(seconds)
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds}s"


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Saves dictionary data to a formatted JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    """Loads JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
