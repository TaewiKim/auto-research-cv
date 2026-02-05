"""YouTube video search and download using yt-dlp."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from research_automation.core.config import get_settings
from research_automation.core.storage import StorageManager


@dataclass
class VideoInfo:
    """YouTube video metadata."""

    video_id: str
    title: str
    description: str | None
    duration: int  # seconds
    upload_date: datetime | None
    channel: str | None
    view_count: int | None
    url: str
    thumbnail_url: str | None


@dataclass
class DownloadResult:
    """Result of video download."""

    video_id: str
    path: Path
    info: VideoInfo
    success: bool
    error: str | None = None


class YouTubeCollector:
    """Search and download YouTube videos."""

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.settings = get_settings()
        self.storage = storage or StorageManager()

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_duration: int | None = None,
    ) -> list[VideoInfo]:
        """Search YouTube for videos."""
        import yt_dlp

        if max_duration is None:
            max_duration = self.settings.youtube.max_duration

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "default_search": "ytsearch",
        }

        search_query = f"ytsearch{max_results * 2}:{query}"  # Get extra to filter

        results: list[VideoInfo] = []

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get("entries", []) if info else []

                for entry in entries:
                    if not entry:
                        continue

                    # Get full video info for duration
                    video_url = entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}"

                    try:
                        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl2:
                            full_info = ydl2.extract_info(video_url, download=False)
                    except Exception:
                        continue

                    if not full_info:
                        continue

                    duration = full_info.get("duration", 0) or 0

                    # Filter by duration
                    if duration > max_duration:
                        continue

                    upload_date = None
                    date_str = full_info.get("upload_date")
                    if date_str:
                        try:
                            upload_date = datetime.strptime(date_str, "%Y%m%d")
                        except ValueError:
                            pass

                    video_info = VideoInfo(
                        video_id=full_info.get("id", ""),
                        title=full_info.get("title", ""),
                        description=full_info.get("description"),
                        duration=duration,
                        upload_date=upload_date,
                        channel=full_info.get("channel") or full_info.get("uploader"),
                        view_count=full_info.get("view_count"),
                        url=full_info.get("webpage_url", video_url),
                        thumbnail_url=full_info.get("thumbnail"),
                    )

                    results.append(video_info)

                    if len(results) >= max_results:
                        break

            except Exception as e:
                print(f"Search error: {e}")

        return results

    def download(
        self,
        video_id: str,
        output_dir: Path | None = None,
    ) -> DownloadResult:
        """Download a YouTube video."""
        import yt_dlp

        if output_dir is None:
            output_dir = self.storage.raw_videos_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://www.youtube.com/watch?v={video_id}"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": self._get_format_string(),
            "outtmpl": str(output_dir / self.settings.youtube.output_template),
            "writeinfojson": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if not info:
                    return DownloadResult(
                        video_id=video_id,
                        path=Path(),
                        info=VideoInfo(
                            video_id=video_id,
                            title="",
                            description=None,
                            duration=0,
                            upload_date=None,
                            channel=None,
                            view_count=None,
                            url=url,
                            thumbnail_url=None,
                        ),
                        success=False,
                        error="Failed to extract video info",
                    )

                # Find downloaded file
                ext = info.get("ext", "mp4")
                output_path = output_dir / f"{video_id}.{ext}"

                upload_date = None
                date_str = info.get("upload_date")
                if date_str:
                    try:
                        upload_date = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        pass

                video_info = VideoInfo(
                    video_id=info.get("id", video_id),
                    title=info.get("title", ""),
                    description=info.get("description"),
                    duration=info.get("duration", 0) or 0,
                    upload_date=upload_date,
                    channel=info.get("channel") or info.get("uploader"),
                    view_count=info.get("view_count"),
                    url=info.get("webpage_url", url),
                    thumbnail_url=info.get("thumbnail"),
                )

                return DownloadResult(
                    video_id=video_id,
                    path=output_path,
                    info=video_info,
                    success=True,
                )

        except Exception as e:
            return DownloadResult(
                video_id=video_id,
                path=Path(),
                info=VideoInfo(
                    video_id=video_id,
                    title="",
                    description=None,
                    duration=0,
                    upload_date=None,
                    channel=None,
                    view_count=None,
                    url=url,
                    thumbnail_url=None,
                ),
                success=False,
                error=str(e),
            )

    def download_from_url(self, url: str) -> DownloadResult:
        """Download video from URL."""
        import yt_dlp

        # Extract video ID from URL
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id", "") if info else ""

        if not video_id:
            return DownloadResult(
                video_id="",
                path=Path(),
                info=VideoInfo(
                    video_id="",
                    title="",
                    description=None,
                    duration=0,
                    upload_date=None,
                    channel=None,
                    view_count=None,
                    url=url,
                    thumbnail_url=None,
                ),
                success=False,
                error="Could not extract video ID from URL",
            )

        return self.download(video_id)

    def _get_format_string(self) -> str:
        """Get yt-dlp format string based on preferred quality."""
        quality = self.settings.youtube.preferred_quality

        quality_map = {
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "best": "bestvideo+bestaudio/best",
        }

        return quality_map.get(quality, quality_map["720p"])

    def get_video_info(self, video_id: str) -> VideoInfo | None:
        """Get video info without downloading."""
        import yt_dlp

        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)

                if not info:
                    return None

                upload_date = None
                date_str = info.get("upload_date")
                if date_str:
                    try:
                        upload_date = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        pass

                return VideoInfo(
                    video_id=info.get("id", video_id),
                    title=info.get("title", ""),
                    description=info.get("description"),
                    duration=info.get("duration", 0) or 0,
                    upload_date=upload_date,
                    channel=info.get("channel") or info.get("uploader"),
                    view_count=info.get("view_count"),
                    url=info.get("webpage_url", url),
                    thumbnail_url=info.get("thumbnail"),
                )
        except Exception:
            return None


def search_youtube(query: str, max_results: int = 10) -> list[VideoInfo]:
    """Convenience function for YouTube search."""
    collector = YouTubeCollector()
    return collector.search(query, max_results)


def download_youtube(video_id: str) -> DownloadResult:
    """Convenience function for YouTube download."""
    collector = YouTubeCollector()
    return collector.download(video_id)
