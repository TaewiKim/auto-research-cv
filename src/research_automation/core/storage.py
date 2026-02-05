"""Storage manager for files and artifacts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


class StorageManager:
    """Manages file storage for papers, videos, and datasets."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize storage manager."""
        if settings is None:
            from .config import get_settings

            settings = get_settings()
        self.settings = settings
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure all storage directories exist."""
        self.settings.ensure_directories()

    @property
    def papers_dir(self) -> Path:
        """Get papers directory."""
        return self.settings.storage.papers_dir

    @property
    def raw_videos_dir(self) -> Path:
        """Get raw videos directory."""
        return self.settings.storage.raw_videos_dir

    @property
    def processed_videos_dir(self) -> Path:
        """Get processed videos directory."""
        return self.settings.storage.processed_videos_dir

    @property
    def datasets_dir(self) -> Path:
        """Get datasets directory."""
        return self.settings.storage.datasets_dir

    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        return self.settings.storage.cache_dir

    def get_paper_path(self, identifier: str, ext: str = "pdf") -> Path:
        """Get path for a paper file."""
        safe_id = self._sanitize_filename(identifier)
        return self.papers_dir / f"{safe_id}.{ext}"

    def get_video_path(self, video_id: str, processed: bool = False) -> Path:
        """Get path for a video file."""
        safe_id = self._sanitize_filename(video_id)
        base_dir = self.processed_videos_dir if processed else self.raw_videos_dir
        # Extension will be determined by downloader
        return base_dir / safe_id

    def get_dataset_path(self, name: str) -> Path:
        """Get path for a dataset directory."""
        safe_name = self._sanitize_filename(name)
        return self.datasets_dir / safe_name

    def get_cache_path(self, key: str, ext: str = "json") -> Path:
        """Get path for a cache file."""
        # Use hash for cache keys to handle long or complex keys
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hash_key}.{ext}"

    def save_paper(self, content: bytes, identifier: str) -> Path:
        """Save paper PDF content."""
        path = self.get_paper_path(identifier)
        path.write_bytes(content)
        return path

    def save_text(self, text: str, identifier: str, subdir: str = "") -> Path:
        """Save text content to a file."""
        if subdir:
            target_dir = self.settings.storage.base_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.settings.storage.base_dir

        safe_id = self._sanitize_filename(identifier)
        path = target_dir / f"{safe_id}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def file_exists(self, path: Path) -> bool:
        """Check if a file exists."""
        return path.exists() and path.is_file()

    def delete_file(self, path: Path) -> bool:
        """Delete a file if it exists."""
        if self.file_exists(path):
            path.unlink()
            return True
        return False

    def delete_directory(self, path: Path) -> bool:
        """Delete a directory and its contents."""
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
        return False

    def list_papers(self) -> list[Path]:
        """List all paper files."""
        return sorted(self.papers_dir.glob("*.pdf"))

    def list_videos(self, processed: bool = False) -> list[Path]:
        """List all video files."""
        base_dir = self.processed_videos_dir if processed else self.raw_videos_dir
        extensions = ["*.mp4", "*.webm", "*.mkv", "*.avi"]
        files = []
        for ext in extensions:
            files.extend(base_dir.glob(ext))
        return sorted(files)

    def list_datasets(self) -> list[Path]:
        """List all dataset directories."""
        return sorted([d for d in self.datasets_dir.iterdir() if d.is_dir()])

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        if self.file_exists(path):
            return path.stat().st_size
        return 0

    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        if not path.exists():
            return 0
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a string for use as filename."""
        # Replace problematic characters
        replacements = {
            "/": "_",
            "\\": "_",
            ":": "_",
            "*": "_",
            "?": "_",
            '"': "_",
            "<": "_",
            ">": "_",
            "|": "_",
            " ": "_",
        }
        result = name
        for old, new in replacements.items():
            result = result.replace(old, new)
        # Limit length
        if len(result) > 200:
            result = result[:200]
        return result
