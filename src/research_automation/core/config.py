"""Configuration management with dataclasses and YAML loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = "sqlite:///data/research.db"
    echo: bool = False


@dataclass
class StorageConfig:
    """Storage paths configuration."""

    base_dir: Path = field(default_factory=lambda: Path("data"))
    papers_dir: Path = field(default_factory=lambda: Path("data/papers"))
    videos_dir: Path = field(default_factory=lambda: Path("data/videos"))
    raw_videos_dir: Path = field(default_factory=lambda: Path("data/videos/raw"))
    processed_videos_dir: Path = field(default_factory=lambda: Path("data/videos/processed"))
    datasets_dir: Path = field(default_factory=lambda: Path("data/datasets"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        for attr_name in [
            "base_dir",
            "papers_dir",
            "videos_dir",
            "raw_videos_dir",
            "processed_videos_dir",
            "datasets_dir",
            "cache_dir",
        ]:
            val = getattr(self, attr_name)
            if isinstance(val, str):
                setattr(self, attr_name, Path(val))


@dataclass
class ClaudeConfig:
    """Claude API configuration."""

    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3

    def __post_init__(self) -> None:
        """Load API key from environment if not set."""
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class YouTubeConfig:
    """YouTube download configuration."""

    max_duration: int = 600  # seconds
    preferred_quality: str = "720p"
    output_template: str = "%(id)s.%(ext)s"


@dataclass
class SearchConfig:
    """Literature search configuration."""

    default_limit: int = 20
    pubmed_email: str = ""
    semantic_scholar_api_key: str = ""

    def __post_init__(self) -> None:
        """Load from environment."""
        if not self.pubmed_email:
            self.pubmed_email = os.getenv("PUBMED_EMAIL", "")
        if not self.semantic_scholar_api_key:
            self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


@dataclass
class Settings:
    """Main application settings."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """Load settings from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Settings:
        """Create Settings from dictionary."""
        return cls(
            database=DatabaseConfig(**data.get("database", {})),
            storage=StorageConfig(**data.get("storage", {})),
            claude=ClaudeConfig(**data.get("claude", {})),
            youtube=YouTubeConfig(**data.get("youtube", {})),
            search=SearchConfig(**data.get("search", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        from dataclasses import asdict

        result = asdict(self)
        # Convert Path objects to strings
        for key in result.get("storage", {}):
            val = result["storage"][key]
            if isinstance(val, Path):
                result["storage"][key] = str(val)
        return result

    def ensure_directories(self) -> None:
        """Create all storage directories if they don't exist."""
        for attr_name in [
            "base_dir",
            "papers_dir",
            "videos_dir",
            "raw_videos_dir",
            "processed_videos_dir",
            "datasets_dir",
            "cache_dir",
        ]:
            path = getattr(self.storage, attr_name)
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None) -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        if config_path is None:
            # Default config paths to check
            for candidate in [
                Path("config/settings.yaml"),
                Path.home() / ".config/research-automation/settings.yaml",
            ]:
                if candidate.exists():
                    config_path = candidate
                    break

        _settings = Settings.from_yaml(config_path) if config_path else Settings()

    return _settings


def reload_settings(config_path: str | Path | None = None) -> Settings:
    """Force reload settings from file."""
    global _settings
    _settings = None
    return get_settings(config_path)
