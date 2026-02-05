"""Pytest fixtures for research-automation tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research_automation.core.config import Settings, StorageConfig
from research_automation.core.database import Base, reset_engine
from research_automation.core.storage import StorageManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary directories."""
    return Settings(
        storage=StorageConfig(
            base_dir=temp_dir,
            papers_dir=temp_dir / "papers",
            videos_dir=temp_dir / "videos",
            raw_videos_dir=temp_dir / "videos/raw",
            processed_videos_dir=temp_dir / "videos/processed",
            datasets_dir=temp_dir / "datasets",
            cache_dir=temp_dir / "cache",
        ),
    )


@pytest.fixture
def storage_manager(test_settings: Settings) -> StorageManager:
    """Create storage manager with test settings."""
    return StorageManager(test_settings)


@pytest.fixture
def test_db(temp_dir: Path):
    """Create a test database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = temp_dir / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)

    yield Session

    # Cleanup
    reset_engine()


@pytest.fixture
def sample_search_result():
    """Create a sample search result."""
    from datetime import datetime

    from research_automation.literature.search import SearchResult

    return SearchResult(
        title="Video-based Parkinson's Disease Assessment Using Deep Learning",
        authors=["Smith, John", "Doe, Jane"],
        abstract="This paper presents a novel approach to automated PD assessment...",
        doi="10.1234/example.2024",
        pmid="12345678",
        arxiv_id=None,
        url="https://example.com/paper",
        source="pubmed",
        journal="Nature Medicine",
        publication_date=datetime(2024, 1, 15),
        citation_count=42,
    )


@pytest.fixture
def sample_video_info():
    """Create sample video info."""
    from datetime import datetime

    from research_automation.collection.youtube import VideoInfo

    return VideoInfo(
        video_id="dQw4w9WgXcQ",
        title="Test Video for Research",
        description="A test video description",
        duration=180,
        upload_date=datetime(2024, 1, 1),
        channel="Test Channel",
        view_count=1000,
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        thumbnail_url="https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg",
    )
