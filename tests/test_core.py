"""Tests for core module."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self):
        """Test default settings creation."""
        from research_automation.core.config import Settings

        settings = Settings()

        assert settings.database.url == "sqlite:///data/research.db"
        assert settings.claude.model == "claude-sonnet-4-20250514"
        assert settings.youtube.max_duration == 600

    def test_settings_from_dict(self):
        """Test settings from dictionary."""
        from research_automation.core.config import Settings

        data = {
            "database": {"url": "sqlite:///custom.db"},
            "claude": {"model": "claude-opus-4-20250514"},
        }

        settings = Settings._from_dict(data)

        assert settings.database.url == "sqlite:///custom.db"
        assert settings.claude.model == "claude-opus-4-20250514"

    def test_settings_to_dict(self):
        """Test settings to dictionary conversion."""
        from research_automation.core.config import Settings

        settings = Settings()
        data = settings.to_dict()

        assert "database" in data
        assert "storage" in data
        assert "claude" in data

    def test_ensure_directories(self, temp_dir):
        """Test directory creation."""
        from research_automation.core.config import Settings, StorageConfig

        settings = Settings(
            storage=StorageConfig(
                base_dir=temp_dir / "data",
                papers_dir=temp_dir / "data/papers",
                videos_dir=temp_dir / "data/videos",
                raw_videos_dir=temp_dir / "data/videos/raw",
                processed_videos_dir=temp_dir / "data/videos/processed",
                datasets_dir=temp_dir / "data/datasets",
                cache_dir=temp_dir / "data/cache",
            )
        )

        settings.ensure_directories()

        assert settings.storage.papers_dir.exists()
        assert settings.storage.raw_videos_dir.exists()


class TestDatabase:
    """Tests for database module."""

    def test_base_class(self):
        """Test Base class exists."""
        from research_automation.core.database import Base

        assert Base is not None

    def test_get_session_context(self, test_db):
        """Test session context manager."""
        # The test_db fixture creates sessions directly
        session = test_db()
        assert session is not None
        session.close()


class TestStorageManager:
    """Tests for StorageManager."""

    def test_list_papers_empty(self, storage_manager):
        """Test listing papers in empty directory."""
        papers = storage_manager.list_papers()
        assert papers == []

    def test_list_videos_empty(self, storage_manager):
        """Test listing videos in empty directory."""
        videos = storage_manager.list_videos()
        assert videos == []

    def test_file_operations(self, storage_manager):
        """Test file existence and deletion."""
        # Create a test file
        test_path = storage_manager.papers_dir / "test.txt"
        test_path.write_text("test")

        assert storage_manager.file_exists(test_path)
        assert storage_manager.get_file_size(test_path) > 0

        # Delete
        assert storage_manager.delete_file(test_path)
        assert not storage_manager.file_exists(test_path)

    def test_cache_path(self, storage_manager):
        """Test cache path generation."""
        path1 = storage_manager.get_cache_path("test_key")
        path2 = storage_manager.get_cache_path("test_key")
        path3 = storage_manager.get_cache_path("different_key")

        # Same key should give same path
        assert path1 == path2
        # Different key should give different path
        assert path1 != path3


class TestClaudeClient:
    """Tests for Claude client (without actual API calls)."""

    def test_client_init(self):
        """Test client initialization."""
        from research_automation.core.claude import ClaudeClient
        from research_automation.core.config import ClaudeConfig

        config = ClaudeConfig(api_key="test_key")
        client = ClaudeClient(config)

        assert client.config.api_key == "test_key"
        assert client.config.model == "claude-sonnet-4-20250514"

    def test_client_missing_key(self):
        """Test error on missing API key."""
        from research_automation.core.claude import ClaudeClient
        from research_automation.core.config import ClaudeConfig

        config = ClaudeConfig(api_key="")
        client = ClaudeClient(config)

        with pytest.raises(ValueError, match="API key not set"):
            _ = client.client
