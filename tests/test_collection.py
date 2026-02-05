"""Tests for collection module."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_video_info_creation(self, sample_video_info):
        """Test VideoInfo creation."""
        assert sample_video_info.video_id == "dQw4w9WgXcQ"
        assert sample_video_info.duration == 180
        assert sample_video_info.view_count == 1000


class TestYouTubeCollector:
    """Tests for YouTubeCollector."""

    def test_collector_init(self, storage_manager):
        """Test YouTubeCollector initialization."""
        from research_automation.collection.youtube import YouTubeCollector

        collector = YouTubeCollector(storage_manager)
        assert collector.storage == storage_manager

    def test_format_string(self, storage_manager):
        """Test format string generation."""
        from research_automation.collection.youtube import YouTubeCollector

        collector = YouTubeCollector(storage_manager)
        fmt = collector._get_format_string()
        assert "bestvideo" in fmt or "best" in fmt


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_quality_metrics_creation(self):
        """Test QualityMetrics creation."""
        from research_automation.collection.quality import QualityMetrics

        metrics = QualityMetrics(
            duration=60.0,
            fps=30.0,
            resolution=(1920, 1080),
            frame_count=1800,
            face_detection_rate=0.8,
            pose_detection_rate=0.7,
            brightness_score=0.6,
            blur_score=0.75,
            stability_score=0.9,
        )

        assert metrics.duration == 60.0
        assert metrics.resolution == (1920, 1080)

    def test_overall_score(self):
        """Test overall score calculation."""
        from research_automation.collection.quality import QualityMetrics

        metrics = QualityMetrics(
            duration=60.0,
            fps=30.0,
            resolution=(1920, 1080),
            frame_count=1800,
            face_detection_rate=1.0,
            pose_detection_rate=1.0,
            brightness_score=1.0,
            blur_score=1.0,
            stability_score=1.0,
        )

        assert abs(metrics.overall_score - 1.0) < 1e-10

    def test_is_usable(self):
        """Test is_usable property."""
        from research_automation.collection.quality import QualityMetrics

        # Good quality video
        good = QualityMetrics(
            duration=60.0,
            fps=30.0,
            resolution=(1920, 1080),
            frame_count=1800,
            face_detection_rate=0.8,
            pose_detection_rate=0.7,
            brightness_score=0.6,
            blur_score=0.75,
            stability_score=0.9,
        )
        assert good.is_usable

        # Poor quality video
        poor = QualityMetrics(
            duration=60.0,
            fps=30.0,
            resolution=(320, 240),
            frame_count=1800,
            face_detection_rate=0.1,
            pose_detection_rate=0.1,
            brightness_score=0.2,
            blur_score=0.1,
            stability_score=0.1,
        )
        assert not poor.is_usable


class TestDatasetRegistry:
    """Tests for dataset registry."""

    def test_list_datasets(self):
        """Test listing datasets."""
        from research_automation.collection.datasets import list_datasets

        datasets = list_datasets()
        assert len(datasets) > 0

    def test_list_datasets_by_category(self):
        """Test filtering datasets by category."""
        from research_automation.collection.datasets import DatasetCategory, list_datasets

        gait_datasets = list_datasets(category="gait")
        assert all(d.category == DatasetCategory.GAIT for d in gait_datasets)

    def test_dataset_manager(self, storage_manager):
        """Test DatasetManager."""
        from research_automation.collection.datasets import DatasetManager

        manager = DatasetManager(storage_manager)

        # Test get_dataset
        care_pd = manager.get_dataset("care-pd")
        assert care_pd is not None
        assert care_pd.name == "CARE-PD"

        # Test non-existent dataset
        not_found = manager.get_dataset("nonexistent")
        assert not_found is None


class TestQuestionnaire:
    """Tests for clinical questionnaire module."""

    def test_get_scale(self):
        """Test getting clinical scales."""
        from research_automation.collection.questionnaire import get_scale

        # By name
        updrs = get_scale("mds-updrs-iii")
        assert updrs is not None
        assert updrs.abbreviation == "MDS-UPDRS-III"

        # By abbreviation
        hy = get_scale("H&Y")
        assert hy is not None
        assert "Hoehn" in hy.name

    def test_list_scales(self):
        """Test listing all scales."""
        from research_automation.collection.questionnaire import list_scales

        scales = list_scales()
        assert len(scales) >= 3  # We have at least 3 defined

    def test_scale_items(self):
        """Test scale items."""
        from research_automation.collection.questionnaire import get_scale

        updrs = get_scale("mds-updrs-iii")
        assert len(updrs.items) > 0

        # Check first item
        item = updrs.items[0]
        assert item.number == "3.1"
        assert item.name == "Speech"
        assert item.min_score == 0
        assert item.max_score == 4

    def test_format_scale(self):
        """Test scale formatting."""
        from research_automation.collection.questionnaire import format_scale, get_scale

        hy = get_scale("H&Y")
        formatted = format_scale(hy)

        assert "Hoehn and Yahr" in formatted
        assert "Stage" in formatted


class TestECG:
    """Tests for ECG module."""

    def test_ecg_signal_dataclass(self):
        """Test ECGSignal dataclass."""
        import numpy as np

        from research_automation.collection.ecg import ECGSignal

        signal = ECGSignal(
            data=np.random.randn(1000, 1),
            sampling_rate=500.0,
            duration=2.0,
            leads=["Lead I"],
            metadata={"source": "test"},
        )

        assert signal.sampling_rate == 500.0
        assert signal.duration == 2.0
        assert len(signal.leads) == 1

    def test_ecg_features_dataclass(self):
        """Test ECGFeatures dataclass."""
        import numpy as np

        from research_automation.collection.ecg import ECGFeatures

        features = ECGFeatures(
            heart_rate=72.0,
            heart_rate_variability={"rmssd": 50.0, "sdnn": 100.0},
            r_peaks=np.array([100, 200, 300]),
            rr_intervals=np.array([800, 850]),
            quality_score=0.9,
        )

        assert features.heart_rate == 72.0
        assert features.quality_score == 0.9


class TestStorageManager:
    """Tests for StorageManager."""

    def test_storage_init(self, test_settings):
        """Test StorageManager initialization."""
        from research_automation.core.storage import StorageManager

        manager = StorageManager(test_settings)

        assert manager.papers_dir.exists()
        assert manager.raw_videos_dir.exists()

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        from research_automation.core.storage import StorageManager

        assert StorageManager._sanitize_filename("test/file") == "test_file"
        assert StorageManager._sanitize_filename("test:file") == "test_file"
        assert StorageManager._sanitize_filename("normal") == "normal"

    def test_get_paper_path(self, storage_manager):
        """Test paper path generation."""
        path = storage_manager.get_paper_path("10.1234/test")
        assert path.suffix == ".pdf"
        assert "10.1234_test" in str(path)

    def test_save_paper(self, storage_manager):
        """Test saving paper content."""
        content = b"%PDF-1.4 test content"
        path = storage_manager.save_paper(content, "test_paper")

        assert path.exists()
        assert path.read_bytes() == content
