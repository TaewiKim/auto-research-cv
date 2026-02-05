"""Public dataset registry and download utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from research_automation.core.storage import StorageManager


class DatasetCategory(str, Enum):
    """Dataset categories."""

    GAIT = "gait"
    POSE = "pose"
    FACE = "face"
    SEIZURE = "seizure"
    STROKE = "stroke"
    GENERAL = "general"


class AccessType(str, Enum):
    """Dataset access type."""

    OPEN = "open"  # Freely available
    ACADEMIC = "academic"  # Requires academic agreement
    REQUEST = "request"  # Requires application
    RESTRICTED = "restricted"  # Limited access


@dataclass
class DatasetInfo:
    """Information about a public dataset."""

    name: str
    description: str
    category: DatasetCategory
    access_type: AccessType
    size: str  # Human-readable size
    subjects: int | None
    url: str
    paper_doi: str | None = None
    download_url: str | None = None  # Direct download if available
    huggingface_id: str | None = None  # HuggingFace dataset ID
    notes: str | None = None
    tags: list[str] = field(default_factory=list)


# Registry of known datasets for video-based health monitoring
DATASET_REGISTRY: dict[str, DatasetInfo] = {
    # Gait / Parkinson's
    "care-pd": DatasetInfo(
        name="CARE-PD",
        description="Largest PD gait dataset with 3D mesh data from 363 participants across 8 clinical sites",
        category=DatasetCategory.GAIT,
        access_type=AccessType.OPEN,
        size="~50GB",
        subjects=363,
        url="https://github.com/TaatiTeam/CARE-PD",
        paper_doi="10.48550/arXiv.2311.09890",
        huggingface_id="vida-adl/CARE-PD",
        tags=["parkinson", "gait", "3d-mesh", "clinical"],
    ),
    "fasteval-pd": DatasetInfo(
        name="FastEval PD",
        description="840 finger-tapping videos from 186 participants for PD severity assessment",
        category=DatasetCategory.GAIT,
        access_type=AccessType.REQUEST,
        size="~10GB",
        subjects=186,
        url="https://github.com/some-repo/fasteval-pd",  # Placeholder
        notes="Requires academic data use agreement",
        tags=["parkinson", "finger-tapping", "severity"],
    ),
    "gavd": DatasetInfo(
        name="GAVD",
        description="Gait Abnormality Video Dataset with 1,874 sequences across 11 abnormality categories",
        category=DatasetCategory.GAIT,
        access_type=AccessType.ACADEMIC,
        size="~30GB",
        subjects=None,
        url="https://example.com/gavd",  # Placeholder
        tags=["gait", "abnormality", "multi-class"],
    ),
    # Pose
    "coco-wholebody": DatasetInfo(
        name="COCO-WholeBody",
        description="Extension of COCO with whole-body keypoint annotations",
        category=DatasetCategory.POSE,
        access_type=AccessType.OPEN,
        size="~25GB",
        subjects=None,
        url="https://github.com/jin-s13/COCO-WholeBody",
        download_url="https://cocodataset.org/#download",
        tags=["pose", "keypoints", "whole-body"],
    ),
    "h36m": DatasetInfo(
        name="Human3.6M",
        description="Large-scale 3D human pose dataset with 3.6M frames",
        category=DatasetCategory.POSE,
        access_type=AccessType.ACADEMIC,
        size="~60GB",
        subjects=11,
        url="http://vision.imar.ro/human3.6m/",
        tags=["pose", "3d", "indoor"],
    ),
    # Face / AU
    "bp4d": DatasetInfo(
        name="BP4D",
        description="3D spontaneous facial expression database with AU labels from 41 subjects",
        category=DatasetCategory.FACE,
        access_type=AccessType.ACADEMIC,
        size="~100GB",
        subjects=41,
        url="http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html",
        tags=["face", "action-units", "3d", "expression"],
    ),
    "ck-plus": DatasetInfo(
        name="CK+",
        description="Extended Cohn-Kanade dataset with 593 emotion sequences",
        category=DatasetCategory.FACE,
        access_type=AccessType.ACADEMIC,
        size="~2GB",
        subjects=123,
        url="https://www.jeffcohn.net/Resources/",
        tags=["face", "emotion", "expression"],
    ),
    "disfa": DatasetInfo(
        name="DISFA",
        description="Denver Intensity of Spontaneous Facial Action database",
        category=DatasetCategory.FACE,
        access_type=AccessType.ACADEMIC,
        size="~10GB",
        subjects=27,
        url="http://mohammadmahoor.com/disfa/",
        tags=["face", "action-units", "intensity"],
    ),
    # Seizure
    "tuh-eeg": DatasetInfo(
        name="TUH EEG Seizure Corpus",
        description="Large EEG dataset for seizure detection (for multimodal integration)",
        category=DatasetCategory.SEIZURE,
        access_type=AccessType.ACADEMIC,
        size="~300GB",
        subjects=None,
        url="https://isip.piconepress.com/projects/tuh_eeg/",
        tags=["eeg", "seizure", "clinical"],
    ),
    # Stroke / Facial Palsy
    "youtube-facial-palsy": DatasetInfo(
        name="YouTube Facial Palsy Database",
        description="22 patients, 32 videos of facial palsy from YouTube",
        category=DatasetCategory.STROKE,
        access_type=AccessType.OPEN,
        size="~500MB",
        subjects=22,
        url="https://github.com/AvLab-CV/YouTube-Facial-Palsy-Database",
        tags=["face", "palsy", "stroke", "clinical"],
    ),
}


class DatasetManager:
    """Manage dataset downloads and metadata."""

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.storage = storage or StorageManager()
        self.registry = DATASET_REGISTRY

    def list_datasets(
        self,
        category: DatasetCategory | None = None,
        access_type: AccessType | None = None,
        tag: str | None = None,
    ) -> list[DatasetInfo]:
        """List available datasets with optional filtering."""
        results = list(self.registry.values())

        if category:
            results = [d for d in results if d.category == category]

        if access_type:
            results = [d for d in results if d.access_type == access_type]

        if tag:
            results = [d for d in results if tag in d.tags]

        return results

    def get_dataset(self, name: str) -> DatasetInfo | None:
        """Get dataset info by name."""
        return self.registry.get(name.lower())

    def download_huggingface(self, dataset_id: str, subset: str | None = None) -> Path:
        """Download dataset from HuggingFace."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")

        output_dir = self.storage.datasets_dir / dataset_id.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(output_dir),
        )

        return output_dir

    def download_care_pd(self) -> Path:
        """Download CARE-PD dataset from HuggingFace."""
        return self.download_huggingface("vida-adl/CARE-PD")

    def get_local_path(self, name: str) -> Path | None:
        """Get local path for downloaded dataset."""
        dataset = self.get_dataset(name)
        if not dataset:
            return None

        # Check standard locations
        candidates = [
            self.storage.datasets_dir / name,
            self.storage.datasets_dir / name.replace("-", "_"),
        ]

        if dataset.huggingface_id:
            candidates.append(
                self.storage.datasets_dir / dataset.huggingface_id.replace("/", "_")
            )

        for path in candidates:
            if path.exists():
                return path

        return None

    def is_downloaded(self, name: str) -> bool:
        """Check if dataset is already downloaded."""
        return self.get_local_path(name) is not None


def list_datasets(
    category: str | None = None,
    access_type: str | None = None,
) -> list[DatasetInfo]:
    """Convenience function to list datasets."""
    manager = DatasetManager()
    return manager.list_datasets(
        category=DatasetCategory(category) if category else None,
        access_type=AccessType(access_type) if access_type else None,
    )


def format_dataset_list(datasets: list[DatasetInfo]) -> str:
    """Format dataset list as readable text."""
    lines = ["# Available Datasets\n"]

    for dataset in sorted(datasets, key=lambda d: d.name):
        lines.append(f"## {dataset.name}")
        lines.append(f"- Category: {dataset.category.value}")
        lines.append(f"- Access: {dataset.access_type.value}")
        lines.append(f"- Size: {dataset.size}")
        if dataset.subjects:
            lines.append(f"- Subjects: {dataset.subjects}")
        lines.append(f"- URL: {dataset.url}")
        lines.append(f"- Tags: {', '.join(dataset.tags)}")
        lines.append(f"\n{dataset.description}\n")

    return "\n".join(lines)
