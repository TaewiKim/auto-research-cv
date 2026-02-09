"""Advanced analysis modules for Parkinson's Disease symptoms."""

from .bradykinesia import (
    BradykinesiaAnalyzer,
    BradykinesiaFeatures,
    analyze_bradykinesia,
)
from .fog import (
    FOGDetector,
    FOGEpisode,
    FOGFeatures,
    FOGType,
    detect_fog,
)
from .tremor import (
    TremorDetector,
    TremorFeatures,
    TremorType,
    detect_tremor,
)
from .walking_detection import (
    WalkingDetector,
    WalkingDetectionResult,
    WalkingSegment,
    detect_walking,
)

__all__ = [
    # Walking Detection
    "WalkingDetector",
    "WalkingDetectionResult",
    "WalkingSegment",
    "detect_walking",
    # Tremor
    "TremorDetector",
    "TremorFeatures",
    "TremorType",
    "detect_tremor",
    # FOG
    "FOGDetector",
    "FOGEpisode",
    "FOGFeatures",
    "FOGType",
    "detect_fog",
    # Bradykinesia
    "BradykinesiaAnalyzer",
    "BradykinesiaFeatures",
    "analyze_bradykinesia",
]
