"""Compatibility wrapper for walking detection analysis."""

from .behavior_analysis.walking_detection import (
    WalkingDetectionResult,
    WalkingDetector,
    WalkingSegment,
    detect_walking,
)

__all__ = [
    "WalkingDetectionResult",
    "WalkingDetector",
    "WalkingSegment",
    "detect_walking",
]
