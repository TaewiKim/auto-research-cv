"""Compatibility wrapper for tremor analysis."""

from .behavior_analysis.tremor import (
    TremorDetector,
    TremorFeatures,
    TremorType,
    detect_tremor,
)

__all__ = [
    "TremorDetector",
    "TremorFeatures",
    "TremorType",
    "detect_tremor",
]
