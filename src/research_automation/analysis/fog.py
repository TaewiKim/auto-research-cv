"""Compatibility wrapper for FOG risk analysis."""

from .risk_analysis.fog import FOGDetector, FOGEpisode, FOGFeatures, FOGType, detect_fog

__all__ = [
    "FOGDetector",
    "FOGEpisode",
    "FOGFeatures",
    "FOGType",
    "detect_fog",
]
