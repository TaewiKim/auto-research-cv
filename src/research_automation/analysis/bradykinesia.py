"""Compatibility wrapper for bradykinesia analysis."""

from .behavior_analysis.bradykinesia import (
    BradykinesiaAnalyzer,
    BradykinesiaFeatures,
    analyze_bradykinesia,
)

__all__ = [
    "BradykinesiaAnalyzer",
    "BradykinesiaFeatures",
    "analyze_bradykinesia",
]
