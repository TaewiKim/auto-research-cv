"""Analysis modules organized by stages.

Execution order:
1) face recognition
2) behavior analysis per person
3) risk analysis per action
"""

from .behavior_analysis import (
    BehaviorAnalyzer,
    BradykinesiaAnalyzer,
    BradykinesiaFeatures,
    PersonBehaviorAnalysis,
    TremorDetector,
    TremorFeatures,
    TremorType,
    WalkingDetectionResult,
    WalkingDetector,
    WalkingSegment,
    analyze_bradykinesia,
    detect_tremor,
    detect_walking,
)
from .face_analysis import FaceIdentity, FaceRecognitionAnalyzer, FaceRecognitionResult
from .risk_analysis import (
    ActionRisk,
    ActionRiskAnalyzer,
    FOGDetector,
    FOGEpisode,
    FOGFeatures,
    FOGType,
    PersonRiskAnalysis,
    detect_fog,
)
from .video_pipeline import OrderedPipelineResult, OrderedVideoAnalysisPipeline, get_ordered_video_pipeline

__all__ = [
    "FaceIdentity",
    "FaceRecognitionAnalyzer",
    "FaceRecognitionResult",
    "BehaviorAnalyzer",
    "PersonBehaviorAnalysis",
    "WalkingDetector",
    "WalkingDetectionResult",
    "WalkingSegment",
    "detect_walking",
    "TremorDetector",
    "TremorFeatures",
    "TremorType",
    "detect_tremor",
    "BradykinesiaAnalyzer",
    "BradykinesiaFeatures",
    "analyze_bradykinesia",
    "ActionRisk",
    "ActionRiskAnalyzer",
    "PersonRiskAnalysis",
    "FOGDetector",
    "FOGEpisode",
    "FOGFeatures",
    "FOGType",
    "detect_fog",
    "OrderedPipelineResult",
    "OrderedVideoAnalysisPipeline",
    "get_ordered_video_pipeline",
]
