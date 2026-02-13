"""Behavior analysis stage that runs per identified person."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bradykinesia import BradykinesiaAnalyzer, BradykinesiaFeatures
from .tremor import TremorDetector, TremorFeatures
from .walking_detection import WalkingDetectionResult, WalkingDetector


@dataclass
class PersonBehaviorAnalysis:
    """Behavior-level analysis outputs for one person."""

    person_id: str
    walking: WalkingDetectionResult
    tremor: TremorFeatures
    bradykinesia: BradykinesiaFeatures

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "walking": {
                "segments": [
                    {
                        "start_frame": s.start_frame,
                        "end_frame": s.end_frame,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "duration": s.duration,
                        "confidence": s.confidence,
                        "mean_speed": s.mean_speed,
                    }
                    for s in self.walking.segments
                ],
                "total_walking_time": self.walking.total_walking_time,
                "total_video_time": self.walking.total_video_time,
                "walking_ratio": self.walking.walking_ratio,
            },
            "tremor": self.tremor.to_dict(),
            "bradykinesia": self.bradykinesia.to_dict(),
        }


class BehaviorAnalyzer:
    """Analyze behavior features for each person track."""

    def __init__(self):
        self.walking_detector = WalkingDetector()
        self.tremor_detector = TremorDetector()
        self.bradykinesia_analyzer = BradykinesiaAnalyzer()

    def analyze_person(
        self,
        person_id: str,
        keypoints: np.ndarray,
        fps: float,
    ) -> PersonBehaviorAnalysis:
        """Run behavior analysis for a person."""
        walking = self.walking_detector.detect(keypoints, fps)
        tremor = self.tremor_detector.analyze(keypoints, fps)
        bradykinesia = self.bradykinesia_analyzer.analyze_gait(keypoints, fps)

        return PersonBehaviorAnalysis(
            person_id=person_id,
            walking=walking,
            tremor=tremor,
            bradykinesia=bradykinesia,
        )
