"""Ordered video analysis pipeline.

Execution order is strictly:
1) Face recognition
2) Behavior analysis per identified person
3) Risk analysis per action
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_automation.pipeline.extractors.pose import PoseExtractor

from .behavior_analysis import BehaviorAnalyzer
from .face_analysis import FaceRecognitionAnalyzer, FaceRecognitionResult
from .risk_analysis import ActionRiskAnalyzer, PersonRiskAnalysis


@dataclass
class OrderedPipelineResult:
    """Structured output from the ordered video analysis pipeline."""

    video_path: str
    stage_order: list[str]
    face: FaceRecognitionResult
    behavior_by_person: dict[str, dict[str, Any]]
    risk_by_person: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "stage_order": self.stage_order,
            "face_analysis": self.face.to_dict(),
            "behavior_analysis": self.behavior_by_person,
            "risk_analysis": self.risk_by_person,
        }


class OrderedVideoAnalysisPipeline:
    """Orchestrates face -> behavior(person) -> risk(action) analysis."""

    STAGE_ORDER = [
        "face_recognition",
        "behavior_analysis_per_person",
        "risk_analysis_per_action",
    ]

    def __init__(
        self,
        face_analyzer: FaceRecognitionAnalyzer | None = None,
        behavior_analyzer: BehaviorAnalyzer | None = None,
        risk_analyzer: ActionRiskAnalyzer | None = None,
    ):
        self.face_analyzer = face_analyzer or FaceRecognitionAnalyzer()
        self.behavior_analyzer = behavior_analyzer or BehaviorAnalyzer()
        self.risk_analyzer = risk_analyzer or ActionRiskAnalyzer()

    def analyze_video(
        self,
        video_path: str | Path,
        sample_rate: int = 1,
        max_frames: int | None = None,
    ) -> OrderedPipelineResult:
        """Run pipeline in fixed stage order."""
        video_path = Path(video_path)

        # Stage 1: Face recognition
        face_result = self.face_analyzer.analyze_video(video_path)
        person_id = (
            face_result.primary_identity.person_id
            if face_result.primary_identity is not None
            else "person_1"
        )

        # Shared pose extraction for downstream stages
        with PoseExtractor(use_mediapipe=True) as extractor:
            pose_seq = extractor.extract_from_video(
                video_path=video_path,
                sample_rate=sample_rate,
                max_frames=max_frames,
            )

        keypoints = pose_seq.to_array()
        if keypoints.size == 0:
            raise ValueError("No pose keypoints detected in video")

        # Stage 2: Behavior analysis per person
        behavior = self.behavior_analyzer.analyze_person(
            person_id=person_id,
            keypoints=keypoints,
            fps=pose_seq.fps,
        )

        # Stage 3: Risk analysis per action
        risk: PersonRiskAnalysis = self.risk_analyzer.analyze_person(
            behavior=behavior,
            keypoints=keypoints,
            fps=pose_seq.fps,
        )

        return OrderedPipelineResult(
            video_path=str(video_path),
            stage_order=self.STAGE_ORDER,
            face=face_result,
            behavior_by_person={person_id: behavior.to_dict()},
            risk_by_person={person_id: risk.to_dict()},
        )


_pipeline: OrderedVideoAnalysisPipeline | None = None


def get_ordered_video_pipeline() -> OrderedVideoAnalysisPipeline:
    """Get or create singleton ordered pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = OrderedVideoAnalysisPipeline()
    return _pipeline
