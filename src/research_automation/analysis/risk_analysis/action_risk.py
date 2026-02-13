"""Risk analysis stage based on behavior analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from research_automation.analysis.behavior_analysis.analyzer import PersonBehaviorAnalysis

from .fog import FOGDetector, FOGFeatures


@dataclass
class ActionRisk:
    """Risk record for one action type."""

    action_type: str
    risk_score: float  # 0-1
    risk_level: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
            "details": self.details,
        }


@dataclass
class PersonRiskAnalysis:
    """Action-wise risk analysis for one person."""

    person_id: str
    fog: FOGFeatures
    action_risks: list[ActionRisk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "fog": self.fog.to_dict(),
            "action_risks": [risk.to_dict() for risk in self.action_risks],
        }


class ActionRiskAnalyzer:
    """Derive risks from behavior features (action-specific)."""

    def __init__(self):
        self.fog_detector = FOGDetector()

    def analyze_person(
        self,
        behavior: PersonBehaviorAnalysis,
        keypoints: np.ndarray,
        fps: float,
    ) -> PersonRiskAnalysis:
        """Analyze risk stage for one person."""
        fog_features = self.fog_detector.analyze(keypoints, fps)

        walking_risk = self._walking_risk(behavior, fog_features)
        tremor_risk = self._tremor_risk(behavior)
        bradykinesia_risk = self._bradykinesia_risk(behavior)

        return PersonRiskAnalysis(
            person_id=behavior.person_id,
            fog=fog_features,
            action_risks=[walking_risk, tremor_risk, bradykinesia_risk],
        )

    def _walking_risk(self, behavior: PersonBehaviorAnalysis, fog: FOGFeatures) -> ActionRisk:
        walking_ratio = behavior.walking.walking_ratio
        fog_ratio = fog.fog_percentage / 100.0

        score = float(np.clip(0.7 * fog_ratio + 0.3 * (1.0 - walking_ratio), 0.0, 1.0))
        return ActionRisk(
            action_type="walking",
            risk_score=score,
            risk_level=self._to_level(score),
            details={
                "walking_ratio": walking_ratio,
                "fog_detected": fog.fog_detected,
                "n_fog_episodes": fog.n_episodes,
                "fog_percentage": fog.fog_percentage,
            },
        )

    def _tremor_risk(self, behavior: PersonBehaviorAnalysis) -> ActionRisk:
        tremor = behavior.tremor
        score = 0.0
        if tremor.tremor_detected:
            freq_score = np.clip((tremor.dominant_frequency - 3.0) / 5.0, 0.0, 1.0)
            amp_score = np.clip(tremor.amplitude_mean / 0.1, 0.0, 1.0)
            active_score = np.clip(tremor.tremor_percentage, 0.0, 1.0)
            score = float(np.clip(0.4 * freq_score + 0.3 * amp_score + 0.3 * active_score, 0.0, 1.0))

        return ActionRisk(
            action_type="tremor",
            risk_score=score,
            risk_level=self._to_level(score),
            details=tremor.to_dict(),
        )

    def _bradykinesia_risk(self, behavior: PersonBehaviorAnalysis) -> ActionRisk:
        brady = behavior.bradykinesia
        score = float(np.clip(brady.bradykinesia_score / 4.0, 0.0, 1.0))

        return ActionRisk(
            action_type="bradykinesia",
            risk_score=score,
            risk_level=self._to_level(score),
            details=brady.to_dict(),
        )

    def _to_level(self, score: float) -> str:
        if score < 0.25:
            return "low"
        if score < 0.5:
            return "mild"
        if score < 0.75:
            return "moderate"
        return "high"
