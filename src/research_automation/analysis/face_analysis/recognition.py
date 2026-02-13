"""Face recognition stage for video analysis pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from research_automation.pipeline.extractors.face import FaceExtractor


@dataclass
class FaceIdentity:
    """Identified person from face analysis."""

    person_id: str
    name: Optional[str]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "confidence": float(self.confidence),
        }


@dataclass
class FaceRecognitionResult:
    """Face recognition output for a full video."""

    detected: bool
    detection_rate: float
    analyzed_frames: int
    primary_identity: Optional[FaceIdentity]

    def to_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "detection_rate": float(self.detection_rate),
            "analyzed_frames": int(self.analyzed_frames),
            "primary_identity": self.primary_identity.to_dict() if self.primary_identity else None,
        }


class FaceRecognitionAnalyzer:
    """Simple face stage that can optionally match registered embeddings."""

    def __init__(self, detection_threshold: float = 0.05, match_threshold: float = 0.90):
        self.detection_threshold = detection_threshold
        self.match_threshold = match_threshold
        self._profiles: Dict[str, tuple[str, np.ndarray]] = {}

    def register_profile(self, person_id: str, name: str, face_embedding: np.ndarray) -> None:
        """Register a profile embedding for identity matching."""
        if face_embedding.ndim != 1:
            raise ValueError("face_embedding must be a 1D vector")
        self._profiles[person_id] = (name, self._normalize(face_embedding))

    def analyze_video(
        self,
        video_path: str | Path,
        sample_rate: int = 5,
        max_frames: int | None = 500,
    ) -> FaceRecognitionResult:
        """Run face recognition stage for a video."""
        with FaceExtractor(use_mediapipe=True) as extractor:
            sequence = extractor.extract_from_video(
                video_path=video_path,
                sample_rate=sample_rate,
                max_frames=max_frames,
            )

        detected_frames = [f for f in sequence.frames if f.detected]
        detection_rate = sequence.detection_rate
        detected = detection_rate >= self.detection_threshold and len(detected_frames) > 0

        if not detected:
            return FaceRecognitionResult(
                detected=False,
                detection_rate=detection_rate,
                analyzed_frames=len(sequence.frames),
                primary_identity=None,
            )

        # If no registered profiles, assign the default tracked person identity.
        if not self._profiles:
            return FaceRecognitionResult(
                detected=True,
                detection_rate=detection_rate,
                analyzed_frames=len(sequence.frames),
                primary_identity=FaceIdentity(
                    person_id="person_1",
                    name="Unknown",
                    confidence=min(1.0, 0.5 + detection_rate / 2),
                ),
            )

        # Use average landmark geometry signature as a deterministic embedding.
        signatures = []
        for frame in detected_frames:
            if frame.landmarks is None or len(frame.landmarks) == 0:
                continue
            signatures.append(self._landmark_signature(frame.landmarks))

        if not signatures:
            return FaceRecognitionResult(
                detected=True,
                detection_rate=detection_rate,
                analyzed_frames=len(sequence.frames),
                primary_identity=FaceIdentity(person_id="person_1", name="Unknown", confidence=0.5),
            )

        query = self._normalize(np.mean(np.stack(signatures), axis=0))
        matched_id, matched_name, confidence = self._match(query)

        if matched_id is None:
            return FaceRecognitionResult(
                detected=True,
                detection_rate=detection_rate,
                analyzed_frames=len(sequence.frames),
                primary_identity=FaceIdentity(person_id="person_1", name="Unknown", confidence=0.5),
            )

        return FaceRecognitionResult(
            detected=True,
            detection_rate=detection_rate,
            analyzed_frames=len(sequence.frames),
            primary_identity=FaceIdentity(person_id=matched_id, name=matched_name, confidence=confidence),
        )

    def _match(self, query_embedding: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        best_person_id = None
        best_name = None
        best_score = -1.0

        for person_id, (name, embedding) in self._profiles.items():
            score = float(np.dot(query_embedding, embedding))
            if score > best_score:
                best_score = score
                best_person_id = person_id
                best_name = name

        if best_score < self.match_threshold:
            return None, None, best_score
        return best_person_id, best_name, best_score

    def _landmark_signature(self, landmarks: np.ndarray) -> np.ndarray:
        """Build scale-invariant geometric signature from facial landmarks."""
        pts = landmarks[:, :2]
        center = pts.mean(axis=0)
        centered = pts - center
        scale = np.linalg.norm(centered, axis=1).mean() + 1e-6
        normalized = centered / scale

        # Reduce to deterministic vector with stable summary stats.
        return np.concatenate(
            [
                normalized.mean(axis=0),
                normalized.std(axis=0),
                np.percentile(normalized[:, 0], [10, 25, 50, 75, 90]),
                np.percentile(normalized[:, 1], [10, 25, 50, 75, 90]),
            ]
        ).astype(np.float64)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector) + 1e-8
        return vector / norm
