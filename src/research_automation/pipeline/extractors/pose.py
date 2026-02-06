"""
Pose Feature Extraction from Video
==================================

Extract pose features from video using OpenCV and optional MediaPipe.
Supports both 2D keypoint extraction and derived gait/movement features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# Optional MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


@dataclass
class PoseFrame:
    """Single frame pose data."""

    frame_idx: int
    timestamp: float  # seconds
    keypoints: np.ndarray  # (N, 3) - x, y, confidence
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h


@dataclass
class PoseSequence:
    """Sequence of poses from a video."""

    frames: list[PoseFrame]
    fps: float
    video_width: int
    video_height: int
    n_keypoints: int
    keypoint_names: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with detected pose."""
        valid = sum(1 for f in self.frames if f.keypoints is not None and len(f.keypoints) > 0)
        return valid / len(self.frames) if self.frames else 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array (T, N, 3)."""
        if not self.frames:
            return np.array([])
        return np.stack([f.keypoints for f in self.frames if f.keypoints is not None])

    def get_joint_trajectory(self, joint_idx: int) -> np.ndarray:
        """Get trajectory of a specific joint (T, 3)."""
        arr = self.to_array()
        if len(arr) == 0:
            return np.array([])
        return arr[:, joint_idx, :]


# MediaPipe pose landmark indices
MEDIAPIPE_POSE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Key joint indices for gait analysis
GAIT_JOINTS = {
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_shoulder": 11, "right_shoulder": 12,
}


class PoseExtractor:
    """Extract pose from video frames."""

    def __init__(
        self,
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

    @property
    def pose_model(self):
        """Lazy-load pose model."""
        if self._pose is None and self.use_mediapipe and mp is not None:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._pose

    def extract_from_video(
        self,
        video_path: str | Path,
        sample_rate: int = 1,
        max_frames: int | None = None,
    ) -> PoseSequence:
        """
        Extract pose sequence from video.

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            max_frames: Maximum frames to process

        Returns:
            PoseSequence with extracted poses
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                keypoints = self._extract_frame(frame)

                pose_frame = PoseFrame(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    keypoints=keypoints,
                )
                frames.append(pose_frame)
                processed += 1

                if max_frames and processed >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        n_keypoints = 33 if self.use_mediapipe else 0
        keypoint_names = MEDIAPIPE_POSE_LANDMARKS if self.use_mediapipe else []

        return PoseSequence(
            frames=frames,
            fps=fps / sample_rate,
            video_width=width,
            video_height=height,
            n_keypoints=n_keypoints,
            keypoint_names=keypoint_names,
        )

    def _extract_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract pose from single frame."""
        if not self.use_mediapipe or self.pose_model is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_model.process(rgb)

        if results.pose_landmarks is None:
            return None

        h, w = frame.shape[:2]
        keypoints = []

        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x * w, lm.y * h, lm.visibility])

        return np.array(keypoints)

    def close(self):
        """Release resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class GaitFeatures:
    """Extracted gait features from pose sequence."""

    # Temporal
    duration: float
    n_frames: int
    detection_rate: float

    # Velocity
    walking_speed: float
    speed_variability: float

    # Step features
    step_length_mean: float
    step_length_std: float
    step_width_mean: float
    step_width_std: float

    # Joint angles
    hip_flexion_mean: float
    hip_flexion_range: float
    knee_flexion_mean: float
    knee_flexion_range: float

    # Symmetry
    hip_asymmetry: float
    knee_asymmetry: float
    ankle_asymmetry: float

    # Stability
    trunk_sway: float
    vertical_oscillation: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "n_frames": self.n_frames,
            "detection_rate": self.detection_rate,
            "walking_speed": self.walking_speed,
            "speed_variability": self.speed_variability,
            "step_length_mean": self.step_length_mean,
            "step_length_std": self.step_length_std,
            "step_width_mean": self.step_width_mean,
            "step_width_std": self.step_width_std,
            "hip_flexion_mean": self.hip_flexion_mean,
            "hip_flexion_range": self.hip_flexion_range,
            "knee_flexion_mean": self.knee_flexion_mean,
            "knee_flexion_range": self.knee_flexion_range,
            "hip_asymmetry": self.hip_asymmetry,
            "knee_asymmetry": self.knee_asymmetry,
            "ankle_asymmetry": self.ankle_asymmetry,
            "trunk_sway": self.trunk_sway,
            "vertical_oscillation": self.vertical_oscillation,
        }

    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array(list(self.to_dict().values()))


def extract_gait_features(pose_seq: PoseSequence) -> GaitFeatures:
    """
    Extract gait features from pose sequence.

    Args:
        pose_seq: PoseSequence from video

    Returns:
        GaitFeatures dataclass
    """
    arr = pose_seq.to_array()  # (T, 33, 3)

    if len(arr) < 2:
        # Return zeros if insufficient data
        return GaitFeatures(
            duration=pose_seq.duration,
            n_frames=len(pose_seq.frames),
            detection_rate=pose_seq.detection_rate,
            walking_speed=0, speed_variability=0,
            step_length_mean=0, step_length_std=0,
            step_width_mean=0, step_width_std=0,
            hip_flexion_mean=0, hip_flexion_range=0,
            knee_flexion_mean=0, knee_flexion_range=0,
            hip_asymmetry=0, knee_asymmetry=0, ankle_asymmetry=0,
            trunk_sway=0, vertical_oscillation=0,
        )

    fps = pose_seq.fps

    # Get key joints
    left_hip = arr[:, GAIT_JOINTS["left_hip"], :2]
    right_hip = arr[:, GAIT_JOINTS["right_hip"], :2]
    left_knee = arr[:, GAIT_JOINTS["left_knee"], :2]
    right_knee = arr[:, GAIT_JOINTS["right_knee"], :2]
    left_ankle = arr[:, GAIT_JOINTS["left_ankle"], :2]
    right_ankle = arr[:, GAIT_JOINTS["right_ankle"], :2]
    left_shoulder = arr[:, GAIT_JOINTS["left_shoulder"], :2]
    right_shoulder = arr[:, GAIT_JOINTS["right_shoulder"], :2]

    # Hip center (pelvis proxy)
    hip_center = (left_hip + right_hip) / 2

    # Walking speed (hip center displacement)
    hip_velocity = np.diff(hip_center, axis=0) * fps
    speed = np.linalg.norm(hip_velocity, axis=1)
    walking_speed = speed.mean()
    speed_variability = speed.std()

    # Step length (distance between ankles in forward direction)
    ankle_diff = np.abs(left_ankle[:, 0] - right_ankle[:, 0])  # x-direction
    step_length_mean = ankle_diff.mean()
    step_length_std = ankle_diff.std()

    # Step width (lateral distance between ankles)
    step_width = np.abs(left_ankle[:, 1] - right_ankle[:, 1])  # y-direction
    step_width_mean = step_width.mean()
    step_width_std = step_width.std()

    # Hip flexion angle (simplified: angle at hip)
    def compute_angle(p1, p2, p3):
        """Angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6
        )
        return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    # Left hip angle: shoulder-hip-knee
    left_hip_angle = compute_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = compute_angle(right_shoulder, right_hip, right_knee)
    hip_flexion_mean = (left_hip_angle.mean() + right_hip_angle.mean()) / 2
    hip_flexion_range = max(left_hip_angle.max() - left_hip_angle.min(),
                           right_hip_angle.max() - right_hip_angle.min())

    # Knee flexion angle: hip-knee-ankle
    left_knee_angle = compute_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = compute_angle(right_hip, right_knee, right_ankle)
    knee_flexion_mean = (left_knee_angle.mean() + right_knee_angle.mean()) / 2
    knee_flexion_range = max(left_knee_angle.max() - left_knee_angle.min(),
                            right_knee_angle.max() - right_knee_angle.min())

    # Asymmetry (difference between left and right)
    hip_asymmetry = np.abs(left_hip_angle - right_hip_angle).mean()
    knee_asymmetry = np.abs(left_knee_angle - right_knee_angle).mean()
    ankle_asymmetry = np.abs(
        np.linalg.norm(left_ankle - hip_center, axis=1) -
        np.linalg.norm(right_ankle - hip_center, axis=1)
    ).mean()

    # Trunk sway (shoulder center lateral movement)
    shoulder_center = (left_shoulder + right_shoulder) / 2
    trunk_sway = shoulder_center[:, 0].std()  # lateral (x) variation

    # Vertical oscillation (hip center vertical movement)
    vertical_oscillation = hip_center[:, 1].std()

    return GaitFeatures(
        duration=pose_seq.duration,
        n_frames=len(pose_seq.frames),
        detection_rate=pose_seq.detection_rate,
        walking_speed=walking_speed,
        speed_variability=speed_variability,
        step_length_mean=step_length_mean,
        step_length_std=step_length_std,
        step_width_mean=step_width_mean,
        step_width_std=step_width_std,
        hip_flexion_mean=hip_flexion_mean,
        hip_flexion_range=hip_flexion_range,
        knee_flexion_mean=knee_flexion_mean,
        knee_flexion_range=knee_flexion_range,
        hip_asymmetry=hip_asymmetry,
        knee_asymmetry=knee_asymmetry,
        ankle_asymmetry=ankle_asymmetry,
        trunk_sway=trunk_sway,
        vertical_oscillation=vertical_oscillation,
    )


def extract_pose_from_video(
    video_path: str | Path,
    sample_rate: int = 1,
) -> tuple[PoseSequence, GaitFeatures]:
    """
    Convenience function to extract pose and gait features.

    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame

    Returns:
        Tuple of (PoseSequence, GaitFeatures)
    """
    with PoseExtractor() as extractor:
        pose_seq = extractor.extract_from_video(video_path, sample_rate)

    gait_features = extract_gait_features(pose_seq)
    return pose_seq, gait_features
