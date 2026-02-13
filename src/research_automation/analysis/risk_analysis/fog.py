"""
Freezing of Gait (FOG) Detection
================================

Detect and analyze freezing of gait episodes from video-based pose data.
FOG is characterized by sudden inability to move forward despite intention to walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import numpy as np
from scipy import signal


class FOGType(Enum):
    """Types of freezing episodes."""

    START_HESITATION = "start_hesitation"  # Difficulty initiating gait
    TURN_HESITATION = "turn_hesitation"  # Difficulty during turns
    DESTINATION_HESITATION = "destination_hesitation"  # Near destination
    TIGHT_SPACE = "tight_space"  # In narrow passages
    FESTINATION = "festination"  # Accelerating shuffling
    UNKNOWN = "unknown"


@dataclass
class FOGEpisode:
    """A single freezing episode."""

    start_frame: int
    end_frame: int
    duration_seconds: float
    fog_type: FOGType
    severity: float  # 0-1
    confidence: float  # 0-1

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class FOGFeatures:
    """FOG analysis features for a video."""

    # Detection
    fog_detected: bool
    n_episodes: int

    # Timing
    total_fog_duration: float  # seconds
    fog_percentage: float  # % of video with FOG
    mean_episode_duration: float
    max_episode_duration: float

    # Severity
    mean_severity: float
    max_severity: float

    # Pattern
    fog_frequency: float  # episodes per minute
    freezing_index: float  # Power ratio in freeze band

    # Episodes (default empty)
    episodes: list[FOGEpisode] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary (excludes episode details)."""
        return {
            "fog_detected": float(self.fog_detected),
            "n_episodes": float(self.n_episodes),
            "total_fog_duration": self.total_fog_duration,
            "fog_percentage": self.fog_percentage,
            "mean_episode_duration": self.mean_episode_duration,
            "max_episode_duration": self.max_episode_duration,
            "mean_severity": self.mean_severity,
            "max_severity": self.max_severity,
            "fog_frequency": self.fog_frequency,
            "freezing_index": self.freezing_index,
        }


class FOGDetector:
    """
    Detect freezing of gait from pose sequences.

    Based on:
    - Decreased leg movement amplitude
    - Increased frequency of small shuffling steps
    - Forward trunk lean
    - High-frequency trembling of legs (festination)
    """

    # FOG typically shows high power in 3-8 Hz range
    FREEZE_BAND = (3.0, 8.0)
    LOCOMOTION_BAND = (0.5, 3.0)

    def __init__(
        self,
        min_episode_duration: float = 0.5,  # seconds
        movement_threshold: float = 0.02,  # minimum movement to not be frozen
        window_size: float = 1.0,  # analysis window in seconds
    ):
        """
        Initialize FOG detector.

        Args:
            min_episode_duration: Minimum duration to count as FOG episode
            movement_threshold: Movement below this is considered frozen
            window_size: Sliding window size for analysis
        """
        self.min_episode_duration = min_episode_duration
        self.movement_threshold = movement_threshold
        self.window_size = window_size

    def analyze(
        self,
        keypoints: np.ndarray,
        fps: float,
    ) -> FOGFeatures:
        """
        Analyze FOG from keypoint sequence.

        Args:
            keypoints: (T, N, 2 or 3) keypoint array
            fps: Frames per second

        Returns:
            FOGFeatures
        """
        T = len(keypoints)
        if T < int(fps * self.window_size * 2):
            return self._empty_features()

        duration = T / fps

        # Extract leg joint trajectories (MediaPipe indices)
        # Ankles: 27 (left), 28 (right)
        # Knees: 25 (left), 26 (right)
        ankle_left = keypoints[:, 27, :2] if keypoints.shape[1] > 27 else None
        ankle_right = keypoints[:, 28, :2] if keypoints.shape[1] > 28 else None

        if ankle_left is None or ankle_right is None:
            return self._empty_features()

        # Compute movement features
        movement_left = self._compute_movement(ankle_left, fps)
        movement_right = self._compute_movement(ankle_right, fps)
        movement = (movement_left + movement_right) / 2

        # Compute freezing index (power ratio)
        freezing_index = self._compute_freezing_index(ankle_left, ankle_right, fps)

        # Detect freeze episodes
        episodes = self._detect_episodes(movement, fps, freezing_index)

        if not episodes:
            return FOGFeatures(
                fog_detected=False,
                n_episodes=0,
                episodes=[],
                total_fog_duration=0,
                fog_percentage=0,
                mean_episode_duration=0,
                max_episode_duration=0,
                mean_severity=0,
                max_severity=0,
                fog_frequency=0,
                freezing_index=np.mean(freezing_index) if len(freezing_index) > 0 else 0,
            )

        # Aggregate statistics
        total_duration = sum(ep.duration_seconds for ep in episodes)
        mean_duration = np.mean([ep.duration_seconds for ep in episodes])
        max_duration = max(ep.duration_seconds for ep in episodes)
        mean_severity = np.mean([ep.severity for ep in episodes])
        max_severity = max(ep.severity for ep in episodes)

        return FOGFeatures(
            fog_detected=True,
            n_episodes=len(episodes),
            episodes=episodes,
            total_fog_duration=total_duration,
            fog_percentage=total_duration / duration * 100,
            mean_episode_duration=mean_duration,
            max_episode_duration=max_duration,
            mean_severity=mean_severity,
            max_severity=max_severity,
            fog_frequency=len(episodes) / (duration / 60),  # per minute
            freezing_index=np.mean(freezing_index),
        )

    def _compute_movement(self, positions: np.ndarray, fps: float) -> np.ndarray:
        """Compute movement magnitude over time."""
        # Velocity
        velocity = np.diff(positions, axis=0) * fps
        speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)

        # Smooth with rolling window
        window = int(fps * 0.1)  # 100ms window
        if window > 1:
            kernel = np.ones(window) / window
            speed = np.convolve(speed, kernel, mode="same")

        return speed

    def _compute_freezing_index(
        self,
        ankle_left: np.ndarray,
        ankle_right: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """
        Compute freezing index using power ratio in freeze vs locomotion bands.

        Higher ratio indicates more freezing (high freq trembling, low locomotion).
        """
        # Use vertical (y) movement primarily
        combined = (ankle_left[:, 1] + ankle_right[:, 1]) / 2

        # Compute power in sliding windows
        window_samples = int(fps * self.window_size)
        step = window_samples // 2

        freezing_indices = []

        for i in range(0, len(combined) - window_samples, step):
            window = combined[i:i + window_samples]

            # Detrend
            window = window - np.mean(window)

            # FFT
            freqs = np.fft.fftfreq(len(window), 1 / fps)
            fft_vals = np.abs(np.fft.fft(window))

            # Power in freeze band (3-8 Hz)
            freeze_mask = (np.abs(freqs) >= self.FREEZE_BAND[0]) & (np.abs(freqs) <= self.FREEZE_BAND[1])
            freeze_power = np.sum(fft_vals[freeze_mask] ** 2)

            # Power in locomotion band (0.5-3 Hz)
            loco_mask = (np.abs(freqs) >= self.LOCOMOTION_BAND[0]) & (np.abs(freqs) <= self.LOCOMOTION_BAND[1])
            loco_power = np.sum(fft_vals[loco_mask] ** 2)

            # Freezing index = freeze / (freeze + locomotion)
            total_power = freeze_power + loco_power
            if total_power > 0:
                fi = freeze_power / total_power
            else:
                fi = 0.0

            freezing_indices.append(fi)

        return np.array(freezing_indices)

    def _detect_episodes(
        self,
        movement: np.ndarray,
        fps: float,
        freezing_index: np.ndarray,
    ) -> list[FOGEpisode]:
        """Detect FOG episodes from movement signal."""
        min_frames = int(self.min_episode_duration * fps)

        # Identify low movement periods
        is_frozen = movement < self.movement_threshold

        # Find contiguous frozen regions
        episodes = []
        in_episode = False
        start = 0

        for i, frozen in enumerate(is_frozen):
            if frozen and not in_episode:
                in_episode = True
                start = i
            elif not frozen and in_episode:
                in_episode = False
                duration_frames = i - start
                if duration_frames >= min_frames:
                    episodes.append(self._create_episode(
                        start, i, fps, movement, freezing_index
                    ))

        # Handle episode at end
        if in_episode:
            duration_frames = len(is_frozen) - start
            if duration_frames >= min_frames:
                episodes.append(self._create_episode(
                    start, len(is_frozen), fps, movement, freezing_index
                ))

        return episodes

    def _create_episode(
        self,
        start: int,
        end: int,
        fps: float,
        movement: np.ndarray,
        freezing_index: np.ndarray,
    ) -> FOGEpisode:
        """Create a FOG episode."""
        duration = (end - start) / fps

        # Severity based on how little movement there is
        episode_movement = movement[start:end]
        severity = 1.0 - min(1.0, np.mean(episode_movement) / self.movement_threshold)

        # Confidence based on freezing index
        # Map episode frames to FI windows
        window_samples = int(fps * self.window_size)
        step = window_samples // 2
        fi_start = max(0, start // step - 1)
        fi_end = min(len(freezing_index), end // step + 1)

        if fi_end > fi_start and len(freezing_index) > 0:
            episode_fi = freezing_index[fi_start:fi_end]
            confidence = np.mean(episode_fi)
        else:
            confidence = 0.5

        # Classify FOG type (simplified)
        fog_type = self._classify_fog_type(start, end, fps)

        return FOGEpisode(
            start_frame=start,
            end_frame=end,
            duration_seconds=duration,
            fog_type=fog_type,
            severity=severity,
            confidence=confidence,
        )

    def _classify_fog_type(
        self,
        start: int,
        end: int,
        fps: float,
    ) -> FOGType:
        """
        Classify FOG type based on position in video.

        Note: More sophisticated classification would use context
        (turning, obstacles, etc.)
        """
        # Simple heuristic based on position
        if start < fps * 2:  # First 2 seconds
            return FOGType.START_HESITATION
        return FOGType.UNKNOWN

    def _empty_features(self) -> FOGFeatures:
        """Return empty features."""
        return FOGFeatures(
            fog_detected=False,
            n_episodes=0,
            episodes=[],
            total_fog_duration=0,
            fog_percentage=0,
            mean_episode_duration=0,
            max_episode_duration=0,
            mean_severity=0,
            max_severity=0,
            fog_frequency=0,
            freezing_index=0,
        )


def detect_fog(
    keypoints: np.ndarray,
    fps: float,
) -> FOGFeatures:
    """
    Convenience function to detect FOG.

    Args:
        keypoints: (T, N, 2 or 3) keypoint array
        fps: Frames per second

    Returns:
        FOGFeatures
    """
    detector = FOGDetector()
    return detector.analyze(keypoints, fps)
