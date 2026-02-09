"""
Walking Detection Module
========================

Detect walking segments from pose sequences.
Essential for filtering video data before gait analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


@dataclass
class WalkingSegment:
    """A detected walking segment."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    mean_speed: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class WalkingDetectionResult:
    """Result of walking detection."""

    segments: List[WalkingSegment]
    is_walking: np.ndarray  # Boolean mask for each frame
    walking_confidence: np.ndarray  # Confidence score for each frame
    total_walking_time: float
    total_video_time: float
    walking_ratio: float

    def get_walking_frames(self, keypoints: np.ndarray) -> np.ndarray:
        """Extract only walking frames from keypoints."""
        return keypoints[self.is_walking]

    def get_segment_keypoints(self, keypoints: np.ndarray, segment_idx: int) -> np.ndarray:
        """Extract keypoints for a specific walking segment."""
        seg = self.segments[segment_idx]
        return keypoints[seg.start_frame:seg.end_frame]


class WalkingDetector:
    """
    Detect walking segments from pose sequences.

    Walking is detected based on:
    1. Rhythmic leg movement (alternating left-right pattern)
    2. Forward/horizontal progression
    3. Vertical oscillation (body bouncing)
    4. Consistent movement speed above threshold
    """

    # MediaPipe pose landmark indices
    ANKLE_LEFT = 27
    ANKLE_RIGHT = 28
    KNEE_LEFT = 25
    KNEE_RIGHT = 26
    HIP_LEFT = 23
    HIP_RIGHT = 24
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12

    def __init__(
        self,
        min_walking_duration: float = 1.0,  # Minimum seconds to count as walking
        speed_threshold: float = 0.01,  # Minimum movement speed
        rhythm_threshold: float = 0.3,  # Minimum rhythm score
        smoothing_window: float = 0.5,  # Smoothing window in seconds
    ):
        self.min_walking_duration = min_walking_duration
        self.speed_threshold = speed_threshold
        self.rhythm_threshold = rhythm_threshold
        self.smoothing_window = smoothing_window

    def detect(
        self,
        keypoints: np.ndarray,
        fps: float,
    ) -> WalkingDetectionResult:
        """
        Detect walking segments in pose sequence.

        Args:
            keypoints: (T, N, 2 or 3) keypoint array
            fps: Frames per second

        Returns:
            WalkingDetectionResult
        """
        T = len(keypoints)
        if T < int(fps * 0.5):
            return self._empty_result(T, fps)

        # Extract joint positions
        ankle_left = keypoints[:, self.ANKLE_LEFT, :2]
        ankle_right = keypoints[:, self.ANKLE_RIGHT, :2]
        hip_center = (keypoints[:, self.HIP_LEFT, :2] + keypoints[:, self.HIP_RIGHT, :2]) / 2

        # Compute features for walking detection
        speed_score = self._compute_speed_score(ankle_left, ankle_right, fps)
        rhythm_score = self._compute_rhythm_score(ankle_left, ankle_right, fps)
        progression_score = self._compute_progression_score(hip_center, fps)
        oscillation_score = self._compute_oscillation_score(hip_center, fps)

        # Combine scores
        walking_confidence = (
            0.35 * speed_score +
            0.30 * rhythm_score +
            0.20 * progression_score +
            0.15 * oscillation_score
        )

        # Smooth the confidence
        window_frames = max(3, int(self.smoothing_window * fps))
        walking_confidence = uniform_filter1d(walking_confidence, window_frames)

        # Threshold to get walking mask
        is_walking = walking_confidence > 0.4

        # Apply minimum duration filter
        is_walking = self._filter_short_segments(is_walking, fps)

        # Extract segments
        segments = self._extract_segments(is_walking, walking_confidence, speed_score, fps)

        # Compute statistics
        total_walking_frames = np.sum(is_walking)
        total_walking_time = total_walking_frames / fps
        total_video_time = T / fps
        walking_ratio = total_walking_time / total_video_time if total_video_time > 0 else 0

        return WalkingDetectionResult(
            segments=segments,
            is_walking=is_walking,
            walking_confidence=walking_confidence,
            total_walking_time=total_walking_time,
            total_video_time=total_video_time,
            walking_ratio=walking_ratio,
        )

    def _compute_speed_score(
        self,
        ankle_left: np.ndarray,
        ankle_right: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """Compute movement speed score."""
        # Velocity of ankles
        vel_left = np.diff(ankle_left, axis=0) * fps
        vel_right = np.diff(ankle_right, axis=0) * fps

        speed_left = np.sqrt(vel_left[:, 0]**2 + vel_left[:, 1]**2)
        speed_right = np.sqrt(vel_right[:, 0]**2 + vel_right[:, 1]**2)

        # Combined speed
        speed = (speed_left + speed_right) / 2

        # Pad to match original length
        speed = np.concatenate([[speed[0]], speed])

        # Normalize to 0-1
        speed_threshold = self.speed_threshold
        speed_score = np.clip(speed / (speed_threshold * 5), 0, 1)

        return speed_score

    def _compute_rhythm_score(
        self,
        ankle_left: np.ndarray,
        ankle_right: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """
        Compute rhythm score based on alternating leg movement.
        Walking has a characteristic left-right alternating pattern.
        """
        T = len(ankle_left)

        # Compute leg phase (x-position difference)
        leg_diff = ankle_left[:, 0] - ankle_right[:, 0]

        # Compute local rhythm using sliding window correlation
        window_size = int(fps * 1.0)  # 1 second window
        if window_size < 4:
            window_size = 4

        rhythm_score = np.zeros(T)

        for i in range(T):
            start = max(0, i - window_size // 2)
            end = min(T, i + window_size // 2)
            window = leg_diff[start:end]

            if len(window) < 4:
                continue

            # Check for alternating pattern (zero crossings)
            zero_crossings = np.sum(np.diff(np.sign(window - np.mean(window))) != 0)
            expected_crossings = len(window) / (fps / 2)  # ~2 steps per second

            # Score based on regularity of crossings
            if expected_crossings > 0:
                rhythm_score[i] = min(1.0, zero_crossings / (expected_crossings * 2))

        return rhythm_score

    def _compute_progression_score(
        self,
        hip_center: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """Compute forward/horizontal progression score."""
        T = len(hip_center)

        # Compute horizontal movement over sliding window
        window_size = int(fps * 0.5)
        if window_size < 2:
            window_size = 2

        progression_score = np.zeros(T)

        for i in range(window_size, T):
            displacement = np.sqrt(
                (hip_center[i, 0] - hip_center[i - window_size, 0])**2
            )
            # Normalize by expected walking displacement
            progression_score[i] = min(1.0, displacement / 0.1)

        # Fill beginning
        progression_score[:window_size] = progression_score[window_size]

        return progression_score

    def _compute_oscillation_score(
        self,
        hip_center: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """
        Compute vertical oscillation score.
        Walking produces characteristic up-down motion.
        """
        T = len(hip_center)

        # Vertical position
        y_pos = hip_center[:, 1]

        # Detrend
        y_detrend = y_pos - uniform_filter1d(y_pos, max(3, int(fps)))

        # Compute local variance (oscillation amplitude)
        window_size = int(fps * 0.5)
        if window_size < 2:
            window_size = 2

        oscillation_score = np.zeros(T)

        for i in range(window_size, T - window_size):
            window = y_detrend[i - window_size:i + window_size]
            variance = np.var(window)
            # Walking typically has variance in range 0.0001 - 0.01
            oscillation_score[i] = min(1.0, variance / 0.005)

        return oscillation_score

    def _filter_short_segments(
        self,
        is_walking: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """Remove walking segments shorter than minimum duration."""
        min_frames = int(self.min_walking_duration * fps)

        result = is_walking.copy()

        # Find contiguous segments
        changes = np.diff(np.concatenate([[0], is_walking.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_frames:
                result[start:end] = False

        return result

    def _extract_segments(
        self,
        is_walking: np.ndarray,
        confidence: np.ndarray,
        speed: np.ndarray,
        fps: float,
    ) -> List[WalkingSegment]:
        """Extract walking segments from mask."""
        segments = []

        changes = np.diff(np.concatenate([[0], is_walking.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            seg_confidence = confidence[start:end]
            seg_speed = speed[start:end]

            segments.append(WalkingSegment(
                start_frame=int(start),
                end_frame=int(end),
                start_time=start / fps,
                end_time=end / fps,
                confidence=float(np.mean(seg_confidence)),
                mean_speed=float(np.mean(seg_speed)),
            ))

        return segments

    def _empty_result(self, T: int, fps: float) -> WalkingDetectionResult:
        """Return empty result for short sequences."""
        return WalkingDetectionResult(
            segments=[],
            is_walking=np.zeros(T, dtype=bool),
            walking_confidence=np.zeros(T),
            total_walking_time=0,
            total_video_time=T / fps,
            walking_ratio=0,
        )


def detect_walking(
    keypoints: np.ndarray,
    fps: float,
    min_duration: float = 1.0,
) -> WalkingDetectionResult:
    """
    Convenience function to detect walking segments.

    Args:
        keypoints: (T, N, 2 or 3) keypoint array
        fps: Frames per second
        min_duration: Minimum walking segment duration

    Returns:
        WalkingDetectionResult
    """
    detector = WalkingDetector(min_walking_duration=min_duration)
    return detector.detect(keypoints, fps)
