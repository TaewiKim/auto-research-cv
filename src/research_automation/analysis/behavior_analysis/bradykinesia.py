"""
Bradykinesia Analysis
=====================

Analyze slowness of movement and motor control from video-based pose data.
Bradykinesia is a cardinal symptom of Parkinson's Disease.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal


@dataclass
class BradykinesiaFeatures:
    """Bradykinesia analysis features."""

    # Speed
    mean_speed: float
    speed_decrement: float  # How much speed decreases over time
    speed_variability: float

    # Amplitude
    mean_amplitude: float
    amplitude_decrement: float  # How much amplitude decreases
    amplitude_variability: float

    # Hesitation
    hesitation_ratio: float  # % time with very slow movement
    pause_count: int  # Number of pauses
    mean_pause_duration: float

    # Rhythm
    rhythm_regularity: float
    movement_frequency: float  # Hz

    # Overall score (0-4 like UPDRS)
    bradykinesia_score: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_speed": self.mean_speed,
            "speed_decrement": self.speed_decrement,
            "speed_variability": self.speed_variability,
            "mean_amplitude": self.mean_amplitude,
            "amplitude_decrement": self.amplitude_decrement,
            "amplitude_variability": self.amplitude_variability,
            "hesitation_ratio": self.hesitation_ratio,
            "pause_count": float(self.pause_count),
            "mean_pause_duration": self.mean_pause_duration,
            "rhythm_regularity": self.rhythm_regularity,
            "movement_frequency": self.movement_frequency,
            "bradykinesia_score": self.bradykinesia_score,
        }


class BradykinesiaAnalyzer:
    """
    Analyze bradykinesia from repetitive movements.

    Designed for finger tapping, hand movements, or gait analysis.
    """

    def __init__(
        self,
        pause_threshold: float = 0.02,  # Speed below this is a pause
        min_pause_duration: float = 0.1,  # seconds
    ):
        """
        Initialize analyzer.

        Args:
            pause_threshold: Speed threshold for pauses
            min_pause_duration: Minimum pause duration
        """
        self.pause_threshold = pause_threshold
        self.min_pause_duration = min_pause_duration

    def analyze_finger_tap(
        self,
        keypoints: np.ndarray,
        fps: float,
    ) -> BradykinesiaFeatures:
        """
        Analyze finger tapping task.

        Args:
            keypoints: (T, N, 2 or 3) keypoint array
            fps: Frames per second

        Returns:
            BradykinesiaFeatures
        """
        T = len(keypoints)
        if T < int(fps * 2):
            return self._empty_features()

        # Use thumb and index finger tips (MediaPipe indices 4 and 8)
        thumb = keypoints[:, 4, :2] if keypoints.shape[1] > 4 else None
        index = keypoints[:, 8, :2] if keypoints.shape[1] > 8 else None

        if thumb is None or index is None:
            # Fallback to wrist movement
            if keypoints.shape[1] > 15:
                wrist = keypoints[:, 15, :2]
                return self._analyze_single_joint(wrist, fps)
            return self._empty_features()

        # Compute thumb-index distance over time
        distance = np.sqrt(np.sum((thumb - index) ** 2, axis=1))

        return self._analyze_tapping(distance, fps)

    def analyze_gait(
        self,
        keypoints: np.ndarray,
        fps: float,
    ) -> BradykinesiaFeatures:
        """
        Analyze gait bradykinesia.

        Args:
            keypoints: (T, N, 2 or 3) keypoint array
            fps: Frames per second

        Returns:
            BradykinesiaFeatures
        """
        T = len(keypoints)
        if T < int(fps * 2):
            return self._empty_features()

        # Use ankle positions (MediaPipe indices 27, 28)
        left_ankle = keypoints[:, 27, :2] if keypoints.shape[1] > 27 else None
        right_ankle = keypoints[:, 28, :2] if keypoints.shape[1] > 28 else None

        if left_ankle is None or right_ankle is None:
            return self._empty_features()

        # Compute step lengths over time
        step_length = np.abs(left_ankle[:, 0] - right_ankle[:, 0])

        # Compute velocities
        left_vel = np.diff(left_ankle, axis=0) * fps
        right_vel = np.diff(right_ankle, axis=0) * fps
        left_speed = np.sqrt(np.sum(left_vel ** 2, axis=1))
        right_speed = np.sqrt(np.sum(right_vel ** 2, axis=1))
        avg_speed = (left_speed + right_speed) / 2

        return self._analyze_gait_features(avg_speed, step_length, fps)

    def _analyze_single_joint(
        self,
        positions: np.ndarray,
        fps: float,
    ) -> BradykinesiaFeatures:
        """Analyze movement of a single joint."""
        # Compute velocity
        velocity = np.diff(positions, axis=0) * fps
        speed = np.sqrt(np.sum(velocity ** 2, axis=1))

        # Speed metrics
        mean_speed = np.mean(speed)
        speed_variability = np.std(speed) / (mean_speed + 1e-6)

        # Speed decrement (compare first half to second half)
        half = len(speed) // 2
        first_half_speed = np.mean(speed[:half])
        second_half_speed = np.mean(speed[half:])
        speed_decrement = (first_half_speed - second_half_speed) / (first_half_speed + 1e-6)

        # Amplitude from displacement
        amplitude = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        mean_amplitude = np.mean(amplitude)
        amplitude_variability = np.std(amplitude) / (mean_amplitude + 1e-6)

        first_half_amp = np.mean(amplitude[:half])
        second_half_amp = np.mean(amplitude[half:])
        amplitude_decrement = (first_half_amp - second_half_amp) / (first_half_amp + 1e-6)

        # Pauses
        pauses = self._detect_pauses(speed, fps)
        hesitation_ratio = np.mean(speed < self.pause_threshold)

        # Rhythm (autocorrelation)
        regularity, frequency = self._analyze_rhythm(speed, fps)

        # Score
        score = self._compute_score(
            speed_decrement, amplitude_decrement,
            hesitation_ratio, regularity
        )

        return BradykinesiaFeatures(
            mean_speed=mean_speed,
            speed_decrement=speed_decrement,
            speed_variability=speed_variability,
            mean_amplitude=mean_amplitude,
            amplitude_decrement=amplitude_decrement,
            amplitude_variability=amplitude_variability,
            hesitation_ratio=hesitation_ratio,
            pause_count=len(pauses),
            mean_pause_duration=np.mean(pauses) if pauses else 0,
            rhythm_regularity=regularity,
            movement_frequency=frequency,
            bradykinesia_score=score,
        )

    def _analyze_tapping(
        self,
        distance: np.ndarray,
        fps: float,
    ) -> BradykinesiaFeatures:
        """Analyze finger tapping distance signal."""
        # Find tap cycles (local minima = finger closed)
        # Smooth first
        window = max(3, int(fps * 0.05))
        smoothed = np.convolve(distance, np.ones(window) / window, mode="same")

        # Find peaks (maxima = fingers apart)
        peaks, properties = signal.find_peaks(smoothed, distance=int(fps * 0.1))

        if len(peaks) < 3:
            return self._empty_features()

        # Compute tap amplitudes
        amplitudes = smoothed[peaks]
        mean_amplitude = np.mean(amplitudes)
        amplitude_variability = np.std(amplitudes) / (mean_amplitude + 1e-6)

        # Amplitude decrement
        half = len(amplitudes) // 2
        if half > 0:
            first_half = np.mean(amplitudes[:half])
            second_half = np.mean(amplitudes[half:])
            amplitude_decrement = (first_half - second_half) / (first_half + 1e-6)
        else:
            amplitude_decrement = 0

        # Tap intervals (speed)
        intervals = np.diff(peaks) / fps
        mean_interval = np.mean(intervals)
        speed = 1 / mean_interval if mean_interval > 0 else 0
        speed_variability = np.std(intervals) / (mean_interval + 1e-6)

        # Speed decrement
        half = len(intervals) // 2
        if half > 0:
            first_half = np.mean(intervals[:half])
            second_half = np.mean(intervals[half:])
            speed_decrement = (second_half - first_half) / (first_half + 1e-6)  # Longer intervals = slower
        else:
            speed_decrement = 0

        # Pauses (very long intervals)
        pause_threshold = mean_interval * 2
        pauses = [i for i in intervals if i > pause_threshold]
        hesitation_ratio = len(pauses) / len(intervals)

        # Regularity
        regularity = 1 - min(1, speed_variability)

        # Score
        score = self._compute_score(
            speed_decrement, amplitude_decrement,
            hesitation_ratio, regularity
        )

        return BradykinesiaFeatures(
            mean_speed=speed,
            speed_decrement=speed_decrement,
            speed_variability=speed_variability,
            mean_amplitude=mean_amplitude,
            amplitude_decrement=amplitude_decrement,
            amplitude_variability=amplitude_variability,
            hesitation_ratio=hesitation_ratio,
            pause_count=len(pauses),
            mean_pause_duration=np.mean(pauses) if pauses else 0,
            rhythm_regularity=regularity,
            movement_frequency=speed,
            bradykinesia_score=score,
        )

    def _analyze_gait_features(
        self,
        speed: np.ndarray,
        step_length: np.ndarray,
        fps: float,
    ) -> BradykinesiaFeatures:
        """Analyze gait-specific bradykinesia features."""
        # Speed metrics
        mean_speed = np.mean(speed)
        speed_variability = np.std(speed) / (mean_speed + 1e-6)

        half = len(speed) // 2
        first_half_speed = np.mean(speed[:half])
        second_half_speed = np.mean(speed[half:])
        speed_decrement = (first_half_speed - second_half_speed) / (first_half_speed + 1e-6)

        # Step length as amplitude
        mean_amplitude = np.mean(step_length)
        amplitude_variability = np.std(step_length) / (mean_amplitude + 1e-6)

        first_half_amp = np.mean(step_length[:half])
        second_half_amp = np.mean(step_length[half:])
        amplitude_decrement = (first_half_amp - second_half_amp) / (first_half_amp + 1e-6)

        # Pauses
        pauses = self._detect_pauses(speed, fps)
        hesitation_ratio = np.mean(speed < self.pause_threshold)

        # Rhythm
        regularity, frequency = self._analyze_rhythm(speed, fps)

        # Score
        score = self._compute_score(
            speed_decrement, amplitude_decrement,
            hesitation_ratio, regularity
        )

        return BradykinesiaFeatures(
            mean_speed=mean_speed,
            speed_decrement=speed_decrement,
            speed_variability=speed_variability,
            mean_amplitude=mean_amplitude,
            amplitude_decrement=amplitude_decrement,
            amplitude_variability=amplitude_variability,
            hesitation_ratio=hesitation_ratio,
            pause_count=len(pauses),
            mean_pause_duration=np.mean(pauses) if pauses else 0,
            rhythm_regularity=regularity,
            movement_frequency=frequency,
            bradykinesia_score=score,
        )

    def _detect_pauses(
        self,
        speed: np.ndarray,
        fps: float,
    ) -> list[float]:
        """Detect pauses in movement."""
        min_frames = int(self.min_pause_duration * fps)

        is_paused = speed < self.pause_threshold
        pauses = []
        in_pause = False
        start = 0

        for i, paused in enumerate(is_paused):
            if paused and not in_pause:
                in_pause = True
                start = i
            elif not paused and in_pause:
                in_pause = False
                duration = i - start
                if duration >= min_frames:
                    pauses.append(duration / fps)

        if in_pause:
            duration = len(is_paused) - start
            if duration >= min_frames:
                pauses.append(duration / fps)

        return pauses

    def _analyze_rhythm(
        self,
        speed: np.ndarray,
        fps: float,
    ) -> Tuple[float, float]:
        """Analyze rhythm regularity and frequency."""
        if len(speed) < 10:
            return 0.0, 0.0

        # Autocorrelation
        speed_centered = speed - np.mean(speed)
        autocorr = np.correlate(speed_centered, speed_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-6)

        # Find first major peak
        peaks, _ = signal.find_peaks(autocorr, height=0.1)

        if len(peaks) == 0:
            return 0.0, 0.0

        first_peak = peaks[0]
        regularity = autocorr[first_peak]
        frequency = fps / first_peak

        return regularity, frequency

    def _compute_score(
        self,
        speed_dec: float,
        amp_dec: float,
        hesitation: float,
        regularity: float,
    ) -> float:
        """Compute UPDRS-like bradykinesia score (0-4)."""
        # Higher decrements and hesitation = worse
        # Lower regularity = worse
        score = 0.0

        # Speed decrement contribution (0-1)
        score += min(1, max(0, speed_dec)) * 1.0

        # Amplitude decrement (0-1)
        score += min(1, max(0, amp_dec)) * 1.0

        # Hesitation (0-1)
        score += min(1, hesitation) * 1.0

        # Irregularity (0-1)
        score += (1 - min(1, max(0, regularity))) * 1.0

        return score

    def _empty_features(self) -> BradykinesiaFeatures:
        """Return empty features."""
        return BradykinesiaFeatures(
            mean_speed=0,
            speed_decrement=0,
            speed_variability=0,
            mean_amplitude=0,
            amplitude_decrement=0,
            amplitude_variability=0,
            hesitation_ratio=0,
            pause_count=0,
            mean_pause_duration=0,
            rhythm_regularity=0,
            movement_frequency=0,
            bradykinesia_score=0,
        )


def analyze_bradykinesia(
    keypoints: np.ndarray,
    fps: float,
    task: str = "gait",
) -> BradykinesiaFeatures:
    """
    Convenience function to analyze bradykinesia.

    Args:
        keypoints: (T, N, 2 or 3) keypoint array
        fps: Frames per second
        task: "gait" or "finger_tap"

    Returns:
        BradykinesiaFeatures
    """
    analyzer = BradykinesiaAnalyzer()

    if task == "finger_tap":
        return analyzer.analyze_finger_tap(keypoints, fps)
    else:
        return analyzer.analyze_gait(keypoints, fps)
