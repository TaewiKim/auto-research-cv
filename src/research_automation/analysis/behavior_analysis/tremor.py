"""
Tremor Detection and Analysis
=============================

Detect and quantify tremor from video-based pose data.
Supports rest tremor, postural tremor, and action tremor analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


class TremorType(Enum):
    """Types of tremor."""

    REST = "rest"
    POSTURAL = "postural"
    ACTION = "action"
    UNKNOWN = "unknown"


@dataclass
class TremorFeatures:
    """Tremor analysis features."""

    # Presence
    tremor_detected: bool
    tremor_type: TremorType

    # Frequency
    dominant_frequency: float  # Hz
    frequency_stability: float  # std of frequency over time

    # Amplitude
    amplitude_mean: float
    amplitude_std: float
    amplitude_max: float

    # Regularity
    regularity: float  # 0-1, how regular the tremor is

    # Asymmetry
    left_right_asymmetry: float

    # Duration
    tremor_percentage: float  # % of time with tremor

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "tremor_detected": float(self.tremor_detected),
            "tremor_type": self.tremor_type.value,
            "dominant_frequency": self.dominant_frequency,
            "frequency_stability": self.frequency_stability,
            "amplitude_mean": self.amplitude_mean,
            "amplitude_std": self.amplitude_std,
            "amplitude_max": self.amplitude_max,
            "regularity": self.regularity,
            "left_right_asymmetry": self.left_right_asymmetry,
            "tremor_percentage": self.tremor_percentage,
        }


class TremorDetector:
    """Detect and analyze tremor from pose sequences."""

    # PD tremor typically 4-6 Hz (rest), 5-8 Hz (postural)
    REST_TREMOR_RANGE = (3.0, 7.0)
    POSTURAL_TREMOR_RANGE = (4.0, 12.0)
    ACTION_TREMOR_RANGE = (5.0, 12.0)

    def __init__(
        self,
        min_frequency: float = 3.0,
        max_frequency: float = 12.0,
        amplitude_threshold: float = 0.01,
        window_size: int = 64,
    ):
        """
        Initialize tremor detector.

        Args:
            min_frequency: Minimum tremor frequency (Hz)
            max_frequency: Maximum tremor frequency (Hz)
            amplitude_threshold: Minimum amplitude to consider tremor
            window_size: FFT window size (frames)
        """
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.amplitude_threshold = amplitude_threshold
        self.window_size = window_size

    def analyze(
        self,
        keypoints: np.ndarray,
        fps: float,
        joint_indices: list[int] | None = None,
    ) -> TremorFeatures:
        """
        Analyze tremor from keypoint sequence.

        Args:
            keypoints: (T, N, 2 or 3) keypoint array
            fps: Frames per second
            joint_indices: Which joints to analyze (default: hands/wrists)

        Returns:
            TremorFeatures
        """
        if len(keypoints) < self.window_size:
            return self._empty_features()

        # Default to hand/wrist joints (MediaPipe indices)
        if joint_indices is None:
            joint_indices = [15, 16, 19, 20]  # wrists and index fingers

        # Extract position trajectories
        trajectories = []
        for idx in joint_indices:
            if idx < keypoints.shape[1]:
                trajectories.append(keypoints[:, idx, :2])  # x, y only

        if not trajectories:
            return self._empty_features()

        trajectories = np.array(trajectories)  # (J, T, 2)

        # Compute velocity (first derivative)
        velocities = np.diff(trajectories, axis=1) * fps

        # Analyze each joint
        joint_features = []
        for vel in velocities:
            features = self._analyze_signal(vel, fps)
            joint_features.append(features)

        # Aggregate features
        return self._aggregate_features(joint_features)

    def _analyze_signal(
        self,
        signal_2d: np.ndarray,
        fps: float,
    ) -> dict:
        """Analyze tremor in a 2D position/velocity signal."""
        # Combine x and y into magnitude
        magnitude = np.sqrt(signal_2d[:, 0] ** 2 + signal_2d[:, 1] ** 2)

        # Remove DC component
        magnitude = magnitude - np.mean(magnitude)

        # Apply bandpass filter
        nyquist = fps / 2
        low = self.min_frequency / nyquist
        high = min(self.max_frequency / nyquist, 0.99)

        if low >= high or low <= 0:
            return self._empty_signal_features()

        try:
            b, a = signal.butter(4, [low, high], btype="band")
            filtered = signal.filtfilt(b, a, magnitude)
        except ValueError:
            filtered = magnitude

        # FFT analysis
        n = len(filtered)
        if n < self.window_size:
            return self._empty_signal_features()

        # Use Welch's method for robust frequency estimation
        frequencies, psd = signal.welch(
            filtered,
            fs=fps,
            nperseg=min(self.window_size, n // 2),
        )

        # Find dominant frequency in tremor range
        mask = (frequencies >= self.min_frequency) & (frequencies <= self.max_frequency)
        if not np.any(mask):
            return self._empty_signal_features()

        tremor_psd = psd[mask]
        tremor_freqs = frequencies[mask]

        peak_idx = np.argmax(tremor_psd)
        dominant_freq = tremor_freqs[peak_idx]
        peak_power = tremor_psd[peak_idx]

        # Amplitude estimation
        amplitude = np.abs(filtered)
        amplitude_mean = np.mean(amplitude)
        amplitude_std = np.std(amplitude)
        amplitude_max = np.max(amplitude)

        # Detect if tremor present
        tremor_detected = (
            peak_power > np.median(psd) * 2 and
            amplitude_mean > self.amplitude_threshold
        )

        # Regularity (how periodic is the signal)
        if tremor_detected:
            # Autocorrelation-based regularity
            autocorr = np.correlate(filtered, filtered, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / autocorr[0]

            # Find first peak after zero
            expected_lag = int(fps / dominant_freq)
            search_range = int(expected_lag * 0.3)
            start = max(0, expected_lag - search_range)
            end = min(len(autocorr), expected_lag + search_range)

            if end > start:
                regularity = np.max(autocorr[start:end])
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        # Frequency stability (sliding window analysis)
        freq_stability = self._compute_frequency_stability(filtered, fps)

        # Tremor percentage
        window_size = int(fps * 0.5)  # 0.5 second windows
        if window_size > 0:
            n_windows = len(amplitude) // window_size
            if n_windows > 0:
                windows = amplitude[: n_windows * window_size].reshape(n_windows, window_size)
                window_means = np.mean(windows, axis=1)
                tremor_windows = window_means > self.amplitude_threshold
                tremor_percentage = np.mean(tremor_windows)
            else:
                tremor_percentage = float(amplitude_mean > self.amplitude_threshold)
        else:
            tremor_percentage = float(amplitude_mean > self.amplitude_threshold)

        return {
            "tremor_detected": tremor_detected,
            "dominant_frequency": dominant_freq,
            "frequency_stability": freq_stability,
            "amplitude_mean": amplitude_mean,
            "amplitude_std": amplitude_std,
            "amplitude_max": amplitude_max,
            "regularity": regularity,
            "tremor_percentage": tremor_percentage,
        }

    def _compute_frequency_stability(
        self,
        signal_data: np.ndarray,
        fps: float,
    ) -> float:
        """Compute frequency stability over sliding windows."""
        window_size = self.window_size
        step = window_size // 2

        if len(signal_data) < window_size * 2:
            return 0.0

        frequencies = []
        for i in range(0, len(signal_data) - window_size, step):
            window = signal_data[i:i + window_size]
            freqs, psd = signal.welch(window, fs=fps, nperseg=min(32, len(window)))

            mask = (freqs >= self.min_frequency) & (freqs <= self.max_frequency)
            if np.any(mask):
                peak_idx = np.argmax(psd[mask])
                frequencies.append(freqs[mask][peak_idx])

        if len(frequencies) < 2:
            return 0.0

        # Lower std = more stable
        stability = 1.0 / (1.0 + np.std(frequencies))
        return stability

    def _empty_signal_features(self) -> dict:
        """Return empty signal features."""
        return {
            "tremor_detected": False,
            "dominant_frequency": 0.0,
            "frequency_stability": 0.0,
            "amplitude_mean": 0.0,
            "amplitude_std": 0.0,
            "amplitude_max": 0.0,
            "regularity": 0.0,
            "tremor_percentage": 0.0,
        }

    def _empty_features(self) -> TremorFeatures:
        """Return empty features."""
        return TremorFeatures(
            tremor_detected=False,
            tremor_type=TremorType.UNKNOWN,
            dominant_frequency=0.0,
            frequency_stability=0.0,
            amplitude_mean=0.0,
            amplitude_std=0.0,
            amplitude_max=0.0,
            regularity=0.0,
            left_right_asymmetry=0.0,
            tremor_percentage=0.0,
        )

    def _aggregate_features(self, joint_features: list[dict]) -> TremorFeatures:
        """Aggregate features across joints."""
        if not joint_features:
            return self._empty_features()

        # Check if any joint shows tremor
        tremor_detected = any(f["tremor_detected"] for f in joint_features)

        # Average metrics across joints with tremor
        active = [f for f in joint_features if f["tremor_detected"]]
        if not active:
            active = joint_features

        dominant_freq = np.mean([f["dominant_frequency"] for f in active])
        freq_stability = np.mean([f["frequency_stability"] for f in active])
        amplitude_mean = np.mean([f["amplitude_mean"] for f in active])
        amplitude_std = np.mean([f["amplitude_std"] for f in active])
        amplitude_max = np.max([f["amplitude_max"] for f in active])
        regularity = np.mean([f["regularity"] for f in active])
        tremor_pct = np.mean([f["tremor_percentage"] for f in active])

        # Determine tremor type from frequency
        tremor_type = self._classify_tremor_type(dominant_freq)

        # Compute asymmetry (assuming pairs of joints)
        if len(joint_features) >= 2:
            left_amp = np.mean([joint_features[i]["amplitude_mean"] for i in range(0, len(joint_features), 2)])
            right_amp = np.mean([joint_features[i]["amplitude_mean"] for i in range(1, len(joint_features), 2)])
            total_amp = left_amp + right_amp
            asymmetry = abs(left_amp - right_amp) / total_amp if total_amp > 0 else 0.0
        else:
            asymmetry = 0.0

        return TremorFeatures(
            tremor_detected=tremor_detected,
            tremor_type=tremor_type,
            dominant_frequency=dominant_freq,
            frequency_stability=freq_stability,
            amplitude_mean=amplitude_mean,
            amplitude_std=amplitude_std,
            amplitude_max=amplitude_max,
            regularity=regularity,
            left_right_asymmetry=asymmetry,
            tremor_percentage=tremor_pct,
        )

    def _classify_tremor_type(self, frequency: float) -> TremorType:
        """Classify tremor type based on frequency."""
        if 3.0 <= frequency <= 6.0:
            return TremorType.REST
        elif 4.0 <= frequency <= 8.0:
            return TremorType.POSTURAL
        elif frequency > 8.0:
            return TremorType.ACTION
        return TremorType.UNKNOWN


def detect_tremor(
    keypoints: np.ndarray,
    fps: float,
    joint_indices: list[int] | None = None,
) -> TremorFeatures:
    """
    Convenience function to detect tremor.

    Args:
        keypoints: (T, N, 2 or 3) keypoint array
        fps: Frames per second
        joint_indices: Which joints to analyze

    Returns:
        TremorFeatures
    """
    detector = TremorDetector()
    return detector.analyze(keypoints, fps, joint_indices)
