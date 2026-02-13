"""
Smart Gait Analyzer with Walking Detection and Face Recognition
===============================================================
Automatically detects walking segments and identifies users.
Calibrated for home video analysis.
"""

import os
import json
import uuid
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import mediapipe as mp
from scipy.ndimage import uniform_filter1d
from scipy import signal
from scipy import stats as scipy_stats


@dataclass
class SegmentStatistics:
    """Statistical summary for a single gait parameter across segments."""
    parameter: str
    n_segments: int
    mean: float
    std: float
    cv: float  # Coefficient of variation (%)
    min_val: float
    max_val: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    values: List[float] = None  # Individual segment values

    def to_dict(self):
        return _convert_numpy({
            'parameter': self.parameter,
            'n_segments': self.n_segments,
            'mean': round(self.mean, 4),
            'std': round(self.std, 4),
            'cv': round(self.cv, 2),
            'min': round(self.min_val, 4),
            'max': round(self.max_val, 4),
            'ci_95': [round(self.ci_lower, 4), round(self.ci_upper, 4)],
            'values': [round(v, 4) for v in self.values] if self.values else None
        })


@dataclass
class StatisticalSummary:
    """Comprehensive statistical summary across all walking segments."""
    n_segments_total: int
    n_segments_analyzed: int
    min_segment_duration: float
    total_walking_time: float
    parameters: Dict[str, SegmentStatistics]
    pd_assessment: Dict
    reliability_score: float  # Intra-session reliability (0-1)
    analysis_quality: str  # "Excellent", "Good", "Fair", "Poor"

    def to_dict(self):
        return _convert_numpy({
            'n_segments_total': self.n_segments_total,
            'n_segments_analyzed': self.n_segments_analyzed,
            'min_segment_duration_sec': self.min_segment_duration,
            'total_walking_time_sec': round(self.total_walking_time, 1),
            'parameters': {k: v.to_dict() for k, v in self.parameters.items()},
            'pd_assessment': self.pd_assessment,
            'reliability_score': round(self.reliability_score, 3),
            'analysis_quality': self.analysis_quality
        })


@dataclass
class UserProfile:
    """User profile with face embedding."""
    user_id: str
    name: str
    height_cm: float = 170.0  # User's height for calibration
    face_embedding: Optional[np.ndarray] = None
    registered_at: str = ""

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'name': self.name,
            'height_cm': self.height_cm,
            'registered_at': self.registered_at
        }


@dataclass
class WalkingSegment:
    """Detected walking segment."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    user_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class GaitAnalysisResult:
    """Result of gait analysis for a segment."""
    segment_id: str
    user_id: Optional[str]
    user_name: Optional[str]
    start_time: float
    end_time: float
    duration: float
    walking_speed: float
    stride_length: float
    cadence: float
    step_width: float
    asymmetry: float
    stability_score: float
    classification: str
    risk_level: str
    confidence: float
    percentile: int
    # PD-specific biomarkers
    stride_time_cv: float = 0.0
    arm_swing_asymmetry: float = 0.0
    arm_swing_amplitude: float = 0.0
    step_time_asymmetry: float = 0.0
    pd_risk_score: float = 0.0
    pd_risk_level: str = "Unknown"

    def to_dict(self):
        return _convert_numpy(asdict(self))


class FaceRecognizer:
    """Simple face recognition using OpenCV."""

    def __init__(self, profiles_dir: Path):
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, UserProfile] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._load_profiles()
        print("Using OpenCV face detection for user identification")

    def _load_profiles(self):
        """Load saved user profiles."""
        profiles_file = self.profiles_dir / 'profiles.json'
        if profiles_file.exists():
            with open(profiles_file) as f:
                data = json.load(f)
                for user_data in data.get('users', []):
                    user = UserProfile(**user_data)
                    self.users[user.user_id] = user
                    emb_file = self.profiles_dir / f"{user.user_id}_embedding.npy"
                    if emb_file.exists():
                        self.embeddings[user.user_id] = np.load(emb_file)

    def _save_profiles(self):
        """Save user profiles."""
        profiles_file = self.profiles_dir / 'profiles.json'
        data = {'users': [u.to_dict() for u in self.users.values()]}
        with open(profiles_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_user(self, name: str, face_image: np.ndarray, height_cm: float = 170.0) -> Optional[UserProfile]:
        """Register a new user with their face."""
        user_id = str(uuid.uuid4())[:8]
        embedding = self._extract_embedding(face_image)
        if embedding is None:
            return None

        from datetime import datetime
        profile = UserProfile(
            user_id=user_id,
            name=name,
            height_cm=height_cm,
            registered_at=datetime.now().isoformat()
        )

        self.users[user_id] = profile
        self.embeddings[user_id] = embedding
        np.save(self.profiles_dir / f"{user_id}_embedding.npy", embedding)
        self._save_profiles()
        cv2.imwrite(str(self.profiles_dir / f"{user_id}_face.jpg"), face_image)
        return profile

    def _extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (64, 64))
            return face_resized.flatten().astype(float) / 255.0
        return None

    def identify(self, face_image: np.ndarray, threshold: float = 0.85) -> Tuple[Optional[str], float]:
        """Identify a face against registered users."""
        if not self.embeddings:
            return None, 0.0

        query_embedding = self._extract_embedding(face_image)
        if query_embedding is None:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for user_id, stored_embedding in self.embeddings.items():
            if len(query_embedding) == len(stored_embedding):
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding) + 1e-6
                )
                if similarity > best_score:
                    best_score = similarity
                    best_match = user_id

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) > 0:
            areas = [w * h for (x, y, w, h) in faces]
            idx = np.argmax(areas)
            return tuple(faces[idx])
        return None

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        return self.users.get(user_id)

    def list_users(self) -> List[Dict]:
        return [u.to_dict() for u in self.users.values()]


class WalkingDetector:
    """Detect walking segments from pose sequences."""

    ANKLE_LEFT = 27
    ANKLE_RIGHT = 28
    HIP_LEFT = 23
    HIP_RIGHT = 24
    KNEE_LEFT = 25
    KNEE_RIGHT = 26

    def __init__(self, min_duration: float = 1.0, confidence_threshold: float = 0.35):
        self.min_duration = min_duration
        self.confidence_threshold = confidence_threshold

    def detect(self, keypoints: np.ndarray, fps: float) -> Tuple[List[WalkingSegment], np.ndarray]:
        """Detect walking segments in pose sequence."""
        T = len(keypoints)
        if T < int(fps * 1.0):
            return [], np.zeros(T, dtype=bool)

        ankle_left = keypoints[:, self.ANKLE_LEFT, :2]
        ankle_right = keypoints[:, self.ANKLE_RIGHT, :2]
        hip_center = (keypoints[:, self.HIP_LEFT, :2] + keypoints[:, self.HIP_RIGHT, :2]) / 2
        knee_left = keypoints[:, self.KNEE_LEFT, :2]
        knee_right = keypoints[:, self.KNEE_RIGHT, :2]

        # Compute scores
        movement_score = self._compute_movement_score(hip_center, fps)
        rhythm_score = self._compute_rhythm_score(ankle_left, ankle_right, fps)
        leg_motion_score = self._compute_leg_motion_score(knee_left, knee_right, ankle_left, ankle_right, fps)

        # Combine
        walking_confidence = 0.4 * movement_score + 0.3 * rhythm_score + 0.3 * leg_motion_score

        # Smooth
        window = max(3, int(0.5 * fps))
        walking_confidence = uniform_filter1d(walking_confidence, window)

        # Threshold
        is_walking = walking_confidence > self.confidence_threshold

        # Filter short segments
        is_walking = self._filter_short_segments(is_walking, fps)

        # Extract segments
        segments = self._extract_segments(is_walking, walking_confidence, fps)

        return segments, is_walking

    def _compute_movement_score(self, hip_center, fps):
        """Compute overall body movement score."""
        T = len(hip_center)
        displacement = np.sqrt(np.sum(np.diff(hip_center, axis=0)**2, axis=1))
        displacement = np.concatenate([[0], displacement])

        # Smooth and normalize
        smoothed = uniform_filter1d(displacement, max(3, int(fps * 0.3)))
        normalized = np.clip(smoothed / np.percentile(smoothed[smoothed > 0] if np.any(smoothed > 0) else [1], 75), 0, 1)
        return normalized

    def _compute_rhythm_score(self, ankle_left, ankle_right, fps):
        """Compute leg alternation rhythm score."""
        T = len(ankle_left)
        leg_diff = ankle_left[:, 0] - ankle_right[:, 0]

        window = int(fps * 1.5)
        rhythm_score = np.zeros(T)

        for i in range(window, T - window):
            segment = leg_diff[i-window:i+window]
            centered = segment - np.mean(segment)

            # Count sign changes (leg alternations)
            sign_changes = np.sum(np.abs(np.diff(np.sign(centered))) > 0)

            # Expect ~2-4 changes per second for walking
            expected = (2 * window / fps) * 3
            rhythm_score[i] = min(1.0, sign_changes / expected)

        return rhythm_score

    def _compute_leg_motion_score(self, knee_left, knee_right, ankle_left, ankle_right, fps):
        """Compute leg joint motion score."""
        T = len(knee_left)

        # Knee vertical movement
        knee_motion = np.abs(np.diff(knee_left[:, 1])) + np.abs(np.diff(knee_right[:, 1]))
        knee_motion = np.concatenate([[0], knee_motion])

        # Ankle movement
        ankle_motion = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1))
        ankle_motion += np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1))
        ankle_motion = np.concatenate([[0], ankle_motion])

        combined = knee_motion + ankle_motion
        if np.max(combined) > 0:
            normalized = combined / np.percentile(combined[combined > 0], 90)
        else:
            normalized = combined

        return np.clip(normalized, 0, 1)

    def _filter_short_segments(self, is_walking, fps):
        """Remove segments shorter than minimum duration."""
        min_frames = int(self.min_duration * fps)
        result = is_walking.copy()

        changes = np.diff(np.concatenate([[0], is_walking.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_frames:
                result[start:end] = False

        return result

    def _extract_segments(self, is_walking, confidence, fps):
        """Extract walking segments."""
        segments = []
        changes = np.diff(np.concatenate([[0], is_walking.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            segments.append(WalkingSegment(
                start_frame=int(start),
                end_frame=int(end),
                start_time=start / fps,
                end_time=end / fps,
                confidence=float(np.mean(confidence[start:end]))
            ))

        return segments


class GaitParameterEstimator:
    """Estimate gait parameters from pose sequences with data-driven calibration.

    PD detection based on calibration with 10 normal + 10 Parkinson's videos.

    Key findings from calibration:
    - Walking speed and stride length are the BEST discriminators (AUC=0.96)
    - Lower speed/stride = higher PD risk (opposite to some literature claims)
    - Thresholds optimized for HIGH SPECIFICITY (no false positives on normal)

    Calibration results (2024-02):
    - Normal: speed=0.95±0.25 m/s, stride=1.03±0.27 m
    - PD: speed=0.47±0.13 m/s, stride=0.50±0.14 m
    """

    # Reference values from CALIBRATION DATA (10 normal + 10 PD videos)
    REFERENCE = {
        'healthy': {
            'speed': 0.95, 'cadence': 112, 'stride_length': 1.03, 'stride_width': 0.064,
            'asymmetry': 0.089, 'stride_time_cv': 10.3, 'arm_swing_asymmetry': 0.20
        },
        'pd_on': {
            'speed': 0.60, 'cadence': 112, 'stride_length': 0.65, 'stride_width': 0.045,
            'asymmetry': 0.050, 'stride_time_cv': 7.0, 'arm_swing_asymmetry': 0.16
        },
        'pd_off': {
            'speed': 0.47, 'cadence': 112, 'stride_length': 0.50, 'stride_width': 0.040,
            'asymmetry': 0.047, 'stride_time_cv': 7.0, 'arm_swing_asymmetry': 0.16
        },
    }

    # DATA-DRIVEN THRESHOLDS (optimized for high specificity)
    # Calibrated from 10 normal + 10 PD videos
    # Direction: "lower" means values BELOW threshold indicate PD risk
    PD_THRESHOLDS = {
        # Primary indicators (AUC > 0.95) - MOST RELIABLE
        'walking_speed': {'threshold': 0.55, 'direction': 'lower', 'auc': 0.96, 'weight': 0.35},
        'stride_length': {'threshold': 0.60, 'direction': 'lower', 'auc': 0.96, 'weight': 0.35},
        # Secondary indicators (AUC 0.65-0.75)
        'asymmetry': {'threshold': 0.05, 'direction': 'lower', 'auc': 0.74, 'weight': 0.15},
        'step_width': {'threshold': 0.05, 'direction': 'lower', 'auc': 0.68, 'weight': 0.15},
    }

    # MediaPipe pose landmarks
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Legacy aliases
    ANKLE_LEFT = 27
    ANKLE_RIGHT = 28
    HIP_LEFT = 23
    HIP_RIGHT = 24
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12

    def __init__(self, user_height_cm: float = 170.0):
        self.user_height_cm = user_height_cm

    def estimate(self, keypoints: np.ndarray, fps: float) -> Dict:
        """Estimate gait parameters from keypoints.

        Includes PD-specific biomarkers:
        - stride_time_cv: Coefficient of variation of stride timing (>2.6% abnormal)
        - arm_swing_asymmetry: Asymmetry in arm swing amplitude (key early PD marker)
        - step_time_asymmetry: Asymmetry in step timing (most predictive for PD)
        """
        T = len(keypoints)
        if T < 10:
            return self._empty_result()

        # Calculate body scale from pose
        pixels_per_meter = self._estimate_scale(keypoints)

        # Extract joints - legs
        ankle_left = keypoints[:, self.LEFT_ANKLE, :2]
        ankle_right = keypoints[:, self.RIGHT_ANKLE, :2]
        hip_center = (keypoints[:, self.LEFT_HIP, :2] + keypoints[:, self.RIGHT_HIP, :2]) / 2

        # Extract joints - arms (for PD-specific analysis)
        wrist_left = keypoints[:, self.LEFT_WRIST, :2]
        wrist_right = keypoints[:, self.RIGHT_WRIST, :2]
        elbow_left = keypoints[:, self.LEFT_ELBOW, :2]
        elbow_right = keypoints[:, self.RIGHT_ELBOW, :2]
        shoulder_left = keypoints[:, self.LEFT_SHOULDER, :2]
        shoulder_right = keypoints[:, self.RIGHT_SHOULDER, :2]

        # PRIMARY: Stride length from leg swing amplitude
        stride_length = self._estimate_stride_length(ankle_left, ankle_right, pixels_per_meter)

        # DERIVED: Walking speed using empirical relationship
        speed = stride_length * 0.93
        speed = np.clip(speed, 0.2, 2.0)

        # DERIVED: Cadence from speed and stride length
        cadence = (speed / stride_length * 120) if stride_length > 0 else 105
        cadence = np.clip(cadence, 80, 130)

        # Step width
        step_width = self._estimate_step_width(ankle_left, ankle_right, pixels_per_meter)

        # ============ PD-SPECIFIC BIOMARKERS ============

        # 1. STRIDE TIME VARIABILITY (CV) - Most important PD marker
        # >2.6% is abnormal (PMC7349580)
        stride_time_cv = self._estimate_stride_time_variability(ankle_left, ankle_right, fps)

        # 2. ARM SWING ASYMMETRY - Early PD detection marker
        # Reduced amplitude and increased asymmetry in PD
        arm_swing_asymmetry, arm_swing_amplitude = self._estimate_arm_swing(
            wrist_left, wrist_right, shoulder_left, shoulder_right, pixels_per_meter
        )

        # 3. STEP TIME ASYMMETRY - "Most relevant factor" for PD diagnosis
        step_time_asymmetry = self._estimate_step_time_asymmetry(ankle_left, ankle_right, fps)

        # Legacy asymmetry (leg-based)
        leg_asymmetry = self._estimate_asymmetry(ankle_left, ankle_right, fps)

        # Stability from gait regularity
        stability = self._estimate_stability_from_cadence(ankle_left, ankle_right, fps)

        # ============ PD RISK SCORE (DATA-DRIVEN) ============
        pd_indicators = self._calculate_pd_indicators(
            stride_time_cv, arm_swing_asymmetry, step_time_asymmetry, speed,
            stride_length=stride_length, asymmetry=leg_asymmetry, step_width=step_width
        )

        return {
            'walking_speed': round(speed, 3),
            'stride_length': round(stride_length, 3),
            'cadence': round(cadence, 1),
            'step_width': round(step_width, 3),
            'asymmetry': round(leg_asymmetry, 3),
            'stability': round(stability, 3),
            'duration': T / fps,
            # PD-specific biomarkers
            'stride_time_cv': round(stride_time_cv, 2),
            'arm_swing_asymmetry': round(arm_swing_asymmetry, 3),
            'arm_swing_amplitude': round(arm_swing_amplitude, 1),
            'step_time_asymmetry': round(step_time_asymmetry, 3),
            'pd_indicators': pd_indicators,
        }

    def _estimate_scale(self, keypoints: np.ndarray) -> float:
        """Estimate pixels per meter from body proportions."""
        # Use shoulder-to-hip distance as reference
        shoulder_center = (keypoints[:, self.SHOULDER_LEFT, :2] + keypoints[:, self.SHOULDER_RIGHT, :2]) / 2
        hip_center = (keypoints[:, self.HIP_LEFT, :2] + keypoints[:, self.HIP_RIGHT, :2]) / 2

        torso_length_pixels = np.median(np.sqrt(np.sum((shoulder_center - hip_center)**2, axis=1)))

        # Torso is approximately 30% of total height
        torso_length_meters = self.user_height_cm / 100 * 0.30

        pixels_per_meter = torso_length_pixels / torso_length_meters if torso_length_meters > 0 else 100

        return max(pixels_per_meter, 50)  # Minimum reasonable scale

    def _estimate_stride_length(self, ankle_left: np.ndarray, ankle_right: np.ndarray,
                                  scale: float) -> float:
        """Estimate stride length from leg swing amplitude.

        In walking, the horizontal ankle movement during a gait cycle
        approximates the step length. Stride = 2 × step length.
        """
        # Calculate horizontal ankle positions relative to body center
        ankle_mid = (ankle_left[:, 0] + ankle_right[:, 0]) / 2

        # Detrend to remove drift
        ankle_left_x = ankle_left[:, 0] - ankle_mid
        ankle_right_x = ankle_right[:, 0] - ankle_mid

        # Get peak-to-peak amplitude for each ankle
        # This represents the swing range ≈ step length
        left_amplitude = np.percentile(ankle_left_x, 95) - np.percentile(ankle_left_x, 5)
        right_amplitude = np.percentile(ankle_right_x, 95) - np.percentile(ankle_right_x, 5)

        # Average step length in pixels
        step_length_pixels = (left_amplitude + right_amplitude) / 2

        # Convert to meters
        step_length_meters = step_length_pixels / scale

        # Stride = 2 × step length, clamped to reasonable range
        stride_length = step_length_meters * 2
        return np.clip(stride_length, 0.4, 2.0)

    def _estimate_cadence(self, ankle_left: np.ndarray, ankle_right: np.ndarray, fps: float) -> float:
        """Estimate cadence (steps per minute) using FFT frequency analysis.

        More robust than peak counting for noisy signals.
        """
        T = len(ankle_left)
        duration = T / fps

        if T < 30:
            return 105.0  # Default reasonable cadence for older adults

        # Use both X and Y components for better signal
        ankle_diff_x = ankle_left[:, 0] - ankle_right[:, 0]
        ankle_diff_y = ankle_left[:, 1] - ankle_right[:, 1]

        # Combined motion magnitude
        combined = np.sqrt(ankle_diff_x**2 + ankle_diff_y**2)

        # Detrend (remove slow drift)
        detrended = combined - uniform_filter1d(combined, max(3, int(fps * 2)))

        # FFT analysis
        from scipy.fft import fft, fftfreq
        n = len(detrended)
        yf = np.abs(fft(detrended))
        xf = fftfreq(n, 1/fps)

        # Look at positive frequencies in walking range
        # Stride frequency: 0.7 Hz to 1.2 Hz (42-72 strides/min = 84-144 steps/min)
        pos_mask = (xf > 0.7) & (xf < 1.2)
        xf_walk = xf[pos_mask]
        yf_walk = yf[pos_mask]

        if len(yf_walk) > 0 and np.max(yf_walk) > np.mean(yf[xf > 0]) * 2:
            # Found significant peak in walking frequency range
            peak_idx = np.argmax(yf_walk)
            stride_freq = xf_walk[peak_idx]
            cadence = stride_freq * 60 * 2  # stride freq to step freq
        else:
            # Fallback: estimate from stride length using empirical relationship
            # For comfortable walking: cadence ≈ 105-115 for healthy, 100-110 for PD
            # Use stride length to refine estimate
            cadence = 105.0  # Default for older adults

        # Clamp to reasonable range
        return np.clip(cadence, 80, 130)

    def _estimate_step_width(self, ankle_left: np.ndarray, ankle_right: np.ndarray, scale: float) -> float:
        """Estimate step width (lateral distance between feet)."""
        # Use y-coordinate difference (assuming side view, y is lateral)
        # Or x-coordinate if front view
        lateral_dist = np.abs(ankle_left[:, 1] - ankle_right[:, 1])

        # Also check x distance
        frontal_dist = np.abs(ankle_left[:, 0] - ankle_right[:, 0])

        # Use minimum of the two (more likely to be the lateral direction)
        width_pixels = np.median(np.minimum(lateral_dist, frontal_dist))
        width_meters = width_pixels / scale

        # Clamp to reasonable range (normal: 0.05-0.20 m)
        return np.clip(width_meters, 0.03, 0.30)

    def _estimate_asymmetry(self, ankle_left: np.ndarray, ankle_right: np.ndarray, fps: float) -> float:
        """Estimate gait asymmetry (0 = symmetric, 1 = very asymmetric)."""
        # Calculate velocity of each ankle
        vel_left = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1))
        vel_right = np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1))

        # RMS velocity for each leg
        rms_left = np.sqrt(np.mean(vel_left**2))
        rms_right = np.sqrt(np.mean(vel_right**2))

        # Asymmetry ratio
        if rms_left + rms_right > 0:
            asymmetry = np.abs(rms_left - rms_right) / (rms_left + rms_right)
        else:
            asymmetry = 0

        return np.clip(asymmetry, 0, 1)

    def _estimate_stability_from_cadence(self, ankle_left: np.ndarray,
                                          ankle_right: np.ndarray, fps: float) -> float:
        """Estimate gait stability from step timing regularity.

        Regular, consistent stepping indicates stable gait.
        """
        # Get ankle difference signal for step detection
        ankle_diff = ankle_left[:, 0] - ankle_right[:, 0]
        smoothed = uniform_filter1d(ankle_diff, max(3, int(fps * 0.1)))
        detrended = smoothed - uniform_filter1d(smoothed, max(3, int(fps)))

        if len(detrended) < 20:
            return 0.5

        # Find step events (zero crossings or peaks)
        peaks_pos, _ = signal.find_peaks(detrended, distance=int(fps * 0.3))
        peaks_neg, _ = signal.find_peaks(-detrended, distance=int(fps * 0.3))

        # Combine and sort all step events
        all_peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))

        if len(all_peaks) < 3:
            return 0.5

        # Calculate step intervals
        step_intervals = np.diff(all_peaks) / fps

        if len(step_intervals) < 2:
            return 0.5

        # Coefficient of variation of step timing
        mean_interval = np.mean(step_intervals)
        if mean_interval > 0:
            cv = np.std(step_intervals) / mean_interval
            # Lower CV = more stable gait
            stability = max(0, 1 - cv * 2)  # Scale CV for reasonable range
        else:
            stability = 0.5

        return np.clip(stability, 0, 1)

    def _estimate_stride_time_variability(self, ankle_left: np.ndarray,
                                           ankle_right: np.ndarray, fps: float) -> float:
        """Estimate stride time coefficient of variation (CV).

        Uses VERTICAL (Y) ankle movement for more robust step detection,
        as Y-axis movement is consistent regardless of camera angle.

        Higher CV indicates more irregular gait, common in PD.
        """
        T = len(ankle_left)
        if T < int(fps * 2):
            return 3.0

        # Use VERTICAL (Y) movement for step detection - more reliable
        # During walking, ankles move up and down with each step
        left_y = ankle_left[:, 1]
        right_y = ankle_right[:, 1]

        # Smooth to reduce noise
        window = max(3, int(fps * 0.1))
        left_smooth = uniform_filter1d(left_y, window)
        right_smooth = uniform_filter1d(right_y, window)

        # Detect step events using peaks in vertical movement
        # Left foot steps = local minima in left_y (foot on ground)
        # Right foot steps = local minima in right_y (foot on ground)

        # Find when each foot touches ground (local minima = peaks in inverted signal)
        min_distance = int(fps * 0.35)  # Minimum 0.35s between steps

        left_peaks, _ = signal.find_peaks(-left_smooth, distance=min_distance,
                                           prominence=np.std(left_smooth) * 0.3)
        right_peaks, _ = signal.find_peaks(-right_smooth, distance=min_distance,
                                            prominence=np.std(right_smooth) * 0.3)

        # Combine all step events
        all_steps = np.sort(np.concatenate([left_peaks, right_peaks]))

        if len(all_steps) < 4:
            # Fallback: use combined ankle velocity
            velocity = np.sqrt(np.diff(ankle_left, axis=0)**2 +
                              np.diff(ankle_right, axis=0)**2).sum(axis=1)
            velocity_smooth = uniform_filter1d(velocity, window)
            step_peaks, _ = signal.find_peaks(velocity_smooth, distance=min_distance,
                                               prominence=np.std(velocity_smooth) * 0.5)
            all_steps = step_peaks

        if len(all_steps) < 4:
            return 3.0

        # Calculate step intervals
        step_intervals = np.diff(all_steps) / fps

        # Filter out unrealistic intervals (< 0.3s or > 1.5s per step)
        valid_intervals = step_intervals[(step_intervals > 0.3) & (step_intervals < 1.5)]

        if len(valid_intervals) < 3:
            return 3.0

        # Calculate stride times (2 steps = 1 stride)
        # Group consecutive step pairs
        stride_times = []
        for i in range(0, len(valid_intervals) - 1, 2):
            stride_times.append(valid_intervals[i] + valid_intervals[i + 1])

        if len(stride_times) < 2:
            stride_times = valid_intervals  # Use step times if not enough strides

        # Coefficient of variation as percentage
        mean_stride = np.mean(stride_times)
        std_stride = np.std(stride_times, ddof=1) if len(stride_times) > 1 else 0

        if mean_stride > 0:
            cv = (std_stride / mean_stride) * 100
        else:
            cv = 3.0

        return np.clip(cv, 0.5, 20.0)

    def _estimate_arm_swing(self, wrist_left: np.ndarray, wrist_right: np.ndarray,
                            shoulder_left: np.ndarray, shoulder_right: np.ndarray,
                            scale: float) -> Tuple[float, float]:
        """Estimate arm swing asymmetry and amplitude.

        Key early PD marker - reduced amplitude and increased asymmetry in PD.
        Normal arm swing: ~7-10 degrees, PD: often <5 degrees

        Improved method: Uses 2D magnitude of arm movement for camera-angle robustness.
        """
        T = len(wrist_left)
        if T < 10:
            return 0.1, 5.0

        # Calculate wrist position relative to shoulder (arm swing)
        left_swing = wrist_left - shoulder_left
        right_swing = wrist_right - shoulder_right

        # Use 2D MAGNITUDE for camera-angle robustness
        # Arm swing creates movement in the sagittal plane (forward/backward)
        # We measure the total range of motion

        # Method 1: Range of motion in both X and Y
        left_range_x = np.percentile(left_swing[:, 0], 95) - np.percentile(left_swing[:, 0], 5)
        left_range_y = np.percentile(left_swing[:, 1], 95) - np.percentile(left_swing[:, 1], 5)
        right_range_x = np.percentile(right_swing[:, 0], 95) - np.percentile(right_swing[:, 0], 5)
        right_range_y = np.percentile(right_swing[:, 1], 95) - np.percentile(right_swing[:, 1], 5)

        # Use the larger range (more likely to be the swing direction)
        left_amplitude_pixels = max(left_range_x, left_range_y)
        right_amplitude_pixels = max(right_range_x, right_range_y)

        # Method 2: Velocity-based amplitude (more robust)
        left_velocity = np.sqrt(np.sum(np.diff(left_swing, axis=0)**2, axis=1))
        right_velocity = np.sqrt(np.sum(np.diff(right_swing, axis=0)**2, axis=1))

        # RMS velocity as measure of swing activity
        left_rms = np.sqrt(np.mean(left_velocity**2))
        right_rms = np.sqrt(np.mean(right_velocity**2))

        # Convert pixel amplitude to degrees
        arm_length_pixels = self.user_height_cm / 100 * 0.44 * scale
        if arm_length_pixels > 0:
            left_degrees = np.degrees(np.arcsin(np.clip(left_amplitude_pixels / (2 * arm_length_pixels), -1, 1)))
            right_degrees = np.degrees(np.arcsin(np.clip(right_amplitude_pixels / (2 * arm_length_pixels), -1, 1)))
        else:
            left_degrees = right_degrees = 5.0

        avg_amplitude = (left_degrees + right_degrees) / 2

        # ASYMMETRY: Compare left vs right swing activity
        # Use velocity-based metric for robustness
        total_rms = left_rms + right_rms
        if total_rms > 0:
            # Asymmetry based on velocity difference
            velocity_asymmetry = abs(left_rms - right_rms) / total_rms
        else:
            velocity_asymmetry = 0

        # Also compute amplitude-based asymmetry
        total_amplitude = left_degrees + right_degrees
        if total_amplitude > 0:
            amplitude_asymmetry = abs(left_degrees - right_degrees) / total_amplitude
        else:
            amplitude_asymmetry = 0

        # Combine both asymmetry measures (weighted average)
        asymmetry = 0.6 * velocity_asymmetry + 0.4 * amplitude_asymmetry

        return np.clip(asymmetry, 0, 1), np.clip(avg_amplitude, 0, 45)

    def _estimate_step_time_asymmetry(self, ankle_left: np.ndarray,
                                       ankle_right: np.ndarray, fps: float) -> float:
        """Estimate step time asymmetry.

        "Most relevant factor" for PD diagnosis (PMC7349580).
        Measures timing difference between left and right steps.
        """
        # Detect step events for each leg separately
        def get_step_times(ankle_x):
            velocity = np.abs(np.diff(ankle_x))
            smoothed = uniform_filter1d(velocity, max(3, int(fps * 0.1)))
            peaks, _ = signal.find_peaks(smoothed, distance=int(fps * 0.4),
                                         prominence=np.std(smoothed) * 0.5)
            return peaks / fps if len(peaks) > 0 else np.array([])

        left_steps = get_step_times(ankle_left[:, 0])
        right_steps = get_step_times(ankle_right[:, 0])

        if len(left_steps) < 2 or len(right_steps) < 2:
            return 0.05  # Default small asymmetry

        # Calculate mean step duration for each leg
        left_intervals = np.diff(left_steps)
        right_intervals = np.diff(right_steps)

        if len(left_intervals) == 0 or len(right_intervals) == 0:
            return 0.05

        mean_left = np.mean(left_intervals)
        mean_right = np.mean(right_intervals)

        # Asymmetry ratio
        total = mean_left + mean_right
        if total > 0:
            asymmetry = abs(mean_left - mean_right) / total
        else:
            asymmetry = 0

        return np.clip(asymmetry, 0, 0.5)

    def _calculate_pd_indicators(self, stride_time_cv: float, arm_swing_asymmetry: float,
                                  step_time_asymmetry: float, speed: float,
                                  stride_length: float = 0, asymmetry: float = 0,
                                  step_width: float = 0) -> Dict:
        """Calculate PD indicator scores based on DATA-DRIVEN thresholds.

        Calibrated from 10 normal + 10 Parkinson's videos.
        Focus: HIGH SPECIFICITY (don't classify normal as PD).

        Key insight: Walking speed and stride length are the BEST discriminators.
        """
        indicators = {}
        weighted_score = 0.0
        total_weight = 0.0

        # 1. WALKING SPEED - Primary indicator (AUC=0.96)
        speed_thresh = self.PD_THRESHOLDS['walking_speed']
        is_abnormal = speed < speed_thresh['threshold']
        if is_abnormal:
            # Severity: how far below threshold (normalized)
            severity = min((speed_thresh['threshold'] - speed) / 0.3, 1.0)
        else:
            severity = 0.0

        indicators['walking_speed'] = {
            'value': speed,
            'threshold': speed_thresh['threshold'],
            'direction': 'lower',  # lower = PD risk
            'status': 'abnormal' if is_abnormal else 'normal',
            'severity': severity,
            'auc': speed_thresh['auc']
        }
        weighted_score += severity * speed_thresh['weight']
        total_weight += speed_thresh['weight']

        # 2. STRIDE LENGTH - Primary indicator (AUC=0.96)
        stride_thresh = self.PD_THRESHOLDS['stride_length']
        is_abnormal = stride_length < stride_thresh['threshold']
        if is_abnormal:
            severity = min((stride_thresh['threshold'] - stride_length) / 0.3, 1.0)
        else:
            severity = 0.0

        indicators['stride_length'] = {
            'value': stride_length,
            'threshold': stride_thresh['threshold'],
            'direction': 'lower',
            'status': 'abnormal' if is_abnormal else 'normal',
            'severity': severity,
            'auc': stride_thresh['auc']
        }
        weighted_score += severity * stride_thresh['weight']
        total_weight += stride_thresh['weight']

        # 3. ASYMMETRY - Secondary indicator (AUC=0.74)
        # NOTE: Lower asymmetry indicates PD in our data
        asym_thresh = self.PD_THRESHOLDS['asymmetry']
        is_abnormal = asymmetry < asym_thresh['threshold']
        if is_abnormal:
            severity = min((asym_thresh['threshold'] - asymmetry) / 0.04, 1.0)
        else:
            severity = 0.0

        indicators['asymmetry'] = {
            'value': asymmetry,
            'threshold': asym_thresh['threshold'],
            'direction': 'lower',
            'status': 'abnormal' if is_abnormal else 'normal',
            'severity': severity,
            'auc': asym_thresh['auc']
        }
        weighted_score += severity * asym_thresh['weight']
        total_weight += asym_thresh['weight']

        # 4. STEP WIDTH - Secondary indicator (AUC=0.68)
        width_thresh = self.PD_THRESHOLDS['step_width']
        is_abnormal = step_width < width_thresh['threshold']
        if is_abnormal:
            severity = min((width_thresh['threshold'] - step_width) / 0.02, 1.0)
        else:
            severity = 0.0

        indicators['step_width'] = {
            'value': step_width,
            'threshold': width_thresh['threshold'],
            'direction': 'lower',
            'status': 'abnormal' if is_abnormal else 'normal',
            'severity': severity,
            'auc': width_thresh['auc']
        }
        weighted_score += severity * width_thresh['weight']
        total_weight += width_thresh['weight']

        # Calculate overall risk score
        risk_score = weighted_score / total_weight if total_weight > 0 else 0

        # Count abnormal indicators (only primary ones for classification)
        primary_abnormal = sum(1 for key in ['walking_speed', 'stride_length']
                               if indicators.get(key, {}).get('status') == 'abnormal')
        secondary_abnormal = sum(1 for key in ['asymmetry', 'step_width']
                                 if indicators.get(key, {}).get('status') == 'abnormal')

        # Risk level determination (conservative - prioritize specificity)
        # Only classify as High risk if BOTH primary indicators are abnormal
        if primary_abnormal >= 2:
            risk_level = 'High'
        elif primary_abnormal == 1 and secondary_abnormal >= 1:
            risk_level = 'Moderate'
        elif primary_abnormal == 1 or (secondary_abnormal >= 2 and risk_score > 0.3):
            risk_level = 'Low-Moderate'
        else:
            risk_level = 'Low'

        indicators['overall'] = {
            'abnormal_count': primary_abnormal + secondary_abnormal,
            'primary_abnormal': primary_abnormal,
            'secondary_abnormal': secondary_abnormal,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level
        }

        return indicators

    def _empty_result(self):
        return {
            'walking_speed': 0, 'stride_length': 0, 'cadence': 0,
            'step_width': 0, 'asymmetry': 0, 'stability': 0, 'duration': 0,
            'stride_time_cv': 0, 'arm_swing_asymmetry': 0, 'arm_swing_amplitude': 0,
            'step_time_asymmetry': 0, 'pd_indicators': {}
        }

    def classify(self, params: Dict) -> Dict:
        """Classify gait based on DATA-DRIVEN biomarkers.

        Calibrated from 10 normal + 10 Parkinson's videos.
        Primary indicators (AUC=0.96): walking_speed, stride_length
        Secondary indicators: asymmetry, step_width

        Classification priority: HIGH SPECIFICITY (avoid false positives)
        """
        speed = params.get('walking_speed', 0)
        stride_length = params.get('stride_length', 0)
        pd_indicators = params.get('pd_indicators', {})

        healthy = self.REFERENCE['healthy']
        pd_on = self.REFERENCE['pd_on']
        pd_off = self.REFERENCE['pd_off']

        # Calculate z-score relative to calibrated healthy reference
        z_speed = (speed - healthy['speed']) / 0.25  # std from calibration data

        # Percentile (approximate)
        percentile = int(50 + z_speed * 34)
        percentile = max(0, min(100, percentile))

        # Get PD indicator results
        overall = pd_indicators.get('overall', {})
        primary_abnormal = overall.get('primary_abnormal', 0)
        risk_score = overall.get('risk_score', 0)
        risk_level_from_indicators = overall.get('risk_level', 'Low')

        # DATA-DRIVEN CLASSIFICATION
        # Key thresholds from calibration:
        # - Speed < 0.55 m/s AND Stride < 0.60 m → High PD risk (100% specificity)
        # - Speed < 0.55 m/s OR Stride < 0.60 m → Moderate PD risk

        speed_threshold = self.PD_THRESHOLDS['walking_speed']['threshold']
        stride_threshold = self.PD_THRESHOLDS['stride_length']['threshold']

        speed_low = speed < speed_threshold
        stride_low = stride_length < stride_threshold

        if speed_low and stride_low:
            # BOTH primary indicators abnormal → High confidence PD
            if speed < pd_off['speed']:
                classification = 'PD-like: Severe'
                risk = 'High'
            elif speed < pd_on['speed']:
                classification = 'PD-like: Moderate'
                risk = 'High'
            else:
                classification = 'PD-like: Mild'
                risk = 'Moderate'
        elif speed_low or stride_low:
            # ONE primary indicator abnormal → Possible PD
            if risk_score > 0.4:
                classification = 'Possible PD'
                risk = 'Moderate'
            else:
                classification = 'Reduced Mobility'
                risk = 'Low-Moderate'
        else:
            # No primary indicators abnormal → Normal or mild issue
            if speed >= healthy['speed'] * 0.85:
                classification = 'Normal'
                risk = 'Low'
            elif speed >= healthy['speed'] * 0.70:
                classification = 'Mild Reduction'
                risk = 'Low'
            else:
                classification = 'Reduced Mobility'
                risk = 'Low-Moderate'

        # Build indicator summary based on new thresholds
        speed_status = 'abnormal' if speed_low else 'normal'
        stride_status = 'abnormal' if stride_low else 'normal'

        return {
            'classification': classification,
            'risk_level': risk,
            'percentile': percentile,
            'pd_indicators_summary': {
                'walking_speed': speed_status,
                'stride_length': stride_status,
                'combined': 'abnormal' if (speed_low and stride_low) else 'normal',
            },
            'comparison': {
                'vs_healthy': round((speed / healthy['speed'] - 1) * 100, 1) if healthy['speed'] > 0 else 0,
                'vs_pd_on': round((speed / pd_on['speed'] - 1) * 100, 1) if pd_on['speed'] > 0 else 0,
                'vs_pd_off': round((speed / pd_off['speed'] - 1) * 100, 1) if pd_off['speed'] > 0 else 0,
            },
            'calibration_note': 'Data-driven thresholds from 10 normal + 10 PD videos'
        }


class GaitStatisticalAnalyzer:
    """Statistical analysis of gait parameters across multiple walking segments.

    Provides:
    - Per-parameter statistics (mean, SD, CV, 95% CI)
    - Intra-session reliability assessment
    - PD risk assessment based on aggregated data
    - Analysis quality scoring
    """

    # Parameters to analyze statistically
    GAIT_PARAMETERS = [
        'walking_speed', 'stride_length', 'cadence', 'step_width',
        'asymmetry', 'stability_score',
        'stride_time_cv', 'arm_swing_asymmetry', 'arm_swing_amplitude',
        'step_time_asymmetry', 'pd_risk_score'
    ]

    # Minimum requirements for reliable analysis
    MIN_SEGMENTS_EXCELLENT = 5
    MIN_SEGMENTS_GOOD = 3
    MIN_SEGMENTS_FAIR = 2
    MIN_WALKING_TIME_EXCELLENT = 120  # seconds
    MIN_WALKING_TIME_GOOD = 60
    MIN_WALKING_TIME_FAIR = 30

    def __init__(self, min_segment_duration: float = 1.0):
        """Initialize analyzer.

        Args:
            min_segment_duration: Minimum segment duration (seconds) for analysis.
                                  Segments shorter than this are excluded.
        """
        self.min_segment_duration = min_segment_duration

    def analyze(self, results: List[GaitAnalysisResult],
                total_segments: int) -> StatisticalSummary:
        """Perform statistical analysis across all qualifying segments.

        Args:
            results: List of GaitAnalysisResult from individual segments
            total_segments: Total number of detected walking segments (before filtering)

        Returns:
            StatisticalSummary with comprehensive statistics
        """
        # Filter segments by minimum duration
        valid_results = [r for r in results if r.duration >= self.min_segment_duration]

        if not valid_results:
            return self._empty_summary(total_segments)

        total_walking_time = sum(r.duration for r in valid_results)

        # Calculate statistics for each parameter
        param_stats = {}
        for param in self.GAIT_PARAMETERS:
            values = [getattr(r, param, 0) for r in valid_results]
            if all(v == 0 for v in values):
                continue
            param_stats[param] = self._calculate_statistics(param, values)

        # Calculate reliability score
        reliability = self._calculate_reliability(valid_results, param_stats)

        # Assess analysis quality
        quality = self._assess_quality(len(valid_results), total_walking_time)

        # Aggregate PD assessment
        pd_assessment = self._aggregate_pd_assessment(valid_results, param_stats)

        return StatisticalSummary(
            n_segments_total=total_segments,
            n_segments_analyzed=len(valid_results),
            min_segment_duration=self.min_segment_duration,
            total_walking_time=total_walking_time,
            parameters=param_stats,
            pd_assessment=pd_assessment,
            reliability_score=reliability,
            analysis_quality=quality
        )

    def _calculate_statistics(self, param: str, values: List[float]) -> SegmentStatistics:
        """Calculate comprehensive statistics for a single parameter."""
        n = len(values)
        arr = np.array(values)

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # Coefficient of variation (%)
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0.0

        # 95% Confidence interval
        if n >= 2:
            se = std_val / np.sqrt(n)
            t_val = scipy_stats.t.ppf(0.975, n - 1)
            ci_lower = mean_val - t_val * se
            ci_upper = mean_val + t_val * se
        else:
            ci_lower = ci_upper = mean_val

        return SegmentStatistics(
            parameter=param,
            n_segments=n,
            mean=mean_val,
            std=std_val,
            cv=cv,
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            values=values
        )

    def _calculate_reliability(self, results: List[GaitAnalysisResult],
                                param_stats: Dict[str, SegmentStatistics]) -> float:
        """Calculate intra-session reliability score (0-1).

        Based on coefficient of variation of key parameters.
        Lower CV = higher reliability.
        """
        if len(results) < 2:
            return 0.5  # Cannot assess with single segment

        # Key parameters for reliability assessment
        key_params = ['walking_speed', 'stride_length', 'cadence']
        cvs = []

        for param in key_params:
            if param in param_stats:
                cvs.append(param_stats[param].cv)

        if not cvs:
            return 0.5

        avg_cv = np.mean(cvs)

        # Convert CV to reliability score (lower CV = higher reliability)
        # CV < 5% = excellent (0.9-1.0)
        # CV 5-10% = good (0.7-0.9)
        # CV 10-15% = fair (0.5-0.7)
        # CV > 15% = poor (<0.5)
        if avg_cv < 5:
            reliability = 0.9 + (5 - avg_cv) / 50  # 0.9-1.0
        elif avg_cv < 10:
            reliability = 0.7 + (10 - avg_cv) / 25  # 0.7-0.9
        elif avg_cv < 15:
            reliability = 0.5 + (15 - avg_cv) / 25  # 0.5-0.7
        else:
            reliability = max(0.2, 0.5 - (avg_cv - 15) / 50)  # 0.2-0.5

        return np.clip(reliability, 0, 1)

    def _assess_quality(self, n_segments: int, total_time: float) -> str:
        """Assess overall analysis quality."""
        if n_segments >= self.MIN_SEGMENTS_EXCELLENT and total_time >= self.MIN_WALKING_TIME_EXCELLENT:
            return "Excellent"
        elif n_segments >= self.MIN_SEGMENTS_GOOD and total_time >= self.MIN_WALKING_TIME_GOOD:
            return "Good"
        elif n_segments >= self.MIN_SEGMENTS_FAIR and total_time >= self.MIN_WALKING_TIME_FAIR:
            return "Fair"
        else:
            return "Poor"

    def _aggregate_pd_assessment(self, results: List[GaitAnalysisResult],
                                  param_stats: Dict[str, SegmentStatistics]) -> Dict:
        """Aggregate PD assessment across all segments with statistical rigor."""

        # Get mean values for PD indicators
        stride_cv_stats = param_stats.get('stride_time_cv')
        arm_asym_stats = param_stats.get('arm_swing_asymmetry')
        step_asym_stats = param_stats.get('step_time_asymmetry')
        arm_amp_stats = param_stats.get('arm_swing_amplitude')
        pd_risk_stats = param_stats.get('pd_risk_score')

        # Count how often each indicator is abnormal across segments
        indicators = {
            'stride_variability': {
                'threshold': 2.6,
                'mean': stride_cv_stats.mean if stride_cv_stats else 0,
                'std': stride_cv_stats.std if stride_cv_stats else 0,
                'ci_95': [stride_cv_stats.ci_lower, stride_cv_stats.ci_upper] if stride_cv_stats else [0, 0],
                'abnormal_count': sum(1 for r in results if r.stride_time_cv > 2.6),
                'abnormal_pct': sum(1 for r in results if r.stride_time_cv > 2.6) / len(results) * 100
            },
            'arm_swing': {
                'threshold': 0.10,
                'mean': arm_asym_stats.mean if arm_asym_stats else 0,
                'std': arm_asym_stats.std if arm_asym_stats else 0,
                'ci_95': [arm_asym_stats.ci_lower, arm_asym_stats.ci_upper] if arm_asym_stats else [0, 0],
                'abnormal_count': sum(1 for r in results if r.arm_swing_asymmetry > 0.10),
                'abnormal_pct': sum(1 for r in results if r.arm_swing_asymmetry > 0.10) / len(results) * 100
            },
            'step_timing': {
                'threshold': 0.05,
                'mean': step_asym_stats.mean if step_asym_stats else 0,
                'std': step_asym_stats.std if step_asym_stats else 0,
                'ci_95': [step_asym_stats.ci_lower, step_asym_stats.ci_upper] if step_asym_stats else [0, 0],
                'abnormal_count': sum(1 for r in results if r.step_time_asymmetry > 0.05),
                'abnormal_pct': sum(1 for r in results if r.step_time_asymmetry > 0.05) / len(results) * 100
            }
        }

        # Statistical significance test: is mean significantly above threshold?
        for key, ind in indicators.items():
            threshold = ind['threshold']
            mean = ind['mean']
            std = ind['std']
            n = len(results)

            if n >= 2 and std > 0:
                # One-sample t-test against threshold
                t_stat = (mean - threshold) / (std / np.sqrt(n))
                p_value = 1 - scipy_stats.t.cdf(t_stat, n - 1)  # One-tailed
                ind['t_statistic'] = round(float(t_stat), 3)
                ind['p_value'] = round(float(p_value), 4)
                ind['statistically_abnormal'] = bool(p_value < 0.05 and mean > threshold)
            else:
                ind['t_statistic'] = None
                ind['p_value'] = None
                ind['statistically_abnormal'] = bool(mean > threshold)

        # Overall PD risk assessment
        n_statistically_abnormal = sum(
            1 for ind in indicators.values()
            if ind.get('statistically_abnormal', False)
        )

        mean_risk_score = pd_risk_stats.mean if pd_risk_stats else 0

        # Risk level based on statistical evidence
        if n_statistically_abnormal >= 2 or mean_risk_score > 0.5:
            overall_risk_level = "High"
            risk_confidence = "Strong evidence of PD-like gait patterns"
        elif n_statistically_abnormal == 1 or mean_risk_score > 0.25:
            overall_risk_level = "Moderate"
            risk_confidence = "Some indicators suggest possible impairment"
        else:
            overall_risk_level = "Low"
            risk_confidence = "Gait parameters within normal range"

        # Arm swing amplitude assessment
        arm_amplitude_assessment = "Unknown"
        if arm_amp_stats:
            if arm_amp_stats.mean < 5:
                arm_amplitude_assessment = "Reduced (PD-like)"
            elif arm_amp_stats.mean < 7:
                arm_amplitude_assessment = "Mildly reduced"
            else:
                arm_amplitude_assessment = "Normal"

        return _convert_numpy({
            'indicators': indicators,
            'arm_swing_amplitude': {
                'mean': arm_amp_stats.mean if arm_amp_stats else 0,
                'std': arm_amp_stats.std if arm_amp_stats else 0,
                'assessment': arm_amplitude_assessment
            },
            'overall': {
                'risk_score_mean': round(mean_risk_score, 3),
                'risk_score_std': round(pd_risk_stats.std, 3) if pd_risk_stats else 0,
                'risk_level': overall_risk_level,
                'n_abnormal_indicators': n_statistically_abnormal,
                'confidence': risk_confidence
            }
        })

    def _empty_summary(self, total_segments: int) -> StatisticalSummary:
        """Return empty summary when no valid segments."""
        return StatisticalSummary(
            n_segments_total=total_segments,
            n_segments_analyzed=0,
            min_segment_duration=self.min_segment_duration,
            total_walking_time=0,
            parameters={},
            pd_assessment={
                'indicators': {},
                'overall': {
                    'risk_level': 'Unknown',
                    'confidence': 'Insufficient data for analysis'
                }
            },
            reliability_score=0,
            analysis_quality="Insufficient Data"
        )


class SmartGaitAnalyzer:
    """Complete gait analysis system."""

    def __init__(self, data_dir: Path, min_segment_duration: float = 1.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.min_segment_duration = min_segment_duration

        self.face_recognizer = FaceRecognizer(self.data_dir / 'profiles')
        self.walking_detector = WalkingDetector()
        self.statistical_analyzer = GaitStatisticalAnalyzer(min_segment_duration)

        self.mp_pose = mp.solutions.pose
        self.pose = None

    def _init_pose(self):
        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def analyze_video(self, video_path: str, identify_user: bool = True,
                      user_height_cm: float = 170.0, progress_callback=None) -> Dict:
        """Analyze video with automatic walking detection and user identification."""
        self._init_pose()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Extract poses
        keypoints_list = []
        face_detections = []
        face_check_interval = max(1, int(fps * 2))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = np.array([
                    [lm.x * width, lm.y * height, lm.z]
                    for lm in results.pose_landmarks.landmark
                ])
                keypoints_list.append(landmarks)
            else:
                if keypoints_list:
                    keypoints_list.append(keypoints_list[-1])
                else:
                    keypoints_list.append(np.zeros((33, 3)))

            # Face detection
            if identify_user and frame_idx % face_check_interval == 0:
                face_bbox = self.face_recognizer.detect_face(frame)
                if face_bbox is not None:
                    x, y, w, h = face_bbox
                    pad = int(w * 0.2)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
                    face_detections.append((frame_idx, frame[y1:y2, x1:x2].copy()))

            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx / total_frames * 50)

        cap.release()
        keypoints = np.array(keypoints_list)

        # Walking detection
        segments, is_walking = self.walking_detector.detect(keypoints, fps)

        if progress_callback:
            progress_callback(60)

        # User identification
        identified_user = None
        user_confidence = 0.0
        effective_height = user_height_cm

        if identify_user and face_detections:
            user_votes = defaultdict(list)
            for _, face_img in face_detections:
                user_id, score = self.face_recognizer.identify(face_img)
                if user_id:
                    user_votes[user_id].append(score)

            if user_votes:
                best_user = max(user_votes.items(), key=lambda x: np.mean(x[1]))
                identified_user = best_user[0]
                user_confidence = np.mean(best_user[1])

                # Use registered user's height
                user_profile = self.face_recognizer.get_user(identified_user)
                if user_profile:
                    effective_height = user_profile.height_cm

        if progress_callback:
            progress_callback(70)

        # Analyze each walking segment
        estimator = GaitParameterEstimator(effective_height)
        analysis_results = []

        for i, segment in enumerate(segments):
            segment.user_id = identified_user
            seg_keypoints = keypoints[segment.start_frame:segment.end_frame]

            if len(seg_keypoints) < 10:
                continue

            params = estimator.estimate(seg_keypoints, fps)
            classification = estimator.classify(params)

            user_profile = self.face_recognizer.get_user(identified_user) if identified_user else None

            # Get PD indicator summary
            pd_indicators = params.get('pd_indicators', {})
            pd_overall = pd_indicators.get('overall', {})

            result = GaitAnalysisResult(
                segment_id=f"seg_{i+1}",
                user_id=identified_user,
                user_name=user_profile.name if user_profile else "Unknown",
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                walking_speed=params['walking_speed'],
                stride_length=params['stride_length'],
                cadence=params['cadence'],
                step_width=params['step_width'],
                asymmetry=params['asymmetry'],
                stability_score=params['stability'],
                classification=classification['classification'],
                risk_level=classification['risk_level'],
                confidence=segment.confidence,
                percentile=classification['percentile'],
                # PD-specific biomarkers
                stride_time_cv=params.get('stride_time_cv', 0),
                arm_swing_asymmetry=params.get('arm_swing_asymmetry', 0),
                arm_swing_amplitude=params.get('arm_swing_amplitude', 0),
                step_time_asymmetry=params.get('step_time_asymmetry', 0),
                pd_risk_score=pd_overall.get('risk_score', 0),
                pd_risk_level=pd_overall.get('risk_level', 'Unknown'),
            )
            analysis_results.append(result)

            if progress_callback:
                progress_callback(70 + (i + 1) / max(1, len(segments)) * 25)

        if progress_callback:
            progress_callback(95)

        # Perform statistical analysis across all segments
        statistical_summary = self.statistical_analyzer.analyze(
            analysis_results, len(segments)
        )

        if progress_callback:
            progress_callback(100)

        return _convert_numpy({
            'video_info': {
                'path': video_path,
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps,
                'resolution': f"{width}x{height}"
            },
            'user': {
                'identified': identified_user is not None,
                'user_id': identified_user,
                'user_name': self.face_recognizer.get_user(identified_user).name if identified_user else None,
                'confidence': user_confidence,
                'height_cm': effective_height
            },
            'walking_detection': {
                'total_segments': len(segments),
                'total_walking_time': sum(s.duration for s in segments),
                'walking_ratio': float(np.sum(is_walking) / len(is_walking)) if len(is_walking) > 0 else 0,
                'min_segment_duration_required': self.min_segment_duration,
                'segments': [
                    {'start_time': s.start_time, 'end_time': s.end_time,
                     'duration': s.duration, 'confidence': s.confidence,
                     'meets_min_duration': s.duration >= self.min_segment_duration}
                    for s in segments
                ]
            },
            'analysis_results': [r.to_dict() for r in analysis_results],
            'summary': self._compute_summary(analysis_results) if analysis_results else None,
            'statistical_analysis': statistical_summary.to_dict()
        })

    def _compute_summary(self, results: List[GaitAnalysisResult]) -> Dict:
        if not results:
            return None

        speeds = [r.walking_speed for r in results]
        cadences = [r.cadence for r in results]
        stride_lengths = [r.stride_length for r in results]
        step_widths = [r.step_width for r in results]

        # PD-specific biomarkers
        stride_time_cvs = [r.stride_time_cv for r in results]
        arm_swing_asyms = [r.arm_swing_asymmetry for r in results]
        arm_swing_amps = [r.arm_swing_amplitude for r in results]
        step_time_asyms = [r.step_time_asymmetry for r in results]
        pd_risk_scores = [r.pd_risk_score for r in results]

        return _convert_numpy({
            'n_segments': len(results),
            'total_duration': round(sum(r.duration for r in results), 1),
            # Basic gait parameters
            'avg_speed': round(float(np.mean(speeds)), 3),
            'speed_std': round(float(np.std(speeds)), 3),
            'avg_stride_length': round(float(np.mean(stride_lengths)), 3),
            'stride_length_std': round(float(np.std(stride_lengths)), 3),
            'avg_cadence': round(float(np.mean(cadences)), 1),
            'cadence_std': round(float(np.std(cadences)), 1),
            'avg_step_width': round(float(np.mean(step_widths)), 3),
            'avg_asymmetry': round(float(np.mean([r.asymmetry for r in results])), 3),
            'avg_stability': round(float(np.mean([r.stability_score for r in results])), 3),
            # PD-specific biomarkers (averaged)
            'avg_stride_time_cv': round(float(np.mean(stride_time_cvs)), 2),
            'stride_time_cv_std': round(float(np.std(stride_time_cvs)), 2),
            'avg_arm_swing_asymmetry': round(float(np.mean(arm_swing_asyms)), 3),
            'avg_arm_swing_amplitude': round(float(np.mean(arm_swing_amps)), 1),
            'avg_step_time_asymmetry': round(float(np.mean(step_time_asyms)), 3),
            'avg_pd_risk_score': round(float(np.mean(pd_risk_scores)), 3),
            'pd_risk_score_std': round(float(np.std(pd_risk_scores)), 3),
            # Classification
            'overall_classification': max(set([r.classification for r in results]),
                                          key=[r.classification for r in results].count),
            'overall_risk': max(set([r.risk_level for r in results]),
                               key=[r.risk_level for r in results].count),
            'overall_pd_risk_level': max(set([r.pd_risk_level for r in results]),
                                         key=[r.pd_risk_level for r in results].count),
            'avg_percentile': int(np.mean([r.percentile for r in results]))
        })

    def register_user(self, name: str, image: np.ndarray, height_cm: float = 170.0) -> Optional[Dict]:
        profile = self.face_recognizer.register_user(name, image, height_cm)
        return profile.to_dict() if profile else None

    def list_users(self) -> List[Dict]:
        return self.face_recognizer.list_users()


_analyzer = None

def get_analyzer(data_dir: str = None, min_segment_duration: float = 1.0) -> SmartGaitAnalyzer:
    """Get or create the singleton gait analyzer.

    Args:
        data_dir: Directory for storing user profiles and data
        min_segment_duration: Minimum walking segment duration (seconds) for analysis.
                              Segments shorter than this are excluded from statistical analysis.
                              Default: 1 second.

    Returns:
        SmartGaitAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        if data_dir is None:
            data_dir = Path(__file__).parent / 'data'
        _analyzer = SmartGaitAnalyzer(Path(data_dir), min_segment_duration)
    return _analyzer
