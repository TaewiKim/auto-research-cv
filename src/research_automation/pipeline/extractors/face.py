"""
Facial Feature Extraction from Video
=====================================

Extract facial landmarks, action units, and expression features.
Supports MediaPipe Face Mesh and OpenCV Haar cascades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
class FaceFrame:
    """Single frame face data."""

    frame_idx: int
    timestamp: float
    landmarks: np.ndarray | None  # (N, 3) - x, y, z normalized
    bbox: tuple[int, int, int, int] | None = None
    detected: bool = False


@dataclass
class FaceSequence:
    """Sequence of face data from video."""

    frames: list[FaceFrame]
    fps: float
    video_width: int
    video_height: int
    n_landmarks: int

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0

    @property
    def detection_rate(self) -> float:
        valid = sum(1 for f in self.frames if f.detected)
        return valid / len(self.frames) if self.frames else 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array (T, N, 3)."""
        valid_frames = [f for f in self.frames if f.landmarks is not None]
        if not valid_frames:
            return np.array([])
        return np.stack([f.landmarks for f in valid_frames])


# Key facial landmark indices for MediaPipe Face Mesh (468 landmarks)
FACE_LANDMARKS = {
    # Eyes
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "left_eye_upper": 159,
    "left_eye_lower": 145,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_eye_upper": 386,
    "right_eye_lower": 374,
    # Eyebrows
    "left_eyebrow_inner": 107,
    "left_eyebrow_outer": 70,
    "right_eyebrow_inner": 336,
    "right_eyebrow_outer": 300,
    # Nose
    "nose_tip": 1,
    "nose_left": 279,
    "nose_right": 49,
    # Mouth
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_upper": 13,
    "mouth_lower": 14,
    "upper_lip": 0,
    "lower_lip": 17,
    # Face contour
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
}

# Action Unit approximations using landmark distances
AU_DEFINITIONS = {
    "AU1": ("left_eyebrow_inner", "left_eye_upper"),  # Inner brow raiser
    "AU2": ("left_eyebrow_outer", "left_eye_outer"),  # Outer brow raiser
    "AU4": ("left_eyebrow_inner", "right_eyebrow_inner"),  # Brow lowerer
    "AU6": ("left_eye_lower", "left_cheek"),  # Cheek raiser
    "AU12": ("mouth_left", "left_cheek"),  # Lip corner puller (smile)
    "AU15": ("mouth_left", "chin"),  # Lip corner depressor
    "AU25": ("mouth_upper", "mouth_lower"),  # Lips part
    "AU26": ("upper_lip", "lower_lip"),  # Jaw drop
}


class FaceExtractor:
    """Extract facial features from video."""

    def __init__(
        self,
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._face_mesh = None
        self._haar_cascade = None

    @property
    def face_mesh(self):
        """Lazy-load MediaPipe Face Mesh."""
        if self._face_mesh is None and self.use_mediapipe and mp is not None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._face_mesh

    @property
    def haar_cascade(self):
        """Lazy-load OpenCV Haar cascade (fallback)."""
        if self._haar_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        return self._haar_cascade

    def extract_from_video(
        self,
        video_path: str | Path,
        sample_rate: int = 1,
        max_frames: int | None = None,
    ) -> FaceSequence:
        """Extract face sequence from video."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                landmarks, bbox, detected = self._extract_frame(frame)

                face_frame = FaceFrame(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    landmarks=landmarks,
                    bbox=bbox,
                    detected=detected,
                )
                frames.append(face_frame)
                processed += 1

                if max_frames and processed >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        n_landmarks = 468 if self.use_mediapipe else 0

        return FaceSequence(
            frames=frames,
            fps=fps / sample_rate,
            video_width=width,
            video_height=height,
            n_landmarks=n_landmarks,
        )

    def _extract_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, tuple | None, bool]:
        """Extract face from single frame."""
        h, w = frame.shape[:2]

        if self.use_mediapipe and self.face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                ])

                # Compute bounding box
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                x1, x2 = int(x_coords.min()), int(x_coords.max())
                y1, y2 = int(y_coords.min()), int(y_coords.max())
                bbox = (x1, y1, x2 - x1, y2 - y1)

                return landmarks, bbox, True

        # Fallback to Haar cascade (detection only, no landmarks)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            return None, (x, y, fw, fh), True

        return None, None, False

    def close(self):
        """Release resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class FacialFeatures:
    """Extracted facial features."""

    # Basic
    duration: float
    n_frames: int
    detection_rate: float

    # Facial Action Units (approximated)
    au_intensities: dict[str, float]

    # Symmetry
    eye_asymmetry: float
    mouth_asymmetry: float
    eyebrow_asymmetry: float
    overall_asymmetry: float

    # Movement
    eye_blink_rate: float
    mouth_movement: float
    eyebrow_movement: float

    # Expression dynamics
    expression_variability: float
    expression_range: float

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dictionary."""
        d = {
            "duration": self.duration,
            "n_frames": self.n_frames,
            "detection_rate": self.detection_rate,
            "eye_asymmetry": self.eye_asymmetry,
            "mouth_asymmetry": self.mouth_asymmetry,
            "eyebrow_asymmetry": self.eyebrow_asymmetry,
            "overall_asymmetry": self.overall_asymmetry,
            "eye_blink_rate": self.eye_blink_rate,
            "mouth_movement": self.mouth_movement,
            "eyebrow_movement": self.eyebrow_movement,
            "expression_variability": self.expression_variability,
            "expression_range": self.expression_range,
        }
        for au, intensity in self.au_intensities.items():
            d[f"au_{au.lower()}"] = intensity
        return d

    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array(list(self.to_dict().values()))


def compute_au_intensity(
    landmarks: np.ndarray,
    au_name: str,
    baseline: np.ndarray | None = None,
) -> float:
    """
    Compute approximate AU intensity from landmarks.

    Args:
        landmarks: (468, 3) face mesh landmarks
        au_name: Action unit name (e.g., "AU12")
        baseline: Optional neutral face landmarks for normalization

    Returns:
        Intensity value (0-1 normalized)
    """
    if au_name not in AU_DEFINITIONS:
        return 0.0

    lm1_name, lm2_name = AU_DEFINITIONS[au_name]
    idx1 = FACE_LANDMARKS.get(lm1_name)
    idx2 = FACE_LANDMARKS.get(lm2_name)

    if idx1 is None or idx2 is None:
        return 0.0

    dist = np.linalg.norm(landmarks[idx1, :2] - landmarks[idx2, :2])

    if baseline is not None:
        base_dist = np.linalg.norm(baseline[idx1, :2] - baseline[idx2, :2])
        if base_dist > 0:
            return (dist - base_dist) / base_dist

    # Normalize by face size (distance between eyes)
    left_eye = landmarks[FACE_LANDMARKS["left_eye_inner"], :2]
    right_eye = landmarks[FACE_LANDMARKS["right_eye_inner"], :2]
    eye_dist = np.linalg.norm(left_eye - right_eye)

    if eye_dist > 0:
        return dist / eye_dist

    return 0.0


def extract_facial_features(face_seq: FaceSequence) -> FacialFeatures:
    """
    Extract facial features from face sequence.

    Args:
        face_seq: FaceSequence from video

    Returns:
        FacialFeatures dataclass
    """
    arr = face_seq.to_array()  # (T, 468, 3)

    if len(arr) < 2:
        return FacialFeatures(
            duration=face_seq.duration,
            n_frames=len(face_seq.frames),
            detection_rate=face_seq.detection_rate,
            au_intensities={},
            eye_asymmetry=0, mouth_asymmetry=0,
            eyebrow_asymmetry=0, overall_asymmetry=0,
            eye_blink_rate=0, mouth_movement=0,
            eyebrow_movement=0, expression_variability=0,
            expression_range=0,
        )

    fps = face_seq.fps

    # Compute AU intensities (averaged over sequence)
    au_intensities = {}
    for au_name in AU_DEFINITIONS:
        intensities = [compute_au_intensity(arr[i], au_name) for i in range(len(arr))]
        au_intensities[au_name] = np.mean(intensities)

    # Asymmetry features
    def compute_asymmetry(left_idx: int, right_idx: int) -> float:
        left = arr[:, left_idx, :2]
        right = arr[:, right_idx, :2]
        # Mirror right side and compute difference
        center_x = (left[:, 0] + right[:, 0]) / 2
        left_dist = np.abs(left[:, 0] - center_x)
        right_dist = np.abs(right[:, 0] - center_x)
        return np.abs(left_dist - right_dist).mean()

    eye_asymmetry = compute_asymmetry(
        FACE_LANDMARKS["left_eye_inner"],
        FACE_LANDMARKS["right_eye_inner"]
    )
    mouth_asymmetry = compute_asymmetry(
        FACE_LANDMARKS["mouth_left"],
        FACE_LANDMARKS["mouth_right"]
    )
    eyebrow_asymmetry = compute_asymmetry(
        FACE_LANDMARKS["left_eyebrow_inner"],
        FACE_LANDMARKS["right_eyebrow_inner"]
    )
    overall_asymmetry = (eye_asymmetry + mouth_asymmetry + eyebrow_asymmetry) / 3

    # Eye blink detection (simplified: eye aspect ratio)
    left_eye_upper = arr[:, FACE_LANDMARKS["left_eye_upper"], 1]
    left_eye_lower = arr[:, FACE_LANDMARKS["left_eye_lower"], 1]
    eye_opening = np.abs(left_eye_upper - left_eye_lower)

    # Count blinks (local minima in eye opening)
    threshold = eye_opening.mean() * 0.5
    blinks = np.sum((eye_opening[1:-1] < threshold) &
                    (eye_opening[1:-1] < eye_opening[:-2]) &
                    (eye_opening[1:-1] < eye_opening[2:]))
    eye_blink_rate = blinks / face_seq.duration * 60 if face_seq.duration > 0 else 0

    # Movement features
    mouth_upper = arr[:, FACE_LANDMARKS["mouth_upper"], :2]
    mouth_lower = arr[:, FACE_LANDMARKS["mouth_lower"], :2]
    mouth_opening = np.linalg.norm(mouth_upper - mouth_lower, axis=1)
    mouth_movement = mouth_opening.std()

    eyebrow_inner_l = arr[:, FACE_LANDMARKS["left_eyebrow_inner"], 1]
    eyebrow_inner_r = arr[:, FACE_LANDMARKS["right_eyebrow_inner"], 1]
    eyebrow_movement = (eyebrow_inner_l.std() + eyebrow_inner_r.std()) / 2

    # Expression dynamics
    # Use landmark variability as proxy for expression changes
    landmark_std = arr.std(axis=0).mean()
    expression_variability = landmark_std

    landmark_range = (arr.max(axis=0) - arr.min(axis=0)).mean()
    expression_range = landmark_range

    return FacialFeatures(
        duration=face_seq.duration,
        n_frames=len(face_seq.frames),
        detection_rate=face_seq.detection_rate,
        au_intensities=au_intensities,
        eye_asymmetry=eye_asymmetry,
        mouth_asymmetry=mouth_asymmetry,
        eyebrow_asymmetry=eyebrow_asymmetry,
        overall_asymmetry=overall_asymmetry,
        eye_blink_rate=eye_blink_rate,
        mouth_movement=mouth_movement,
        eyebrow_movement=eyebrow_movement,
        expression_variability=expression_variability,
        expression_range=expression_range,
    )


def extract_face_from_video(
    video_path: str | Path,
    sample_rate: int = 1,
) -> tuple[FaceSequence, FacialFeatures]:
    """
    Convenience function to extract face and facial features.

    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame

    Returns:
        Tuple of (FaceSequence, FacialFeatures)
    """
    with FaceExtractor() as extractor:
        face_seq = extractor.extract_from_video(video_path, sample_rate)

    facial_features = extract_facial_features(face_seq)
    return face_seq, facial_features
