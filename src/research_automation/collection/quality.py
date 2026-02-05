"""Video quality assessment using MediaPipe (optional) and OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Check if MediaPipe is available
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


@dataclass
class QualityMetrics:
    """Quality metrics for a video."""

    # Basic info
    duration: float  # seconds
    fps: float
    resolution: tuple[int, int]  # width, height
    frame_count: int

    # Quality scores (0-1)
    face_detection_rate: float  # Percentage of frames with detected face
    pose_detection_rate: float  # Percentage of frames with detected pose
    brightness_score: float  # Average brightness quality
    blur_score: float  # Average sharpness (higher = sharper)
    stability_score: float  # Camera stability estimate

    # Issues
    issues: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        weights = {
            "face": 0.3,
            "pose": 0.25,
            "brightness": 0.15,
            "blur": 0.2,
            "stability": 0.1,
        }

        return (
            self.face_detection_rate * weights["face"]
            + self.pose_detection_rate * weights["pose"]
            + self.brightness_score * weights["brightness"]
            + self.blur_score * weights["blur"]
            + self.stability_score * weights["stability"]
        )

    @property
    def is_usable(self) -> bool:
        """Check if video meets minimum quality threshold."""
        return (
            self.overall_score >= 0.5
            and self.face_detection_rate >= 0.3
            and self.resolution[1] >= 360  # At least 360p
        )


class VideoQualityChecker:
    """Assess video quality for health monitoring research."""

    def __init__(self, sample_rate: int = 30, use_mediapipe: bool = True) -> None:
        """
        Initialize quality checker.

        Args:
            sample_rate: Check every Nth frame (for efficiency)
            use_mediapipe: Use MediaPipe for face/pose detection if available
        """
        self.sample_rate = sample_rate
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self._face_detector = None
        self._pose_detector = None
        self._opencv_face_cascade = None

    @property
    def face_detector(self):
        """Lazy-load face detector."""
        if not self.use_mediapipe:
            return None
        if self._face_detector is None and mp is not None:
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        return self._face_detector

    @property
    def pose_detector(self):
        """Lazy-load pose detector."""
        if not self.use_mediapipe:
            return None
        if self._pose_detector is None and mp is not None:
            self._pose_detector = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
        return self._pose_detector

    @property
    def opencv_face_cascade(self):
        """Lazy-load OpenCV face cascade (fallback)."""
        if self._opencv_face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._opencv_face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._opencv_face_cascade

    def _detect_face_opencv(self, gray_frame: np.ndarray) -> bool:
        """Detect face using OpenCV Haar cascade."""
        faces = self.opencv_face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces) > 0

    def check_video(self, video_path: str | Path) -> QualityMetrics:
        """Analyze video quality."""
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        # Analyze frames
        face_detections = 0
        pose_detections = 0
        brightness_scores: list[float] = []
        blur_scores: list[float] = []
        prev_frame = None
        motion_scores: list[float] = []
        frames_analyzed = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                frames_analyzed += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Face detection
                if self.use_mediapipe and self.face_detector is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_result = self.face_detector.process(rgb_frame)
                    if face_result.detections:
                        face_detections += 1
                else:
                    # Fallback to OpenCV
                    if self._detect_face_opencv(gray):
                        face_detections += 1

                # Pose detection (MediaPipe only)
                if self.use_mediapipe and self.pose_detector is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_result = self.pose_detector.process(rgb_frame)
                    if pose_result.pose_landmarks:
                        pose_detections += 1
                else:
                    # Without MediaPipe, assume pose detection based on face
                    # (rough approximation)
                    pass

                # Brightness
                brightness = np.mean(gray) / 255.0
                # Optimal brightness around 0.4-0.6
                brightness_quality = 1 - abs(brightness - 0.5) * 2
                brightness_scores.append(max(0, brightness_quality))

                # Blur detection (Laplacian variance)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                # Normalize to 0-1 (100+ variance is considered sharp)
                blur_score = min(1.0, laplacian_var / 100.0)
                blur_scores.append(blur_score)

                # Stability (motion between frames)
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion = np.mean(diff) / 255.0
                    # Lower motion = more stable (normalize)
                    stability = 1 - min(1.0, motion * 10)
                    motion_scores.append(stability)

                prev_frame = gray

            frame_idx += 1

        cap.release()

        # Calculate metrics
        face_rate = face_detections / frames_analyzed if frames_analyzed > 0 else 0
        pose_rate = pose_detections / frames_analyzed if frames_analyzed > 0 else 0
        avg_brightness = np.mean(brightness_scores) if brightness_scores else 0.5
        avg_blur = np.mean(blur_scores) if blur_scores else 0.5
        avg_stability = np.mean(motion_scores) if motion_scores else 0.5

        # Identify issues
        issues: list[str] = []
        if face_rate < 0.3:
            issues.append("Low face detection rate")
        if pose_rate < 0.3 and self.use_mediapipe:
            issues.append("Low pose detection rate")
        if avg_brightness < 0.3:
            issues.append("Video too dark")
        elif avg_brightness > 0.8:
            issues.append("Video overexposed")
        if avg_blur < 0.3:
            issues.append("Video too blurry")
        if avg_stability < 0.3:
            issues.append("Excessive camera shake")
        if height < 360:
            issues.append("Resolution too low")
        if fps < 15:
            issues.append("Frame rate too low")
        if not self.use_mediapipe:
            issues.append("MediaPipe not available (limited detection)")

        return QualityMetrics(
            duration=duration,
            fps=fps,
            resolution=(width, height),
            frame_count=frame_count,
            face_detection_rate=face_rate,
            pose_detection_rate=pose_rate,
            brightness_score=avg_brightness,
            blur_score=avg_blur,
            stability_score=avg_stability,
            issues=issues,
        )

    def check_directory(self, dir_path: str | Path) -> dict[str, QualityMetrics]:
        """Check all videos in a directory."""
        dir_path = Path(dir_path)
        results: dict[str, QualityMetrics] = {}

        extensions = ["*.mp4", "*.webm", "*.mkv", "*.avi", "*.mov"]
        video_files: list[Path] = []
        for ext in extensions:
            video_files.extend(dir_path.glob(ext))

        for video_file in video_files:
            try:
                metrics = self.check_video(video_file)
                results[str(video_file)] = metrics
            except Exception as e:
                print(f"Error checking {video_file}: {e}")

        return results

    def close(self) -> None:
        """Release resources."""
        if self._face_detector and hasattr(self._face_detector, "close"):
            self._face_detector.close()
        if self._pose_detector and hasattr(self._pose_detector, "close"):
            self._pose_detector.close()

    def __enter__(self) -> "VideoQualityChecker":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def check_video_quality(video_path: str | Path) -> QualityMetrics:
    """Convenience function to check a single video."""
    with VideoQualityChecker() as checker:
        return checker.check_video(video_path)


def format_quality_report(metrics: QualityMetrics) -> str:
    """Format quality metrics as a readable report."""
    lines = [
        "# Video Quality Report\n",
        "## Basic Info",
        f"- Duration: {metrics.duration:.1f}s",
        f"- Resolution: {metrics.resolution[0]}x{metrics.resolution[1]}",
        f"- FPS: {metrics.fps:.1f}",
        f"- Frames: {metrics.frame_count}",
        "",
        "## Quality Scores",
        f"- Face Detection: {metrics.face_detection_rate*100:.1f}%",
        f"- Pose Detection: {metrics.pose_detection_rate*100:.1f}%",
        f"- Brightness: {metrics.brightness_score*100:.1f}%",
        f"- Sharpness: {metrics.blur_score*100:.1f}%",
        f"- Stability: {metrics.stability_score*100:.1f}%",
        f"- **Overall: {metrics.overall_score*100:.1f}%**",
        "",
        f"## Usable for Research: {'Yes' if metrics.is_usable else 'No'}",
    ]

    if metrics.issues:
        lines.append("")
        lines.append("## Issues")
        for issue in metrics.issues:
            lines.append(f"- {issue}")

    return "\n".join(lines)


def is_mediapipe_available() -> bool:
    """Check if MediaPipe is available."""
    return MEDIAPIPE_AVAILABLE
