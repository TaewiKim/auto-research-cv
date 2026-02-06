"""Tests for pipeline extractors."""

from __future__ import annotations

import numpy as np
import pytest

from research_automation.pipeline.extractors import (
    AU_DEFINITIONS,
    FACE_LANDMARKS,
    GAIT_JOINTS,
    MEDIAPIPE_POSE_LANDMARKS,
    FaceExtractor,
    FaceFrame,
    FaceSequence,
    FacialFeatures,
    GaitFeatures,
    PoseExtractor,
    PoseFrame,
    PoseSequence,
    extract_facial_features,
    extract_gait_features,
)


class TestPoseExtractor:
    """Tests for PoseExtractor."""

    def test_pose_frame_creation(self):
        """Test PoseFrame dataclass."""
        keypoints = np.random.rand(33, 3).astype(np.float32)
        frame = PoseFrame(
            frame_idx=0,
            timestamp=0.0,
            keypoints=keypoints,
        )
        assert frame.frame_idx == 0
        assert frame.timestamp == 0.0
        assert frame.keypoints.shape == (33, 3)
        assert frame.bbox is None

    def test_pose_frame_with_bbox(self):
        """Test PoseFrame with bounding box."""
        frame = PoseFrame(
            frame_idx=1,
            timestamp=0.033,
            keypoints=np.random.rand(33, 3).astype(np.float32),
            bbox=(100, 50, 200, 400),
        )
        assert frame.bbox == (100, 50, 200, 400)

    def test_pose_sequence_empty(self):
        """Test empty PoseSequence."""
        seq = PoseSequence(
            frames=[],
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_keypoints=33,
        )
        assert len(seq.frames) == 0
        assert seq.duration == 0.0
        assert seq.detection_rate == 0

    def test_pose_sequence_with_frames(self):
        """Test PoseSequence with frames."""
        frames = [
            PoseFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                keypoints=np.random.rand(33, 3).astype(np.float32),
            )
            for i in range(30)
        ]
        seq = PoseSequence(
            frames=frames,
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_keypoints=33,
        )
        assert len(seq.frames) == 30
        assert abs(seq.duration - 1.0) < 0.1
        assert seq.detection_rate == 1.0

    def test_pose_sequence_to_array(self):
        """Test PoseSequence to_array method."""
        frames = [
            PoseFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                keypoints=np.ones((33, 3), dtype=np.float32) * i,
            )
            for i in range(10)
        ]
        seq = PoseSequence(
            frames=frames,
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_keypoints=33,
        )
        arr = seq.to_array()
        assert arr.shape == (10, 33, 3)

    def test_pose_extractor_init(self):
        """Test PoseExtractor initialization."""
        extractor = PoseExtractor(use_mediapipe=False)
        assert extractor.use_mediapipe is False

    def test_pose_extractor_context_manager(self):
        """Test PoseExtractor context manager."""
        with PoseExtractor(use_mediapipe=False) as extractor:
            assert extractor is not None

    def test_mediapipe_landmarks_list(self):
        """Test MediaPipe landmarks is a list."""
        assert isinstance(MEDIAPIPE_POSE_LANDMARKS, list)
        assert "nose" in MEDIAPIPE_POSE_LANDMARKS
        assert "left_hip" in MEDIAPIPE_POSE_LANDMARKS
        assert len(MEDIAPIPE_POSE_LANDMARKS) == 33

    def test_gait_joints_dict(self):
        """Test gait joints is a dict with indices."""
        assert isinstance(GAIT_JOINTS, dict)
        assert "left_hip" in GAIT_JOINTS
        assert GAIT_JOINTS["left_hip"] == 23
        assert GAIT_JOINTS["right_ankle"] == 28


class TestGaitFeatures:
    """Tests for GaitFeatures extraction."""

    def test_gait_features_creation(self):
        """Test GaitFeatures dataclass."""
        features = GaitFeatures(
            duration=5.0,
            n_frames=150,
            detection_rate=0.95,
            walking_speed=1.2,
            speed_variability=0.1,
            step_length_mean=0.6,
            step_length_std=0.05,
            step_width_mean=0.15,
            step_width_std=0.02,
            hip_flexion_mean=20.0,
            hip_flexion_range=15.0,
            knee_flexion_mean=45.0,
            knee_flexion_range=30.0,
            hip_asymmetry=2.0,
            knee_asymmetry=3.0,
            ankle_asymmetry=1.5,
            trunk_sway=5.0,
            vertical_oscillation=10.0,
        )
        assert features.duration == 5.0
        assert features.walking_speed == 1.2

    def test_gait_features_to_dict(self):
        """Test GaitFeatures to_dict method."""
        features = GaitFeatures(
            duration=3.0,
            n_frames=90,
            detection_rate=1.0,
            walking_speed=1.0,
            speed_variability=0.1,
            step_length_mean=0.5,
            step_length_std=0.05,
            step_width_mean=0.1,
            step_width_std=0.01,
            hip_flexion_mean=25.0,
            hip_flexion_range=20.0,
            knee_flexion_mean=50.0,
            knee_flexion_range=35.0,
            hip_asymmetry=1.0,
            knee_asymmetry=2.0,
            ankle_asymmetry=1.0,
            trunk_sway=3.0,
            vertical_oscillation=5.0,
        )
        d = features.to_dict()
        assert "duration" in d
        assert "walking_speed" in d
        assert d["walking_speed"] == 1.0

    def test_gait_features_to_array(self):
        """Test GaitFeatures to_array method."""
        features = GaitFeatures(
            duration=1.0, n_frames=30, detection_rate=1.0,
            walking_speed=1.0, speed_variability=0.1,
            step_length_mean=0.5, step_length_std=0.05,
            step_width_mean=0.1, step_width_std=0.01,
            hip_flexion_mean=25.0, hip_flexion_range=20.0,
            knee_flexion_mean=50.0, knee_flexion_range=35.0,
            hip_asymmetry=1.0, knee_asymmetry=2.0, ankle_asymmetry=1.0,
            trunk_sway=3.0, vertical_oscillation=5.0,
        )
        arr = features.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 18  # 18 features

    def test_extract_gait_features_empty_sequence(self):
        """Test gait extraction with empty sequence."""
        seq = PoseSequence(
            frames=[],
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_keypoints=33,
        )
        result = extract_gait_features(seq)
        assert result is not None
        assert result.n_frames == 0
        assert result.walking_speed == 0

    def test_extract_gait_features_with_frames(self):
        """Test gait extraction with frames."""
        frames = []
        for i in range(60):
            keypoints = np.zeros((33, 3), dtype=np.float32)
            # Set hip positions with slight variation
            keypoints[23] = [0.45 + 0.01 * np.sin(i / 10), 0.5, 0.9]  # left hip
            keypoints[24] = [0.55 + 0.01 * np.sin(i / 10), 0.5, 0.9]  # right hip
            # Set ankle positions
            keypoints[27] = [0.44 + 0.02 * np.sin(i / 5), 0.9, 0.9]  # left ankle
            keypoints[28] = [0.56 + 0.02 * np.sin(i / 5 + np.pi), 0.9, 0.9]  # right ankle
            # Set knee positions
            keypoints[25] = [0.44, 0.7, 0.9]  # left knee
            keypoints[26] = [0.56, 0.7, 0.9]  # right knee
            # Set shoulder positions
            keypoints[11] = [0.45, 0.3, 0.9]  # left shoulder
            keypoints[12] = [0.55, 0.3, 0.9]  # right shoulder

            frames.append(PoseFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                keypoints=keypoints,
            ))

        seq = PoseSequence(
            frames=frames,
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_keypoints=33,
        )
        result = extract_gait_features(seq)

        assert result is not None
        assert result.n_frames == 60
        assert result.walking_speed >= 0
        assert result.hip_asymmetry >= 0


class TestFaceExtractor:
    """Tests for FaceExtractor."""

    def test_face_frame_creation(self):
        """Test FaceFrame dataclass."""
        landmarks = np.random.rand(468, 3).astype(np.float32)
        frame = FaceFrame(
            frame_idx=0,
            timestamp=0.0,
            landmarks=landmarks,
            detected=True,
        )
        assert frame.frame_idx == 0
        assert frame.landmarks.shape == (468, 3)
        assert frame.detected is True

    def test_face_frame_no_detection(self):
        """Test FaceFrame with no detection."""
        frame = FaceFrame(
            frame_idx=5,
            timestamp=0.167,
            landmarks=None,
            detected=False,
        )
        assert frame.landmarks is None
        assert frame.detected is False

    def test_face_sequence_empty(self):
        """Test empty FaceSequence."""
        seq = FaceSequence(
            frames=[],
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_landmarks=468,
        )
        assert len(seq.frames) == 0
        assert seq.duration == 0
        assert seq.detection_rate == 0

    def test_face_sequence_with_frames(self):
        """Test FaceSequence with frames."""
        frames = [
            FaceFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                landmarks=np.random.rand(468, 3).astype(np.float32),
                detected=True,
            )
            for i in range(30)
        ]
        seq = FaceSequence(
            frames=frames,
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_landmarks=468,
        )
        assert len(seq.frames) == 30
        assert seq.detection_rate == 1.0

    def test_face_sequence_to_array(self):
        """Test FaceSequence to_array method."""
        frames = [
            FaceFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                landmarks=np.ones((468, 3), dtype=np.float32) * i,
                detected=True,
            )
            for i in range(10)
        ]
        seq = FaceSequence(
            frames=frames,
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_landmarks=468,
        )
        arr = seq.to_array()
        assert arr.shape == (10, 468, 3)

    def test_face_extractor_init(self):
        """Test FaceExtractor initialization."""
        extractor = FaceExtractor(use_mediapipe=False)
        assert extractor.use_mediapipe is False

    def test_face_extractor_context_manager(self):
        """Test FaceExtractor context manager."""
        with FaceExtractor(use_mediapipe=False) as extractor:
            assert extractor is not None

    def test_face_landmarks_dict(self):
        """Test face landmark dictionary."""
        assert isinstance(FACE_LANDMARKS, dict)
        assert "left_eye_outer" in FACE_LANDMARKS
        assert "right_eye_outer" in FACE_LANDMARKS
        assert "mouth_left" in FACE_LANDMARKS
        assert "nose_tip" in FACE_LANDMARKS

    def test_au_definitions_structure(self):
        """Test AU definitions structure."""
        assert isinstance(AU_DEFINITIONS, dict)
        assert "AU1" in AU_DEFINITIONS
        assert "AU12" in AU_DEFINITIONS
        # Each AU is a tuple of (landmark1, landmark2)
        assert isinstance(AU_DEFINITIONS["AU1"], tuple)
        assert len(AU_DEFINITIONS["AU1"]) == 2


class TestFacialFeatures:
    """Tests for FacialFeatures extraction."""

    def test_facial_features_creation(self):
        """Test FacialFeatures dataclass."""
        features = FacialFeatures(
            duration=3.0,
            n_frames=90,
            detection_rate=0.95,
            au_intensities={"AU1": 0.2, "AU12": 0.5},
            eye_asymmetry=0.01,
            mouth_asymmetry=0.02,
            eyebrow_asymmetry=0.015,
            overall_asymmetry=0.015,
            eye_blink_rate=15.0,
            mouth_movement=0.05,
            eyebrow_movement=0.03,
            expression_variability=0.02,
            expression_range=0.1,
        )
        assert features.duration == 3.0
        assert features.eye_blink_rate == 15.0
        assert features.au_intensities["AU12"] == 0.5

    def test_facial_features_to_dict(self):
        """Test FacialFeatures to_dict method."""
        features = FacialFeatures(
            duration=2.0,
            n_frames=60,
            detection_rate=1.0,
            au_intensities={"AU1": 0.1},
            eye_asymmetry=0.01,
            mouth_asymmetry=0.01,
            eyebrow_asymmetry=0.01,
            overall_asymmetry=0.01,
            eye_blink_rate=12.0,
            mouth_movement=0.04,
            eyebrow_movement=0.02,
            expression_variability=0.01,
            expression_range=0.05,
        )
        d = features.to_dict()
        assert "duration" in d
        assert "eye_blink_rate" in d
        assert "au_au1" in d  # AU intensities are flattened
        assert d["eye_blink_rate"] == 12.0

    def test_facial_features_to_array(self):
        """Test FacialFeatures to_array method."""
        features = FacialFeatures(
            duration=1.0, n_frames=30, detection_rate=1.0,
            au_intensities={"AU1": 0.1, "AU12": 0.2},
            eye_asymmetry=0.01, mouth_asymmetry=0.01,
            eyebrow_asymmetry=0.01, overall_asymmetry=0.01,
            eye_blink_rate=10.0, mouth_movement=0.03,
            eyebrow_movement=0.02, expression_variability=0.01,
            expression_range=0.05,
        )
        arr = features.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) >= 12  # At least 12 base features + AU features

    def test_extract_facial_features_empty_sequence(self):
        """Test facial extraction with empty sequence."""
        seq = FaceSequence(
            frames=[],
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_landmarks=468,
        )
        result = extract_facial_features(seq)
        assert result is not None
        assert result.n_frames == 0
        assert result.eye_blink_rate == 0

    def test_extract_facial_features_with_frames(self):
        """Test facial extraction with frames."""
        frames = []
        for i in range(60):
            landmarks = np.random.rand(468, 3).astype(np.float32) * 0.5 + 0.25
            frames.append(FaceFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                landmarks=landmarks,
                detected=True,
            ))

        seq = FaceSequence(
            frames=frames,
            fps=30.0,
            video_width=1920,
            video_height=1080,
            n_landmarks=468,
        )
        result = extract_facial_features(seq)

        assert result is not None
        assert result.n_frames == 60
        assert result.duration == 2.0
        assert result.detection_rate == 1.0
        assert result.eye_asymmetry >= 0
