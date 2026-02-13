#!/usr/bin/env python3
"""
Parkinson's Disease Multi-Symptom Video Analyzer
=================================================
Analyzes multiple PD motor symptoms from video with:
- Multi-person tracking and separation
- Symptom-specific segment detection
- Statistical aggregation across segments

Supported Symptoms:
1. Tremor (resting tremor) - hand frequency analysis
2. Bradykinesia - movement speed, blink rate, facial expression
3. Stooped Posture - spine angle measurement
4. Freezing of Gait (FOG) - gait initiation analysis
5. Postural Instability - balance/sway analysis
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import cv2
import mediapipe as mp


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


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PersonTrack:
    """Tracked person data across video frames."""
    person_id: str
    start_frame: int
    end_frame: int
    keypoints: np.ndarray  # Shape: (frames, 33, 3) for MediaPipe Pose
    face_landmarks: Optional[np.ndarray] = None  # Shape: (frames, 468, 3)
    hand_landmarks_left: Optional[np.ndarray] = None  # Shape: (frames, 21, 3)
    hand_landmarks_right: Optional[np.ndarray] = None  # Shape: (frames, 21, 3)
    confidence_scores: Optional[np.ndarray] = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def get_frame_range(self, start: int, end: int) -> 'PersonTrack':
        """Extract a sub-track for specific frame range."""
        rel_start = max(0, start - self.start_frame)
        rel_end = min(self.duration_frames, end - self.start_frame)
        return PersonTrack(
            person_id=self.person_id,
            start_frame=start,
            end_frame=end,
            keypoints=self.keypoints[rel_start:rel_end],
            face_landmarks=self.face_landmarks[rel_start:rel_end] if self.face_landmarks is not None else None,
            hand_landmarks_left=self.hand_landmarks_left[rel_start:rel_end] if self.hand_landmarks_left is not None else None,
            hand_landmarks_right=self.hand_landmarks_right[rel_start:rel_end] if self.hand_landmarks_right is not None else None,
        )


@dataclass
class AnalysisSegment:
    """A segment suitable for a specific type of analysis."""
    segment_type: str  # 'tremor', 'bradykinesia', 'posture', 'fog', 'gait', 'balance'
    person_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    metadata: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self):
        return _convert_numpy(asdict(self))


@dataclass
class SymptomResult:
    """Result of analyzing a single symptom segment."""
    symptom_type: str
    person_id: str
    segment_id: str
    start_time: float
    end_time: float
    duration: float
    metrics: Dict[str, float]
    severity: str  # 'normal', 'mild', 'moderate', 'severe'
    confidence: float

    def to_dict(self):
        return _convert_numpy(asdict(self))


@dataclass
class PersonSymptomSummary:
    """Statistical summary of a symptom across all segments for one person."""
    person_id: str
    symptom_type: str
    n_segments: int
    total_duration: float
    metrics_stats: Dict[str, Dict]  # {metric_name: {mean, std, ci_lower, ci_upper, values}}
    overall_severity: str
    confidence: float

    def to_dict(self):
        return _convert_numpy(asdict(self))


@dataclass
class ActivitySegment:
    """A classified activity segment in the video."""
    activity_type: str  # 'walking', 'resting', 'task', 'standing'
    person_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    analysis_types: List[str] = field(default_factory=list)  # What analyses can be done
    metadata: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self):
        return _convert_numpy(asdict(self))


# =============================================================================
# MULTI-PERSON TRACKER
# =============================================================================

class MultiPersonTracker:
    """Track multiple people across video frames using pose estimation."""

    # MediaPipe Pose landmarks
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, max_persons: int = 5, min_track_frames: int = 30):
        """
        Args:
            max_persons: Maximum number of people to track
            min_track_frames: Minimum frames for a valid track
        """
        self.max_persons = max_persons
        self.min_track_frames = min_track_frames
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

    def extract_tracks(self, video_path: str, fps: float = None,
                       extract_hands: bool = True, extract_face: bool = True,
                       progress_callback=None) -> Tuple[List[PersonTrack], Dict]:
        """
        Extract person tracks from video.

        Returns:
            List of PersonTrack objects, one per detected person
            Video metadata dict
        """
        cap = cv2.VideoCapture(video_path)
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration': total_frames / fps,
            'width': width,
            'height': height
        }

        # Initialize detectors
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        hands = None
        face_mesh = None
        if extract_hands:
            hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        if extract_face:
            face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_persons,
                min_detection_confidence=0.5
            )

        # Storage for active tracks
        active_tracks: Dict[int, Dict] = {}  # track_id -> {keypoints, hands, face, ...}
        completed_tracks: List[PersonTrack] = []
        next_track_id = 0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect pose
            pose_results = pose.process(frame_rgb)

            # Detect hands
            hand_results = hands.process(frame_rgb) if hands else None

            # Detect face
            face_results = face_mesh.process(frame_rgb) if face_mesh else None

            # Process detections
            if pose_results.pose_landmarks:
                # Extract pose keypoints
                landmarks = np.array([
                    [lm.x * width, lm.y * height, lm.z]
                    for lm in pose_results.pose_landmarks.landmark
                ])

                # Simple tracking: find closest existing track or create new
                track_id = self._match_or_create_track(
                    landmarks, active_tracks, frame_idx, next_track_id
                )
                if track_id == next_track_id:
                    next_track_id += 1

                # Store keypoints
                if track_id not in active_tracks:
                    active_tracks[track_id] = {
                        'start_frame': frame_idx,
                        'keypoints': [],
                        'face_landmarks': [],
                        'hand_left': [],
                        'hand_right': [],
                        'last_frame': frame_idx
                    }

                active_tracks[track_id]['keypoints'].append(landmarks)
                active_tracks[track_id]['last_frame'] = frame_idx

                # Store hand landmarks
                if hand_results and hand_results.multi_hand_landmarks:
                    left_hand = None
                    right_hand = None
                    for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        handedness = hand_results.multi_handedness[i].classification[0].label
                        hand_arr = np.array([
                            [lm.x * width, lm.y * height, lm.z]
                            for lm in hand_landmarks.landmark
                        ])
                        if handedness == 'Left':
                            left_hand = hand_arr
                        else:
                            right_hand = hand_arr
                    active_tracks[track_id]['hand_left'].append(left_hand)
                    active_tracks[track_id]['hand_right'].append(right_hand)
                else:
                    active_tracks[track_id]['hand_left'].append(None)
                    active_tracks[track_id]['hand_right'].append(None)

                # Store face landmarks
                if face_results and face_results.multi_face_landmarks:
                    face_arr = np.array([
                        [lm.x * width, lm.y * height, lm.z]
                        for lm in face_results.multi_face_landmarks[0].landmark
                    ])
                    active_tracks[track_id]['face_landmarks'].append(face_arr)
                else:
                    active_tracks[track_id]['face_landmarks'].append(None)

            # Check for tracks that ended (no detection for N frames)
            ended_tracks = []
            for tid, track in active_tracks.items():
                if frame_idx - track['last_frame'] > fps * 0.5:  # 0.5 second gap
                    ended_tracks.append(tid)

            for tid in ended_tracks:
                track = active_tracks.pop(tid)
                if len(track['keypoints']) >= self.min_track_frames:
                    completed_tracks.append(self._finalize_track(tid, track))

            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx / total_frames * 100)

        cap.release()
        pose.close()
        if hands:
            hands.close()
        if face_mesh:
            face_mesh.close()

        # Finalize remaining active tracks
        for tid, track in active_tracks.items():
            if len(track['keypoints']) >= self.min_track_frames:
                completed_tracks.append(self._finalize_track(tid, track))

        # Sort by start frame and assign person IDs
        completed_tracks.sort(key=lambda t: t.start_frame)
        for i, track in enumerate(completed_tracks):
            track.person_id = f"person_{i + 1}"

        return completed_tracks, video_info

    def _match_or_create_track(self, landmarks: np.ndarray,
                                active_tracks: Dict, frame_idx: int,
                                next_track_id: int) -> int:
        """Match detection to existing track or create new one."""
        if not active_tracks:
            return next_track_id

        # Use torso center for matching
        torso_center = (landmarks[self.LEFT_HIP] + landmarks[self.RIGHT_HIP] +
                        landmarks[self.LEFT_SHOULDER] + landmarks[self.RIGHT_SHOULDER]) / 4

        best_match = None
        best_dist = float('inf')

        for tid, track in active_tracks.items():
            if frame_idx - track['last_frame'] > 15:  # Skip stale tracks
                continue

            # Compare with last known position
            last_kp = track['keypoints'][-1]
            last_center = (last_kp[self.LEFT_HIP] + last_kp[self.RIGHT_HIP] +
                          last_kp[self.LEFT_SHOULDER] + last_kp[self.RIGHT_SHOULDER]) / 4

            dist = np.linalg.norm(torso_center[:2] - last_center[:2])
            if dist < best_dist and dist < 200:  # Max 200 pixel movement
                best_dist = dist
                best_match = tid

        return best_match if best_match is not None else next_track_id

    def _finalize_track(self, track_id: int, track_data: Dict) -> PersonTrack:
        """Convert raw track data to PersonTrack object."""
        keypoints = np.array(track_data['keypoints'])

        # Process hand landmarks (handle None values)
        hand_left = self._process_optional_landmarks(track_data['hand_left'], 21)
        hand_right = self._process_optional_landmarks(track_data['hand_right'], 21)
        face = self._process_optional_landmarks(track_data['face_landmarks'], 468)

        return PersonTrack(
            person_id=f"track_{track_id}",
            start_frame=track_data['start_frame'],
            end_frame=track_data['start_frame'] + len(keypoints),
            keypoints=keypoints,
            face_landmarks=face,
            hand_landmarks_left=hand_left,
            hand_landmarks_right=hand_right
        )

    def _process_optional_landmarks(self, landmarks_list: List,
                                     num_points: int) -> Optional[np.ndarray]:
        """Process list of landmarks that may contain None values."""
        if not landmarks_list or all(l is None for l in landmarks_list):
            return None

        result = np.zeros((len(landmarks_list), num_points, 3))
        last_valid = None

        for i, lm in enumerate(landmarks_list):
            if lm is not None:
                result[i] = lm
                last_valid = lm
            elif last_valid is not None:
                result[i] = last_valid  # Forward fill

        return result


# =============================================================================
# UNIFIED ACTIVITY SEGMENT DETECTOR
# =============================================================================

class UnifiedActivityDetector:
    """
    Detect and classify activity segments in video before analysis.

    Activity types and their corresponding analyses:
    - 'walking' (보행구간): Gait analysis, FOG detection
    - 'resting' (안정구간): Tremor analysis (hands at rest)
    - 'task' (작업구간): Bradykinesia, rigidity analysis (hand movements)
    - 'standing' (서있는 구간): Posture analysis
    """

    # Landmark indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0

    # Activity mapping to analysis types
    ACTIVITY_ANALYSIS_MAP = {
        'walking': ['gait'],  # FOG is analyzed at standing↔walking transitions
        'resting': ['tremor'],
        'task': ['bradykinesia', 'rigidity'],
        'standing': ['posture']
    }

    def __init__(self,
                 min_segment_duration: float = 1.0,
                 walking_velocity_threshold: float = 15.0,  # pixels/sec hip movement (lowered for better detection)
                 resting_velocity_threshold: float = 20.0,  # pixels/sec hand movement
                 task_velocity_threshold: float = 50.0):    # pixels/sec hand movement
        """
        Args:
            min_segment_duration: Minimum segment duration in seconds
            walking_velocity_threshold: Hip velocity above this = walking
            resting_velocity_threshold: Hand velocity below this = resting
            task_velocity_threshold: Hand velocity above this = task/movement
        """
        self.min_segment_duration = min_segment_duration
        self.walking_velocity_threshold = walking_velocity_threshold
        self.resting_velocity_threshold = resting_velocity_threshold
        self.task_velocity_threshold = task_velocity_threshold

    def detect_activities(self, track: PersonTrack, fps: float) -> List[ActivitySegment]:
        """
        Detect and classify all activity segments for a person track.

        Returns:
            List of ActivitySegment objects, sorted by start time
        """
        n_frames = track.duration_frames
        if n_frames < fps * self.min_segment_duration:
            return []

        # Calculate movement velocities
        hip_velocity = self._calculate_hip_velocity(track, fps)
        hand_velocity = self._calculate_hand_velocity(track, fps)
        body_height = self._calculate_body_height(track)

        # Smooth velocities
        window = max(3, int(fps * 0.3))
        hip_velocity_smooth = uniform_filter1d(hip_velocity, window)
        hand_velocity_smooth = uniform_filter1d(hand_velocity, window)

        # Classify each frame
        frame_activities = self._classify_frames(
            hip_velocity_smooth,
            hand_velocity_smooth,
            body_height,
            fps
        )

        # Extract continuous segments
        segments = self._extract_segments(
            frame_activities,
            track,
            fps,
            hip_velocity_smooth,
            hand_velocity_smooth
        )

        return segments

    def _calculate_hip_velocity(self, track: PersonTrack, fps: float) -> np.ndarray:
        """Calculate hip center velocity (for walking detection)."""
        hip_left = track.keypoints[:, self.LEFT_HIP, :2]
        hip_right = track.keypoints[:, self.RIGHT_HIP, :2]
        hip_center = (hip_left + hip_right) / 2

        velocity = np.sqrt(np.sum(np.diff(hip_center, axis=0)**2, axis=1)) * fps
        return np.concatenate([[0], velocity])

    def _calculate_hand_velocity(self, track: PersonTrack, fps: float) -> np.ndarray:
        """Calculate combined hand velocity (for resting/task detection)."""
        wrist_left = track.keypoints[:, self.LEFT_WRIST, :2]
        wrist_right = track.keypoints[:, self.RIGHT_WRIST, :2]

        vel_left = np.sqrt(np.sum(np.diff(wrist_left, axis=0)**2, axis=1)) * fps
        vel_right = np.sqrt(np.sum(np.diff(wrist_right, axis=0)**2, axis=1)) * fps

        # Use max of both hands
        combined = np.maximum(vel_left, vel_right)
        return np.concatenate([[0], combined])

    def _calculate_body_height(self, track: PersonTrack) -> np.ndarray:
        """Calculate body height in pixels (for standing detection)."""
        shoulders = (track.keypoints[:, self.LEFT_SHOULDER, 1] +
                    track.keypoints[:, self.RIGHT_SHOULDER, 1]) / 2
        ankles = (track.keypoints[:, self.LEFT_ANKLE, 1] +
                 track.keypoints[:, self.RIGHT_ANKLE, 1]) / 2

        # Y increases downward in image
        return ankles - shoulders

    def _classify_frames(self, hip_velocity: np.ndarray,
                         hand_velocity: np.ndarray,
                         body_height: np.ndarray,
                         fps: float) -> np.ndarray:
        """
        Classify each frame into activity type.

        Priority order:
        1. Walking (hip moving significantly)
        2. Task (hands moving significantly while not walking)
        3. Standing (upright posture, minimal movement)
        4. Resting (hands at rest)

        Returns:
            Array of activity type strings for each frame
        """
        n_frames = len(hip_velocity)
        activities = np.array(['unknown'] * n_frames, dtype=object)

        # Calculate thresholds based on data distribution
        height_threshold = np.percentile(body_height, 50) * 0.7

        is_walking = hip_velocity > self.walking_velocity_threshold
        is_task = (hand_velocity > self.task_velocity_threshold) & ~is_walking
        is_resting = (hand_velocity < self.resting_velocity_threshold) & ~is_walking
        is_standing = (body_height > height_threshold) & ~is_walking & ~is_task

        # Apply classifications with priority
        activities[is_resting] = 'resting'
        activities[is_standing] = 'standing'
        activities[is_task] = 'task'
        activities[is_walking] = 'walking'

        # Fill remaining 'unknown' with nearest valid activity
        unknown_mask = activities == 'unknown'
        if np.any(unknown_mask) and not np.all(unknown_mask):
            # Forward fill then backward fill
            last_known = 'standing'  # Default
            for i in range(n_frames):
                if activities[i] != 'unknown':
                    last_known = activities[i]
                else:
                    activities[i] = last_known
        elif np.all(unknown_mask):
            activities[:] = 'standing'  # Default if all unknown

        return activities

    def _extract_segments(self, frame_activities: np.ndarray,
                          track: PersonTrack, fps: float,
                          hip_velocity: np.ndarray,
                          hand_velocity: np.ndarray) -> List[ActivitySegment]:
        """Extract continuous activity segments."""
        segments = []
        min_frames = int(self.min_segment_duration * fps)

        # Find activity changes
        current_activity = frame_activities[0]
        segment_start = 0

        for i in range(1, len(frame_activities)):
            if frame_activities[i] != current_activity:
                # End current segment
                if i - segment_start >= min_frames:
                    segments.append(self._create_segment(
                        current_activity, segment_start, i,
                        track, fps, hip_velocity, hand_velocity
                    ))

                # Start new segment
                current_activity = frame_activities[i]
                segment_start = i

        # Handle last segment
        if len(frame_activities) - segment_start >= min_frames:
            segments.append(self._create_segment(
                current_activity, segment_start, len(frame_activities),
                track, fps, hip_velocity, hand_velocity
            ))

        return segments

    def _create_segment(self, activity_type: str, start_idx: int, end_idx: int,
                        track: PersonTrack, fps: float,
                        hip_velocity: np.ndarray,
                        hand_velocity: np.ndarray) -> ActivitySegment:
        """Create an ActivitySegment with metadata."""
        # Calculate confidence based on velocity consistency
        if activity_type == 'walking':
            segment_vel = hip_velocity[start_idx:end_idx]
            confidence = np.mean(segment_vel > self.walking_velocity_threshold)
        elif activity_type == 'resting':
            segment_vel = hand_velocity[start_idx:end_idx]
            confidence = np.mean(segment_vel < self.resting_velocity_threshold)
        elif activity_type == 'task':
            segment_vel = hand_velocity[start_idx:end_idx]
            confidence = np.mean(segment_vel > self.task_velocity_threshold)
        else:  # standing
            confidence = 0.8

        return ActivitySegment(
            activity_type=activity_type,
            person_id=track.person_id,
            start_frame=track.start_frame + start_idx,
            end_frame=track.start_frame + end_idx,
            start_time=(track.start_frame + start_idx) / fps,
            end_time=(track.start_frame + end_idx) / fps,
            confidence=float(confidence),
            analysis_types=self.ACTIVITY_ANALYSIS_MAP.get(activity_type, []),
            metadata={
                'avg_hip_velocity': float(np.mean(hip_velocity[start_idx:end_idx])),
                'avg_hand_velocity': float(np.mean(hand_velocity[start_idx:end_idx]))
            }
        )


# =============================================================================
# SEGMENT DETECTORS (Legacy - kept for compatibility)
# =============================================================================

class SegmentDetector:
    """Base class for detecting analysis-ready segments."""

    def __init__(self, min_duration: float = 1.0):
        self.min_duration = min_duration

    def detect(self, track: PersonTrack, fps: float) -> List[AnalysisSegment]:
        raise NotImplementedError


class TremorSegmentDetector(SegmentDetector):
    """Detect segments where hands are at rest (suitable for tremor analysis).

    Tremor analysis requires hands to be relatively still (no intentional movement).
    We detect when hand velocity is low but may contain tremor oscillations.
    """

    def __init__(self, min_duration: float = 1.0, velocity_threshold: float = 50.0):
        super().__init__(min_duration)
        self.velocity_threshold = velocity_threshold  # pixels/second

    def detect(self, track: PersonTrack, fps: float) -> List[AnalysisSegment]:
        segments = []

        # Use wrist positions from pose if hand landmarks not available
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        for side, wrist_idx in [('left', LEFT_WRIST), ('right', RIGHT_WRIST)]:
            wrist_pos = track.keypoints[:, wrist_idx, :2]

            # Calculate velocity
            velocity = np.sqrt(np.sum(np.diff(wrist_pos, axis=0)**2, axis=1)) * fps
            velocity = np.concatenate([[0], velocity])

            # Smooth velocity
            window = max(3, int(fps * 0.3))
            velocity_smooth = uniform_filter1d(velocity, window)

            # Find low-velocity (resting) segments
            is_resting = velocity_smooth < self.velocity_threshold

            # Extract continuous segments
            changes = np.diff(np.concatenate([[0], is_resting.astype(int), [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            min_frames = int(self.min_duration * fps)
            for start, end in zip(starts, ends):
                if end - start >= min_frames:
                    segments.append(AnalysisSegment(
                        segment_type='tremor',
                        person_id=track.person_id,
                        start_frame=track.start_frame + start,
                        end_frame=track.start_frame + end,
                        start_time=(track.start_frame + start) / fps,
                        end_time=(track.start_frame + end) / fps,
                        confidence=1.0 - np.mean(velocity_smooth[start:end]) / self.velocity_threshold,
                        metadata={'hand': side}
                    ))

        return segments


class BradykinesiaSegmentDetector(SegmentDetector):
    """Detect segments suitable for bradykinesia analysis.

    Looks for:
    - Finger tapping movements (if detected)
    - General hand movements
    - Facial visibility for hypomimia analysis
    """

    def __init__(self, min_duration: float = 1.0):
        super().__init__(min_duration)

    def detect(self, track: PersonTrack, fps: float) -> List[AnalysisSegment]:
        segments = []

        # Detect hand movement segments
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        wrist_left = track.keypoints[:, LEFT_WRIST, :2]
        wrist_right = track.keypoints[:, RIGHT_WRIST, :2]

        # Calculate hand activity
        vel_left = np.sqrt(np.sum(np.diff(wrist_left, axis=0)**2, axis=1)) * fps
        vel_right = np.sqrt(np.sum(np.diff(wrist_right, axis=0)**2, axis=1)) * fps
        combined_activity = np.concatenate([[0], vel_left + vel_right])

        # Smooth
        window = max(3, int(fps * 0.5))
        activity_smooth = uniform_filter1d(combined_activity, window)

        # Detect active movement periods (for movement speed analysis)
        activity_threshold = np.percentile(activity_smooth, 30)
        is_active = activity_smooth > activity_threshold

        # Extract segments
        changes = np.diff(np.concatenate([[0], is_active.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        min_frames = int(self.min_duration * fps)
        for start, end in zip(starts, ends):
            if end - start >= min_frames:
                segments.append(AnalysisSegment(
                    segment_type='bradykinesia',
                    person_id=track.person_id,
                    start_frame=track.start_frame + start,
                    end_frame=track.start_frame + end,
                    start_time=(track.start_frame + start) / fps,
                    end_time=(track.start_frame + end) / fps,
                    confidence=min(1.0, np.mean(activity_smooth[start:end]) / 100),
                    metadata={'subtype': 'movement'}
                ))

        # Detect facial segments (for hypomimia)
        if track.face_landmarks is not None:
            # Check face visibility
            face_valid = np.array([f is not None and np.any(f != 0)
                                   for f in track.face_landmarks])

            # Require continuous face visibility
            changes = np.diff(np.concatenate([[0], face_valid.astype(int), [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            for start, end in zip(starts, ends):
                if end - start >= min_frames:
                    segments.append(AnalysisSegment(
                        segment_type='bradykinesia',
                        person_id=track.person_id,
                        start_frame=track.start_frame + start,
                        end_frame=track.start_frame + end,
                        start_time=(track.start_frame + start) / fps,
                        end_time=(track.start_frame + end) / fps,
                        confidence=np.mean(face_valid[start:end]),
                        metadata={'subtype': 'facial'}
                    ))

        return segments


class PostureSegmentDetector(SegmentDetector):
    """Detect segments suitable for posture analysis.

    Requires standing or walking with visible upper body.
    """

    def __init__(self, min_duration: float = 1.0):
        super().__init__(min_duration)

    def detect(self, track: PersonTrack, fps: float) -> List[AnalysisSegment]:
        segments = []

        # Check for standing/walking posture
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        # Calculate vertical extent (should be large for standing)
        shoulders = (track.keypoints[:, LEFT_SHOULDER, 1] +
                     track.keypoints[:, RIGHT_SHOULDER, 1]) / 2
        hips = (track.keypoints[:, LEFT_HIP, 1] +
                track.keypoints[:, RIGHT_HIP, 1]) / 2
        ankles = (track.keypoints[:, LEFT_ANKLE, 1] +
                  track.keypoints[:, RIGHT_ANKLE, 1]) / 2

        # Body height in pixels
        body_height = ankles - shoulders  # Y increases downward

        # Detect upright segments (body height > threshold)
        height_threshold = np.percentile(body_height, 50) * 0.7
        is_upright = body_height > height_threshold

        # Extract segments
        changes = np.diff(np.concatenate([[0], is_upright.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        min_frames = int(self.min_duration * fps)
        for start, end in zip(starts, ends):
            if end - start >= min_frames:
                segments.append(AnalysisSegment(
                    segment_type='posture',
                    person_id=track.person_id,
                    start_frame=track.start_frame + start,
                    end_frame=track.start_frame + end,
                    start_time=(track.start_frame + start) / fps,
                    end_time=(track.start_frame + end) / fps,
                    confidence=np.mean(body_height[start:end]) / np.max(body_height),
                    metadata={}
                ))

        return segments


class FOGSegmentDetector(SegmentDetector):
    """Detect segments suitable for Freezing of Gait analysis.

    FOG typically occurs:
    - At gait initiation
    - During turning
    - When approaching obstacles/doorways
    """

    def __init__(self, min_duration: float = 1.0):
        super().__init__(min_duration)

    def detect(self, track: PersonTrack, fps: float) -> List[AnalysisSegment]:
        segments = []

        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HIP = 23
        RIGHT_HIP = 24

        # Calculate leg movement
        ankle_left = track.keypoints[:, LEFT_ANKLE, :2]
        ankle_right = track.keypoints[:, RIGHT_ANKLE, :2]
        hip_center = (track.keypoints[:, LEFT_HIP, :2] +
                      track.keypoints[:, RIGHT_HIP, :2]) / 2

        # Ankle velocity
        vel_left = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1)) * fps
        vel_right = np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1)) * fps
        ankle_velocity = np.concatenate([[0], (vel_left + vel_right) / 2])

        # Hip velocity (body movement)
        hip_velocity = np.sqrt(np.sum(np.diff(hip_center, axis=0)**2, axis=1)) * fps
        hip_velocity = np.concatenate([[0], hip_velocity])

        # Smooth
        window = max(3, int(fps * 0.3))
        ankle_smooth = uniform_filter1d(ankle_velocity, window)
        hip_smooth = uniform_filter1d(hip_velocity, window)

        # Detect potential FOG: low ankle movement but body trying to move
        # Or: alternating movement attempts
        is_walking_attempt = (hip_smooth > np.percentile(hip_smooth, 30))

        # Look for segments with walking attempts
        changes = np.diff(np.concatenate([[0], is_walking_attempt.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        min_frames = int(self.min_duration * fps)
        for start, end in zip(starts, ends):
            if end - start >= min_frames:
                # Calculate ankle/hip ratio (low ratio may indicate FOG)
                ankle_activity = np.mean(ankle_smooth[start:end])
                hip_activity = np.mean(hip_smooth[start:end])

                segments.append(AnalysisSegment(
                    segment_type='fog',
                    person_id=track.person_id,
                    start_frame=track.start_frame + start,
                    end_frame=track.start_frame + end,
                    start_time=(track.start_frame + start) / fps,
                    end_time=(track.start_frame + end) / fps,
                    confidence=0.8,
                    metadata={
                        'ankle_activity': float(ankle_activity),
                        'hip_activity': float(hip_activity)
                    }
                ))

        return segments


# =============================================================================
# SYMPTOM ANALYZERS
# =============================================================================

class TremorAnalyzer:
    """
    Analyze tremor characteristics from rest segments.

    Literature-based thresholds:
    - PD resting tremor: 3-6 Hz (classic "pill-rolling" oscillations)
      Reference: Movement Disorders Clinical Practice 2023
    - Essential tremor: 6-12 Hz
    - Action/postural tremor: 4-12 Hz

    MDS-UPDRS scoring:
    - Score 0: Normal
    - Score 1: Slight tremor, barely visible
    - Score 2: Mild amplitude tremor (< 1cm)
    - Score 3: Moderate amplitude (1-3 cm)
    - Score 4: Marked amplitude (3-10 cm)
    """

    # Literature: PD rest tremor typically 3-6 Hz (Pasquini et al., 2023)
    PD_TREMOR_FREQ_RANGE = (3.0, 6.0)
    ESSENTIAL_TREMOR_FREQ_RANGE = (6.0, 12.0)

    # Amplitude thresholds in pixels (calibrated for typical video resolution)
    # Assuming ~500px body height, 1cm ≈ 5-10px
    AMPLITUDE_MILD = 1.0       # Barely visible tremor
    AMPLITUDE_MODERATE = 3.0   # Clear tremor < 1cm equivalent
    AMPLITUDE_MARKED = 8.0     # Moderate tremor 1-3cm equivalent
    AMPLITUDE_SEVERE = 15.0    # Marked tremor > 3cm equivalent

    # Power ratio thresholds (lowered based on literature sensitivity)
    PD_POWER_RATIO_THRESHOLD = 0.15  # Lowered from 0.3 for better sensitivity

    def analyze(self, track: PersonTrack, segment: AnalysisSegment,
                fps: float) -> SymptomResult:
        """Analyze tremor in the given segment."""
        # Get relevant frames
        start_rel = segment.start_frame - track.start_frame
        end_rel = segment.end_frame - track.start_frame

        hand_side = segment.metadata.get('hand', 'right')
        wrist_idx = 15 if hand_side == 'left' else 16

        wrist_pos = track.keypoints[start_rel:end_rel, wrist_idx, :2]

        # Calculate movement signal
        movement = np.diff(wrist_pos, axis=0)
        movement_magnitude = np.sqrt(np.sum(movement**2, axis=1))

        if len(movement_magnitude) < fps * 2:
            return self._empty_result(segment)

        # FFT analysis for frequency detection
        n = len(movement_magnitude)
        yf = np.abs(fft(movement_magnitude - np.mean(movement_magnitude)))
        xf = fftfreq(n, 1/fps)

        # Focus on positive frequencies in tremor range
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = yf[pos_mask]

        # Find dominant frequency
        if len(yf_pos) > 0:
            peak_idx = np.argmax(yf_pos)
            dominant_freq = xf_pos[peak_idx]
            peak_power = yf_pos[peak_idx]

            # Calculate power in PD tremor range (3-6 Hz)
            pd_mask = (xf_pos >= self.PD_TREMOR_FREQ_RANGE[0]) & \
                      (xf_pos <= self.PD_TREMOR_FREQ_RANGE[1])
            pd_power = np.sum(yf_pos[pd_mask]) if np.any(pd_mask) else 0

            # Calculate power in extended tremor range (3-8 Hz for action tremor)
            extended_mask = (xf_pos >= 3.0) & (xf_pos <= 8.0)
            extended_power = np.sum(yf_pos[extended_mask]) if np.any(extended_mask) else 0

            total_power = np.sum(yf_pos)
            pd_power_ratio = pd_power / total_power if total_power > 0 else 0
            extended_power_ratio = extended_power / total_power if total_power > 0 else 0

            # Find peak frequency within PD range
            if np.any(pd_mask):
                pd_freqs = xf_pos[pd_mask]
                pd_powers = yf_pos[pd_mask]
                pd_peak_freq = pd_freqs[np.argmax(pd_powers)]
            else:
                pd_peak_freq = 0
        else:
            dominant_freq = 0
            peak_power = 0
            pd_power_ratio = 0
            extended_power_ratio = 0
            pd_peak_freq = 0

        # Tremor amplitude (RMS of movement in pixels)
        tremor_amplitude = np.sqrt(np.mean(movement_magnitude**2))

        # Peak-to-peak amplitude
        peak_amplitude = np.max(movement_magnitude) if len(movement_magnitude) > 0 else 0

        # Determine severity based on MDS-UPDRS criteria
        is_pd_frequency = self.PD_TREMOR_FREQ_RANGE[0] <= dominant_freq <= self.PD_TREMOR_FREQ_RANGE[1]
        has_significant_power = pd_power_ratio > self.PD_POWER_RATIO_THRESHOLD or extended_power_ratio > 0.2

        if has_significant_power or tremor_amplitude > self.AMPLITUDE_MILD:
            if tremor_amplitude > self.AMPLITUDE_SEVERE:
                severity = 'severe'  # UPDRS 4
            elif tremor_amplitude > self.AMPLITUDE_MARKED:
                severity = 'moderate'  # UPDRS 3
            elif tremor_amplitude > self.AMPLITUDE_MODERATE:
                severity = 'mild'  # UPDRS 2
            elif tremor_amplitude > self.AMPLITUDE_MILD or has_significant_power:
                severity = 'mild'  # UPDRS 1-2
            else:
                severity = 'normal'
        else:
            severity = 'normal'

        # Calculate Freeze Index (FI) - literature method
        # FI = power(freeze band 3-8Hz) / power(locomotor band 0.5-3Hz)
        locomotor_mask = (xf_pos >= 0.5) & (xf_pos <= 3.0)
        freeze_mask = (xf_pos >= 3.0) & (xf_pos <= 8.0)
        locomotor_power = np.sum(yf_pos[locomotor_mask]) if np.any(locomotor_mask) else 1e-6
        freeze_power = np.sum(yf_pos[freeze_mask]) if np.any(freeze_mask) else 0
        freeze_index = freeze_power / locomotor_power

        # Tremor regularity (consistency of oscillation)
        if len(movement_magnitude) > 10:
            # Calculate autocorrelation for regularity
            autocorr = np.correlate(movement_magnitude - np.mean(movement_magnitude),
                                   movement_magnitude - np.mean(movement_magnitude), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-6)

            # Find first peak after zero crossing (regularity measure)
            peaks, _ = signal.find_peaks(autocorr, height=0.3)
            regularity = autocorr[peaks[0]] if len(peaks) > 0 else 0
        else:
            regularity = 0

        return SymptomResult(
            symptom_type='tremor',
            person_id=segment.person_id,
            segment_id=f"tremor_{segment.start_frame}_{hand_side}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={
                'dominant_frequency': float(dominant_freq),
                'pd_peak_frequency': float(pd_peak_freq),
                'tremor_amplitude_rms': float(tremor_amplitude),
                'tremor_amplitude_peak': float(peak_amplitude),
                'pd_power_ratio': float(pd_power_ratio),
                'extended_power_ratio': float(extended_power_ratio),
                'freeze_index': float(freeze_index),
                'tremor_regularity': float(regularity),
                'is_pd_frequency': bool(is_pd_frequency),
            },
            severity=severity,
            confidence=segment.confidence
        )

    def _empty_result(self, segment: AnalysisSegment) -> SymptomResult:
        return SymptomResult(
            symptom_type='tremor',
            person_id=segment.person_id,
            segment_id=f"tremor_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={},
            severity='unknown',
            confidence=0
        )


class BradykinesiaAnalyzer:
    """
    Analyze bradykinesia (slowness of movement) based on MDS-UPDRS criteria.

    Literature-based analysis methods:
    1. Finger tapping velocity and amplitude decay (decrement)
    2. Blink rate analysis (normal: 15-24/min, PD: 3-15/min)
    3. Facial expression analysis (hypomimia)

    References:
    - PLOS One: Smartphone finger tapping validation (2016)
    - Scientific Reports: Distal Finger Tapping test (2021)
    - Frontiers Neurology: Hypomimia review (2020)
    """

    # Literature-based blink rate thresholds
    # Normal: 15-24 blinks/min (Karson et al., 1984)
    # PD: can be 3-15 blinks/min (Neurology 1984)
    NORMAL_BLINK_RATE_MIN = 15
    NORMAL_BLINK_RATE_MAX = 24
    PD_BLINK_RATE_SEVERE = 5   # Very low blink rate
    PD_BLINK_RATE_MODERATE = 10
    PD_BLINK_RATE_MILD = 15

    # Velocity thresholds (relative, as pixels vary by video)
    # Based on: Controls ~1.48°/s, PD ~0.96°/s (65% of normal)
    VELOCITY_RATIO_SEVERE = 0.4    # <40% of normal
    VELOCITY_RATIO_MODERATE = 0.6  # 40-60% of normal
    VELOCITY_RATIO_MILD = 0.8      # 60-80% of normal

    # Decrement detection (amplitude/velocity reduction over time)
    DECREMENT_THRESHOLD_MILD = 0.2     # 20% reduction
    DECREMENT_THRESHOLD_MODERATE = 0.4 # 40% reduction
    DECREMENT_THRESHOLD_SEVERE = 0.6   # 60% reduction

    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Additional facial landmarks for hypomimia
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    def analyze(self, track: PersonTrack, segment: AnalysisSegment,
                fps: float) -> SymptomResult:
        """Analyze bradykinesia in the given segment."""
        subtype = segment.metadata.get('subtype', 'movement')

        if subtype == 'facial' and track.face_landmarks is not None:
            return self._analyze_facial(track, segment, fps)
        else:
            return self._analyze_movement(track, segment, fps)

    def _analyze_movement(self, track: PersonTrack, segment: AnalysisSegment,
                          fps: float) -> SymptomResult:
        """
        Analyze movement speed, amplitude, and decrement.

        MDS-UPDRS bradykinesia criteria:
        - Speed of movement
        - Amplitude of movement
        - Decrement (progressive reduction) in amplitude/speed
        - Hesitations and halts
        """
        start_rel = segment.start_frame - track.start_frame
        end_rel = segment.end_frame - track.start_frame

        # Analyze wrist movements
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        wrist_left = track.keypoints[start_rel:end_rel, LEFT_WRIST, :2]
        wrist_right = track.keypoints[start_rel:end_rel, RIGHT_WRIST, :2]

        # Calculate velocities
        vel_left = np.sqrt(np.sum(np.diff(wrist_left, axis=0)**2, axis=1)) * fps
        vel_right = np.sqrt(np.sum(np.diff(wrist_right, axis=0)**2, axis=1)) * fps

        if len(vel_left) < 10:
            return self._empty_result(segment, 'movement')

        avg_velocity = np.mean(np.concatenate([vel_left, vel_right]))
        max_velocity = np.max(np.concatenate([vel_left, vel_right]))
        min_velocity = np.min(np.concatenate([vel_left, vel_right]))

        # Movement amplitude (peak-to-peak in each window)
        left_range = np.max(wrist_left, axis=0) - np.min(wrist_left, axis=0)
        right_range = np.max(wrist_right, axis=0) - np.min(wrist_right, axis=0)
        avg_amplitude = (np.linalg.norm(left_range) + np.linalg.norm(right_range)) / 2

        # === DECREMENT ANALYSIS (Key PD marker) ===
        # Divide movement into halves and compare velocity/amplitude
        half_point = len(vel_left) // 2
        if half_point > 5:
            vel_first_half = np.mean(np.concatenate([vel_left[:half_point], vel_right[:half_point]]))
            vel_second_half = np.mean(np.concatenate([vel_left[half_point:], vel_right[half_point:]]))

            # Decrement ratio: how much velocity decreases
            velocity_decrement = 1 - (vel_second_half / (vel_first_half + 1e-6))
            velocity_decrement = max(0, velocity_decrement)  # Only count decreases

            # Amplitude decrement
            amp_first = (np.linalg.norm(np.max(wrist_left[:half_point], axis=0) - np.min(wrist_left[:half_point], axis=0)) +
                        np.linalg.norm(np.max(wrist_right[:half_point], axis=0) - np.min(wrist_right[:half_point], axis=0))) / 2
            amp_second = (np.linalg.norm(np.max(wrist_left[half_point:], axis=0) - np.min(wrist_left[half_point:], axis=0)) +
                         np.linalg.norm(np.max(wrist_right[half_point:], axis=0) - np.min(wrist_right[half_point:], axis=0))) / 2
            amplitude_decrement = 1 - (amp_second / (amp_first + 1e-6))
            amplitude_decrement = max(0, amplitude_decrement)
        else:
            velocity_decrement = 0
            amplitude_decrement = 0

        # === HESITATION DETECTION ===
        # Count frames where velocity drops significantly below average
        hesitation_threshold = avg_velocity * 0.3
        hesitation_frames = np.sum(np.concatenate([vel_left, vel_right]) < hesitation_threshold)
        hesitation_ratio = hesitation_frames / (len(vel_left) + len(vel_right))

        # === MOVEMENT REGULARITY ===
        # Calculate coefficient of variation (higher = more irregular)
        velocity_cv = np.std(np.concatenate([vel_left, vel_right])) / (avg_velocity + 1e-6)

        # === SEVERITY SCORING (MDS-UPDRS based) ===
        score = 0

        # Score based on decrement (most specific to PD)
        if velocity_decrement > self.DECREMENT_THRESHOLD_SEVERE:
            score += 3
        elif velocity_decrement > self.DECREMENT_THRESHOLD_MODERATE:
            score += 2
        elif velocity_decrement > self.DECREMENT_THRESHOLD_MILD:
            score += 1

        if amplitude_decrement > self.DECREMENT_THRESHOLD_SEVERE:
            score += 3
        elif amplitude_decrement > self.DECREMENT_THRESHOLD_MODERATE:
            score += 2
        elif amplitude_decrement > self.DECREMENT_THRESHOLD_MILD:
            score += 1

        # Score based on hesitations
        if hesitation_ratio > 0.3:
            score += 2
        elif hesitation_ratio > 0.15:
            score += 1

        # Score based on velocity irregularity
        if velocity_cv > 1.0:
            score += 1

        # Determine severity
        if score >= 6:
            severity = 'severe'
        elif score >= 4:
            severity = 'moderate'
        elif score >= 2:
            severity = 'mild'
        else:
            severity = 'normal'

        return SymptomResult(
            symptom_type='bradykinesia',
            person_id=segment.person_id,
            segment_id=f"brady_movement_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={
                'avg_velocity': float(avg_velocity),
                'max_velocity': float(max_velocity),
                'movement_amplitude': float(avg_amplitude),
                'velocity_decrement': float(velocity_decrement),
                'amplitude_decrement': float(amplitude_decrement),
                'hesitation_ratio': float(hesitation_ratio),
                'velocity_cv': float(velocity_cv),
                'bradykinesia_score': int(score),
            },
            severity=severity,
            confidence=segment.confidence
        )

    def _analyze_facial(self, track: PersonTrack, segment: AnalysisSegment,
                        fps: float) -> SymptomResult:
        """
        Analyze blink rate and facial expression (hypomimia).

        Literature-based thresholds:
        - Normal blink rate: 15-24 blinks/min (Karson et al., 1984)
        - PD blink rate: can be as low as 3-4 blinks/min
        - Severe: < 5 blinks/min
        - Moderate: 5-10 blinks/min
        - Mild: 10-15 blinks/min
        """
        start_rel = segment.start_frame - track.start_frame
        end_rel = segment.end_frame - track.start_frame

        face_landmarks = track.face_landmarks[start_rel:end_rel]

        # Calculate eye aspect ratio (EAR) for blink detection
        ear_values = []
        mouth_movements = []
        valid_frames = 0

        for fl in face_landmarks:
            if fl is not None and np.any(fl != 0):
                valid_frames += 1
                # Eye opening (average of both eyes)
                left_height = np.linalg.norm(fl[self.LEFT_EYE_TOP] - fl[self.LEFT_EYE_BOTTOM])
                right_height = np.linalg.norm(fl[self.RIGHT_EYE_TOP] - fl[self.RIGHT_EYE_BOTTOM])
                ear_values.append((left_height + right_height) / 2)

                # Mouth opening for hypomimia analysis
                mouth_height = np.linalg.norm(fl[self.UPPER_LIP] - fl[self.LOWER_LIP])
                mouth_width = np.linalg.norm(fl[self.LEFT_MOUTH_CORNER] - fl[self.RIGHT_MOUTH_CORNER])
                mouth_movements.append(mouth_height / (mouth_width + 1e-6))

        if len(ear_values) < fps * 2:
            return self._empty_result(segment, 'facial')

        ear_values = np.array(ear_values)
        mouth_movements = np.array(mouth_movements)

        # === BLINK DETECTION ===
        ear_smooth = uniform_filter1d(ear_values, max(3, int(fps * 0.1)))

        # Use adaptive threshold based on signal characteristics
        ear_mean = np.mean(ear_smooth)
        ear_std = np.std(ear_smooth)
        blink_threshold = ear_mean - 0.5 * ear_std  # Blink = significant dip

        # Detect blink events using peak detection on inverted signal
        inverted_ear = -ear_smooth
        blink_peaks, _ = signal.find_peaks(inverted_ear,
                                           height=-blink_threshold,
                                           distance=int(fps * 0.2))  # Min 0.2s between blinks
        n_blinks = len(blink_peaks)

        # Calculate blink rate (per minute)
        duration_minutes = segment.duration / 60
        blink_rate = n_blinks / duration_minutes if duration_minutes > 0 else 0

        # === HYPOMIMIA ANALYSIS ===
        # Measure facial movement variance (less = hypomimia)
        if len(face_landmarks) > 1:
            face_diffs = np.diff(face_landmarks, axis=0)
            # Filter out invalid frames
            valid_diffs = [d for d in face_diffs if d is not None and np.any(d != 0)]
            if len(valid_diffs) > 0:
                face_movement_var = np.mean([np.std(d) for d in valid_diffs])
            else:
                face_movement_var = 0
        else:
            face_movement_var = 0

        # Mouth movement range (reduced in hypomimia)
        mouth_movement_range = np.max(mouth_movements) - np.min(mouth_movements) if len(mouth_movements) > 0 else 0

        # === SEVERITY SCORING ===
        score = 0

        # Blink rate scoring (literature-based)
        if blink_rate < self.PD_BLINK_RATE_SEVERE:  # < 5/min
            score += 4
        elif blink_rate < self.PD_BLINK_RATE_MODERATE:  # < 10/min
            score += 3
        elif blink_rate < self.PD_BLINK_RATE_MILD:  # < 15/min
            score += 2
        elif blink_rate < self.NORMAL_BLINK_RATE_MIN:  # < 15/min (borderline)
            score += 1

        # Facial movement scoring
        if face_movement_var < 0.5:
            score += 2
        elif face_movement_var < 1.0:
            score += 1

        # Mouth movement scoring
        if mouth_movement_range < 0.05:
            score += 2
        elif mouth_movement_range < 0.1:
            score += 1

        # Determine severity
        if score >= 6:
            severity = 'severe'
        elif score >= 4:
            severity = 'moderate'
        elif score >= 2:
            severity = 'mild'
        else:
            severity = 'normal'

        return SymptomResult(
            symptom_type='bradykinesia',
            person_id=segment.person_id,
            segment_id=f"brady_facial_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={
                'blink_rate': float(blink_rate),
                'blink_count': int(n_blinks),
                'facial_movement_var': float(face_movement_var),
                'mouth_movement_range': float(mouth_movement_range),
                'hypomimia_score': int(score),
            },
            severity=severity,
            confidence=segment.confidence
        )

    def _empty_result(self, segment: AnalysisSegment, subtype: str) -> SymptomResult:
        return SymptomResult(
            symptom_type='bradykinesia',
            person_id=segment.person_id,
            segment_id=f"brady_{subtype}_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={},
            severity='unknown',
            confidence=0
        )


class PostureAnalyzer:
    """Analyze posture abnormalities using literature-based criteria.

    Based on consensus diagnostic criteria from:
    - Doherty et al. 2011: Camptocormia diagnostic criteria
    - Tinazzi et al. 2015: Pisa syndrome definition
    - MDS-UPDRS Part III Item 3.13 (Posture)

    Measures:
    - Forward trunk lean angle (stooped posture/camptocormia)
    - Lateral lean (Pisa syndrome)
    - Head drop angle (antecollis/dropped head syndrome)
    - Posture variability (dynamic instability)

    Thresholds from literature:
    - Lower camptocormia (thoracolumbar): ≥30° forward flexion
    - Upper camptocormia (thoracic): ≥45° forward flexion
    - Pisa syndrome: ≥10° sustained lateral trunk deviation
    - Antecollis: >15° neck flexion (moderate), >45° severe
    """

    # Literature-based thresholds (Doherty et al. 2011, Tinazzi et al. 2015)
    NORMAL_TRUNK_ANGLE = 10  # degrees from vertical - normal range
    MILD_STOOPED = 15  # Subtle stooped posture
    MODERATE_STOOPED = 20  # Noticeable stooped posture
    LOWER_CAMPTOCORMIA_THRESHOLD = 30  # Lower (thoracolumbar) camptocormia
    UPPER_CAMPTOCORMIA_THRESHOLD = 45  # Upper (thoracic) camptocormia

    # Pisa syndrome thresholds
    PISA_THRESHOLD = 10  # degrees lateral deviation for Pisa syndrome
    SEVERE_PISA_THRESHOLD = 15  # Severe Pisa syndrome

    # Antecollis thresholds
    NORMAL_HEAD_ANGLE = 10
    MILD_ANTECOLLIS = 15
    MODERATE_ANTECOLLIS = 30
    SEVERE_ANTECOLLIS = 45  # Dropped head syndrome

    def analyze(self, track: PersonTrack, segment: AnalysisSegment,
                fps: float) -> SymptomResult:
        """Analyze posture in the given segment with MDS-UPDRS scoring."""
        start_rel = segment.start_frame - track.start_frame
        end_rel = segment.end_frame - track.start_frame

        # Landmark indices
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_EAR = 7
        RIGHT_EAR = 8

        keypoints = track.keypoints[start_rel:end_rel]

        # =================================================================
        # 1. Forward trunk angle (sagittal plane)
        # =================================================================
        shoulder_center = (keypoints[:, LEFT_SHOULDER, :] + keypoints[:, RIGHT_SHOULDER, :]) / 2
        hip_center = (keypoints[:, LEFT_HIP, :] + keypoints[:, RIGHT_HIP, :]) / 2

        # Trunk vector (hip to shoulder)
        trunk_vector = shoulder_center - hip_center

        # Forward lean angle (from vertical)
        # Vertical is [0, -1] in image coordinates (Y decreases upward)
        vertical = np.array([0, -1])

        trunk_angles = []
        for tv in trunk_vector[:, :2]:
            tv_norm = tv / (np.linalg.norm(tv) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(np.dot(tv_norm, vertical), -1, 1)))
            trunk_angles.append(angle)

        trunk_angles = np.array(trunk_angles)
        avg_trunk_angle = np.mean(trunk_angles)
        trunk_angle_std = np.std(trunk_angles)
        max_trunk_angle = np.max(trunk_angles)

        # =================================================================
        # 2. Lateral trunk deviation (Pisa syndrome - coronal plane)
        # =================================================================
        # Calculate lateral angle using shoulder-hip vectors
        lateral_angles = []
        for i in range(len(keypoints)):
            # Shoulder center to hip center in frontal view
            left_vec = keypoints[i, LEFT_HIP, :2] - keypoints[i, LEFT_SHOULDER, :2]
            right_vec = keypoints[i, RIGHT_HIP, :2] - keypoints[i, RIGHT_SHOULDER, :2]

            # Use X-coordinate difference to estimate lateral lean
            shoulder_x_center = (keypoints[i, LEFT_SHOULDER, 0] + keypoints[i, RIGHT_SHOULDER, 0]) / 2
            hip_x_center = (keypoints[i, LEFT_HIP, 0] + keypoints[i, RIGHT_HIP, 0]) / 2

            # Estimate height for angle calculation
            trunk_height = np.linalg.norm(trunk_vector[i, :2])
            lateral_offset = shoulder_x_center - hip_x_center

            if trunk_height > 0:
                lateral_angle = np.degrees(np.arctan(lateral_offset / trunk_height))
                lateral_angles.append(lateral_angle)

        lateral_angles = np.array(lateral_angles) if lateral_angles else np.array([0])
        avg_lateral_angle = np.mean(np.abs(lateral_angles))
        lateral_direction = 'left' if np.mean(lateral_angles) > 0 else 'right'
        lateral_consistency = 1.0 - (np.std(lateral_angles) / (np.mean(np.abs(lateral_angles)) + 1e-6))
        lateral_consistency = max(0, min(1, lateral_consistency))

        # Pisa syndrome requires SUSTAINED deviation (consistency check)
        has_pisa_syndrome = avg_lateral_angle >= self.PISA_THRESHOLD and lateral_consistency > 0.7

        # =================================================================
        # 3. Head drop angle (antecollis)
        # =================================================================
        ear_center = (keypoints[:, LEFT_EAR, :] + keypoints[:, RIGHT_EAR, :]) / 2
        head_vector = ear_center - shoulder_center

        head_angles = []
        for hv in head_vector[:, :2]:
            hv_norm = hv / (np.linalg.norm(hv) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(np.dot(hv_norm, vertical), -1, 1)))
            head_angles.append(angle)

        head_angles = np.array(head_angles)
        avg_head_angle = np.mean(head_angles)
        max_head_angle = np.max(head_angles)

        # =================================================================
        # 4. Posture variability (dynamic instability)
        # =================================================================
        # High variability indicates postural instability
        posture_variability = trunk_angle_std / (avg_trunk_angle + 1e-6)

        # Coefficient of variation for postural sway
        if len(trunk_angles) > 10:
            # Calculate sway as frame-to-frame angle changes
            angle_changes = np.abs(np.diff(trunk_angles))
            sway_index = np.mean(angle_changes) * fps  # degrees per second
        else:
            sway_index = 0

        # =================================================================
        # 5. MDS-UPDRS based scoring (Item 3.13 Posture)
        # =================================================================
        # Score 0: Normal
        # Score 1: Not quite erect, but posture could be normal for older person
        # Score 2: Definite stooped posture, but could be due to other causes
        # Score 3: Stooped posture with moderate scoliosis or lateral lean
        # Score 4: Severe flexion with extreme abnormality of posture

        updrs_score = 0
        severity_reasons = []

        # Forward flexion scoring
        if avg_trunk_angle >= self.UPPER_CAMPTOCORMIA_THRESHOLD:
            updrs_score = max(updrs_score, 4)
            severity_reasons.append(f'upper_camptocormia_{avg_trunk_angle:.1f}deg')
        elif avg_trunk_angle >= self.LOWER_CAMPTOCORMIA_THRESHOLD:
            updrs_score = max(updrs_score, 3)
            severity_reasons.append(f'lower_camptocormia_{avg_trunk_angle:.1f}deg')
        elif avg_trunk_angle >= self.MODERATE_STOOPED:
            updrs_score = max(updrs_score, 2)
            severity_reasons.append(f'stooped_{avg_trunk_angle:.1f}deg')
        elif avg_trunk_angle >= self.MILD_STOOPED:
            updrs_score = max(updrs_score, 1)
            severity_reasons.append(f'mild_stooped_{avg_trunk_angle:.1f}deg')

        # Lateral lean scoring (adds to severity)
        if has_pisa_syndrome:
            if avg_lateral_angle >= self.SEVERE_PISA_THRESHOLD:
                updrs_score = max(updrs_score, 4)
                severity_reasons.append(f'severe_pisa_{avg_lateral_angle:.1f}deg_{lateral_direction}')
            else:
                updrs_score = max(updrs_score, 3)
                severity_reasons.append(f'pisa_{avg_lateral_angle:.1f}deg_{lateral_direction}')

        # Head drop scoring
        if avg_head_angle >= self.SEVERE_ANTECOLLIS:
            updrs_score = max(updrs_score, 4)
            severity_reasons.append(f'dropped_head_{avg_head_angle:.1f}deg')
        elif avg_head_angle >= self.MODERATE_ANTECOLLIS:
            updrs_score = max(updrs_score, 3)
            severity_reasons.append(f'moderate_antecollis_{avg_head_angle:.1f}deg')
        elif avg_head_angle >= self.MILD_ANTECOLLIS:
            updrs_score = max(updrs_score, 2)
            severity_reasons.append(f'mild_antecollis_{avg_head_angle:.1f}deg')

        # Determine overall severity from score
        if updrs_score >= 4:
            severity = 'severe'
        elif updrs_score >= 3:
            severity = 'moderate'
        elif updrs_score >= 2:
            severity = 'mild'
        elif updrs_score >= 1:
            severity = 'slight'
        else:
            severity = 'normal'

        return SymptomResult(
            symptom_type='posture',
            person_id=segment.person_id,
            segment_id=f"posture_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={
                'trunk_forward_angle': float(avg_trunk_angle),
                'trunk_forward_angle_max': float(max_trunk_angle),
                'trunk_angle_std': float(trunk_angle_std),
                'lateral_angle': float(avg_lateral_angle),
                'lateral_direction': lateral_direction,
                'lateral_consistency': float(lateral_consistency),
                'has_pisa_syndrome': has_pisa_syndrome,
                'head_drop_angle': float(avg_head_angle),
                'head_drop_angle_max': float(max_head_angle),
                'posture_variability': float(posture_variability),
                'sway_index': float(sway_index),
                'updrs_posture_score': updrs_score,
                'severity_reasons': severity_reasons,
            },
            severity=severity,
            confidence=segment.confidence
        )


class FOGTransitionDetector:
    """
    Detect Freezing of Gait at transition points between standing and walking.

    FOG typically occurs at:
    - Gait initiation (standing → walking)
    - Gait termination (walking → standing)
    - Direction changes (turning)

    This detector finds boundaries between segments and creates transition zones
    for FOG analysis.
    """

    # Landmark indices
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self,
                 transition_window: float = 2.0,  # seconds before/after transition
                 min_transition_duration: float = 0.5):
        """
        Args:
            transition_window: Time window around transition to analyze
            min_transition_duration: Minimum duration for a transition zone
        """
        self.transition_window = transition_window
        self.min_transition_duration = min_transition_duration

    def detect_transitions(self, activities: List[ActivitySegment],
                           track: PersonTrack, fps: float) -> List[AnalysisSegment]:
        """
        Detect FOG transition zones at standing↔walking boundaries.

        Returns:
            List of AnalysisSegment objects for FOG transition analysis
        """
        transitions = []

        # Sort activities by start time
        sorted_activities = sorted(activities, key=lambda a: a.start_time)

        for i in range(len(sorted_activities) - 1):
            current = sorted_activities[i]
            next_act = sorted_activities[i + 1]

            # Check for standing↔walking transition
            is_fog_transition = (
                (current.activity_type == 'standing' and next_act.activity_type == 'walking') or
                (current.activity_type == 'walking' and next_act.activity_type == 'standing')
            )

            if is_fog_transition:
                # Create transition zone around the boundary
                boundary_time = current.end_time
                transition_start = max(current.start_time, boundary_time - self.transition_window)
                transition_end = min(next_act.end_time, boundary_time + self.transition_window)

                # Determine transition type
                if current.activity_type == 'standing':
                    transition_type = 'gait_initiation'  # 보행 시작
                else:
                    transition_type = 'gait_termination'  # 보행 종료

                transition_start_frame = int(transition_start * fps)
                transition_end_frame = int(transition_end * fps)

                transitions.append(AnalysisSegment(
                    segment_type='fog_transition',
                    person_id=current.person_id,
                    start_frame=transition_start_frame,
                    end_frame=transition_end_frame,
                    start_time=transition_start,
                    end_time=transition_end,
                    confidence=min(current.confidence, next_act.confidence),
                    metadata={
                        'transition_type': transition_type,
                        'boundary_time': boundary_time,
                        'from_activity': current.activity_type,
                        'to_activity': next_act.activity_type,
                        # Keep full adjacent activity ranges for richer transition-context analysis.
                        'current_start_frame': current.start_frame,
                        'current_end_frame': current.end_frame,
                        'next_start_frame': next_act.start_frame,
                        'next_end_frame': next_act.end_frame,
                    }
                ))

        return transitions


class FOGTransitionAnalyzer:
    """
    Analyze Freezing of Gait at transition points (standing ↔ walking).

    Integrated analysis combining transition-specific and general FOG biomarkers.

    Key biomarkers:
    1. Hesitation time - delay before first step
    2. Start hesitation ratio - time spent hesitating vs moving
    3. Festination - rapid small shuffling steps
    4. Trembling - high frequency leg oscillations (3-8 Hz)
    5. Akinesia - complete motor block
    6. Freeze Index (FI) - Moore et al. 2008: power(3-8Hz) / power(0.5-3Hz)
    7. Step asymmetry - bilateral coordination deficit
    8. Cadence - steps per minute

    FOG typically occurs at:
    - Gait initiation (standing → walking)
    - Gait termination (walking → standing)
    - Direction changes / turning
    """

    # Landmark indices
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    # Thresholds for transition analysis
    HESITATION_VELOCITY_THRESHOLD = 15.0  # pixels/sec - below this = hesitating
    FESTINATION_FREQ_RANGE = (3.0, 8.0)  # Hz - shuffling/trembling frequency
    NORMAL_STEP_FREQ_RANGE = (0.8, 2.5)  # Hz - normal stepping frequency

    # Freeze Index thresholds (Moore et al. 2008)
    FREEZE_BAND = (3.0, 8.0)  # Hz - high frequency trembling
    LOCOMOTOR_BAND = (0.5, 3.0)  # Hz - normal gait cadence
    FI_NORMAL = 1.0
    FI_MILD = 2.0
    FI_MODERATE = 3.0
    FI_SEVERE = 5.0

    # Step asymmetry threshold (Plotnik et al. 2005)
    NORMAL_ASYMMETRY = 0.05
    ABNORMAL_ASYMMETRY = 0.10

    def analyze(self, track: PersonTrack, segment: AnalysisSegment,
                fps: float) -> SymptomResult:
        """Analyze FOG at a transition point with integrated biomarkers."""
        start_rel = segment.start_frame - track.start_frame
        end_rel = segment.end_frame - track.start_frame

        # Ensure valid range
        start_rel = max(0, start_rel)
        end_rel = min(track.duration_frames, end_rel)

        if end_rel - start_rel < fps * 0.5:
            return self._empty_result(segment)

        keypoints = track.keypoints[start_rel:end_rel]

        # Build a wider context from the full adjacent standing/walking activities.
        context_start = segment.metadata.get('current_start_frame', segment.start_frame)
        context_end = segment.metadata.get('next_end_frame', segment.end_frame)
        context_start_rel = max(0, context_start - track.start_frame)
        context_end_rel = min(track.duration_frames, context_end - track.start_frame)
        keypoints_context = track.keypoints[context_start_rel:context_end_rel]
        if len(keypoints_context) < 2:
            keypoints_context = keypoints

        # Get boundary info
        boundary_time = segment.metadata.get('boundary_time', segment.start_time)
        transition_type = segment.metadata.get('transition_type', 'unknown')
        boundary_frame_rel = int((boundary_time - segment.start_time) * fps)
        boundary_frame_rel = max(0, min(len(keypoints) - 1, boundary_frame_rel))

        # Calculate velocities
        ankle_left = keypoints[:, self.LEFT_ANKLE, :2]
        ankle_right = keypoints[:, self.RIGHT_ANKLE, :2]
        hip_center = (keypoints[:, self.LEFT_HIP, :2] + keypoints[:, self.RIGHT_HIP, :2]) / 2

        vel_ankle_left = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1)) * fps
        vel_ankle_right = np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1)) * fps
        vel_ankle_avg = (vel_ankle_left + vel_ankle_right) / 2
        vel_hip = np.sqrt(np.sum(np.diff(hip_center, axis=0)**2, axis=1)) * fps

        # Pad velocities to match keypoints length
        vel_ankle_avg_padded = np.concatenate([[0], vel_ankle_avg])
        vel_hip_padded = np.concatenate([[0], vel_hip])

        # =================================================================
        # 1. Transition-specific analysis
        # =================================================================
        # Hesitation analysis
        hesitation_metrics = self._analyze_hesitation(
            vel_ankle_avg_padded, vel_hip_padded, boundary_frame_rel, transition_type, fps
        )

        # Festination/Trembling analysis (high frequency small movements)
        festination_metrics = self._analyze_festination(
            vel_ankle_left, vel_ankle_right, fps
        )

        # Motor block (akinesia) analysis
        akinesia_metrics = self._analyze_akinesia(
            vel_ankle_avg_padded, vel_hip_padded, fps
        )

        # Step initiation delay
        initiation_delay = self._calculate_initiation_delay(
            vel_ankle_avg_padded, boundary_frame_rel, transition_type, fps
        )

        # =================================================================
        # 2. Freeze Index calculation (Moore et al. 2008)
        # =================================================================
        freeze_index_metrics = self._calculate_freeze_index(keypoints, fps)

        # =================================================================
        # 3. Step asymmetry/cadence from walking-only context
        # =================================================================
        walking_keypoints = self._extract_walking_keypoints(track, segment)
        gait_metrics = self._analyze_gait_pattern(walking_keypoints, fps)

        # Freeze ratio from transition context, with adaptive thresholding from walking baseline
        baseline_ankle_velocity = self._estimate_baseline_ankle_velocity(walking_keypoints, fps)
        freeze_ratio = self._calculate_freeze_ratio(
            keypoints_context,
            fps,
            transition_type=transition_type,
            baseline_ankle_velocity=baseline_ankle_velocity,
        )

        # =================================================================
        # 4. Combine all metrics
        # =================================================================
        metrics = {
            **hesitation_metrics,
            **festination_metrics,
            **akinesia_metrics,
            **freeze_index_metrics,
            **gait_metrics,
            'freeze_ratio': freeze_ratio,
            'initiation_delay_sec': initiation_delay,
            'transition_type': transition_type,
        }

        # Determine severity based on combined indicators
        severity, severity_reasons = self._determine_severity(metrics)
        metrics['severity_reasons'] = severity_reasons

        return SymptomResult(
            symptom_type='fog',  # Changed from 'fog_transition' to unified 'fog'
            person_id=segment.person_id,
            segment_id=f"fog_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics=metrics,
            severity=severity,
            confidence=segment.confidence
        )

    def _calculate_freeze_index(self, keypoints: np.ndarray, fps: float) -> Dict:
        """Calculate Freeze Index using FFT (Moore et al. 2008)."""
        # Use vertical ankle movement for FFT analysis
        ankle_vertical = (keypoints[:, self.LEFT_ANKLE, 1] + keypoints[:, self.RIGHT_ANKLE, 1]) / 2

        freeze_index = 0.0
        freeze_band_power = 0.0
        locomotor_band_power = 0.0

        if len(ankle_vertical) > int(fps * 1):  # Need at least 1 second
            # Detrend the signal
            ankle_detrended = ankle_vertical - np.mean(ankle_vertical)

            # Apply Hanning window
            window = np.hanning(len(ankle_detrended))
            ankle_windowed = ankle_detrended * window

            # FFT
            n = len(ankle_windowed)
            yf = np.abs(fft(ankle_windowed))[:n // 2]
            xf = fftfreq(n, 1/fps)[:n // 2]

            # Calculate power in freeze band (3-8 Hz)
            freeze_mask = (xf >= self.FREEZE_BAND[0]) & (xf <= self.FREEZE_BAND[1])
            freeze_band_power = np.sum(yf[freeze_mask]**2)

            # Calculate power in locomotor band (0.5-3 Hz)
            locomotor_mask = (xf >= self.LOCOMOTOR_BAND[0]) & (xf < self.LOCOMOTOR_BAND[1])
            locomotor_band_power = np.sum(yf[locomotor_mask]**2)

            # Freeze Index
            freeze_index = freeze_band_power / (locomotor_band_power + 1e-6)

        return {
            'freeze_index': float(freeze_index),
            'freeze_band_power': float(freeze_band_power),
            'locomotor_band_power': float(locomotor_band_power),
        }

    def _analyze_gait_pattern(self, keypoints: np.ndarray, fps: float) -> Dict:
        """Analyze step asymmetry, cadence, and stride CV from walking-only keypoints."""
        cadence = 0.0
        step_asymmetry = 0.0
        step_count = 0
        stride_cv = 0.0

        if keypoints is None or len(keypoints) < int(fps * 0.8):
            return {
                'cadence': float(cadence),
                'step_asymmetry': float(step_asymmetry),
                'step_count': int(step_count),
                'stride_cv': float(stride_cv),
            }

        ankle_left = keypoints[:, self.LEFT_ANKLE, :2]
        ankle_right = keypoints[:, self.RIGHT_ANKLE, :2]
        vel_left = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1)) * fps
        vel_right = np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1)) * fps
        combined_vel = (vel_left + vel_right) / 2

        if len(combined_vel) > int(fps * 0.5):
            try:
                # Find peaks in ankle velocity (each peak = step)
                min_step_distance = int(fps * 0.35)  # More conservative to avoid overcount
                peak_height = np.percentile(combined_vel, 50) if len(combined_vel) > 0 else 0
                peak_prominence = max(1.0, np.std(combined_vel) * 0.35)
                step_indices, _ = signal.find_peaks(
                    combined_vel,
                    distance=min_step_distance,
                    height=peak_height,
                    prominence=peak_prominence,
                )

                step_count = len(step_indices)

                if step_count > 1:
                    # Calculate cadence (steps per minute)
                    duration_sec = len(combined_vel) / fps
                    cadence = (step_count / duration_sec) * 60
                    cadence = float(np.clip(cadence, 30, 180))

                    # Stride time variability (2 steps = 1 stride)
                    step_intervals = np.diff(step_indices) / fps
                    step_intervals = step_intervals[(step_intervals > 0.25) & (step_intervals < 1.5)]
                    stride_times = []
                    for i in range(0, len(step_intervals) - 1, 2):
                        stride_times.append(step_intervals[i] + step_intervals[i + 1])
                    if len(stride_times) < 2:
                        stride_times = step_intervals
                    if len(stride_times) > 1:
                        stride_times = np.asarray(stride_times)
                        stride_cv = float(
                            (np.std(stride_times, ddof=1) / (np.mean(stride_times) + 1e-6)) * 100
                        )

                # Calculate step asymmetry
                left_total = np.sum(vel_left)
                right_total = np.sum(vel_right)
                total = left_total + right_total
                if total > 0:
                    step_asymmetry = abs(left_total - right_total) / total

            except Exception:
                pass

        return {
            'cadence': float(cadence),
            'step_asymmetry': float(step_asymmetry),
            'step_count': int(step_count),
            'stride_cv': float(stride_cv),
        }

    def _calculate_freeze_ratio(
        self,
        keypoints_context: np.ndarray,
        fps: float,
        transition_type: str,
        baseline_ankle_velocity: float,
    ) -> float:
        """Estimate freeze ratio with adaptive threshold over transition context."""
        if len(keypoints_context) < 3:
            return 0.0

        ankle_left = keypoints_context[:, self.LEFT_ANKLE, :2]
        ankle_right = keypoints_context[:, self.RIGHT_ANKLE, :2]
        hip_center = (keypoints_context[:, self.LEFT_HIP, :2] + keypoints_context[:, self.RIGHT_HIP, :2]) / 2

        vel_ankle_left = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1)) * fps
        vel_ankle_right = np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1)) * fps
        vel_ankle = (vel_ankle_left + vel_ankle_right) / 2
        vel_hip = np.sqrt(np.sum(np.diff(hip_center, axis=0)**2, axis=1)) * fps

        adaptive_freeze_threshold = max(
            self.HESITATION_VELOCITY_THRESHOLD * 0.5,
            baseline_ankle_velocity * 0.30,
        )
        intent_threshold = self.HESITATION_VELOCITY_THRESHOLD * 0.25
        low_ankle = vel_ankle < adaptive_freeze_threshold
        active_or_attempting = vel_hip > intent_threshold

        if transition_type == 'gait_initiation':
            # During initiation, evaluate when there is intent to move.
            evaluable = active_or_attempting
            freeze_like = low_ankle & active_or_attempting
        else:
            # During termination, avoid counting natural standstill as freezing.
            decel_zone = vel_hip > intent_threshold
            evaluable = decel_zone
            freeze_like = low_ankle & decel_zone

        if not np.any(evaluable):
            return 0.0
        return float(np.mean(freeze_like[evaluable]))

    def _extract_walking_keypoints(self, track: PersonTrack, segment: AnalysisSegment) -> np.ndarray:
        """Extract keypoints only for the walking-side activity adjacent to this transition."""
        current_start = segment.metadata.get('current_start_frame')
        current_end = segment.metadata.get('current_end_frame')
        next_start = segment.metadata.get('next_start_frame')
        next_end = segment.metadata.get('next_end_frame')
        from_activity = segment.metadata.get('from_activity')
        to_activity = segment.metadata.get('to_activity')

        if from_activity == 'walking' and current_start is not None and current_end is not None:
            start_rel = max(0, int(current_start - track.start_frame))
            end_rel = min(track.duration_frames, int(current_end - track.start_frame))
            return track.keypoints[start_rel:end_rel]
        if to_activity == 'walking' and next_start is not None and next_end is not None:
            start_rel = max(0, int(next_start - track.start_frame))
            end_rel = min(track.duration_frames, int(next_end - track.start_frame))
            return track.keypoints[start_rel:end_rel]

        # Fallback to transition window only
        start_rel = max(0, int(segment.start_frame - track.start_frame))
        end_rel = min(track.duration_frames, int(segment.end_frame - track.start_frame))
        return track.keypoints[start_rel:end_rel]

    def _estimate_baseline_ankle_velocity(self, keypoints: np.ndarray, fps: float) -> float:
        """Estimate walking baseline ankle velocity for adaptive freeze thresholding."""
        if keypoints is None or len(keypoints) < 3:
            return self.HESITATION_VELOCITY_THRESHOLD

        ankle_left = keypoints[:, self.LEFT_ANKLE, :2]
        ankle_right = keypoints[:, self.RIGHT_ANKLE, :2]
        vel_left = np.sqrt(np.sum(np.diff(ankle_left, axis=0)**2, axis=1)) * fps
        vel_right = np.sqrt(np.sum(np.diff(ankle_right, axis=0)**2, axis=1)) * fps
        vel = (vel_left + vel_right) / 2
        return float(np.percentile(vel, 60))

    def _analyze_hesitation(self, vel_ankle: np.ndarray, vel_hip: np.ndarray,
                            boundary_frame: int, transition_type: str,
                            fps: float) -> Dict:
        """Analyze hesitation around the transition boundary."""
        n_frames = len(vel_ankle)

        if transition_type == 'gait_initiation':
            # For gait initiation: analyze frames AFTER boundary
            analysis_start = boundary_frame
            analysis_end = n_frames
        else:
            # For gait termination: analyze frames BEFORE boundary
            analysis_start = 0
            analysis_end = boundary_frame

        if analysis_end <= analysis_start:
            return {'hesitation_ratio': 0.0, 'hesitation_duration_sec': 0.0}

        segment_ankle = vel_ankle[analysis_start:analysis_end]
        segment_hip = vel_hip[analysis_start:analysis_end]

        # Hesitation: hip trying to move but ankles not moving
        hip_active = segment_hip > self.HESITATION_VELOCITY_THRESHOLD
        ankle_frozen = segment_ankle < self.HESITATION_VELOCITY_THRESHOLD

        hesitation_frames = hip_active & ankle_frozen
        hesitation_ratio = np.mean(hesitation_frames) if len(hesitation_frames) > 0 else 0

        return {
            'hesitation_ratio': float(hesitation_ratio),
            'hesitation_duration_sec': float(np.sum(hesitation_frames) / fps)
        }

    def _analyze_festination(self, vel_left: np.ndarray, vel_right: np.ndarray,
                              fps: float) -> Dict:
        """Analyze festination (shuffling) using frequency analysis."""
        if len(vel_left) < fps:
            return {'festination_power': 0.0, 'trembling_frequency': 0.0}

        # Combined ankle movement
        ankle_movement = (vel_left + vel_right) / 2

        # FFT analysis
        n = len(ankle_movement)
        yf = np.abs(fft(ankle_movement - np.mean(ankle_movement)))
        xf = fftfreq(n, 1/fps)

        # Positive frequencies only
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = yf[pos_mask]

        if len(yf_pos) == 0:
            return {'festination_power': 0.0, 'trembling_frequency': 0.0}

        # Power in festination/trembling range (3-8 Hz)
        fest_mask = (xf_pos >= self.FESTINATION_FREQ_RANGE[0]) & \
                   (xf_pos <= self.FESTINATION_FREQ_RANGE[1])
        festination_power = np.sum(yf_pos[fest_mask]) if np.any(fest_mask) else 0
        total_power = np.sum(yf_pos) + 1e-6

        # Dominant frequency in festination range
        if np.any(fest_mask) and festination_power > 0:
            fest_freqs = xf_pos[fest_mask]
            fest_powers = yf_pos[fest_mask]
            dominant_fest_freq = fest_freqs[np.argmax(fest_powers)]
        else:
            dominant_fest_freq = 0

        return {
            'festination_power': float(festination_power / total_power),
            'trembling_frequency': float(dominant_fest_freq)
        }

    def _analyze_akinesia(self, vel_ankle: np.ndarray, vel_hip: np.ndarray,
                          fps: float) -> Dict:
        """Analyze akinesia (complete motor block)."""
        # Akinesia: both hip and ankles frozen
        very_low_threshold = 5.0  # Very low velocity
        hip_frozen = vel_hip < very_low_threshold
        ankle_frozen = vel_ankle < very_low_threshold
        complete_freeze = hip_frozen & ankle_frozen

        freeze_ratio = np.mean(complete_freeze) if len(complete_freeze) > 0 else 0
        max_freeze_duration = self._max_consecutive(complete_freeze) / fps

        return {
            'akinesia_ratio': float(freeze_ratio),
            'max_freeze_duration_sec': float(max_freeze_duration)
        }

    def _calculate_initiation_delay(self, vel_ankle: np.ndarray,
                                     boundary_frame: int, transition_type: str,
                                     fps: float) -> float:
        """Calculate delay before first successful step."""
        if transition_type != 'gait_initiation':
            return 0.0

        # Look for first frame after boundary where ankle velocity exceeds threshold
        walking_threshold = 30.0  # Clear walking velocity
        post_boundary = vel_ankle[boundary_frame:]

        if len(post_boundary) == 0:
            return 0.0

        walking_frames = np.where(post_boundary > walking_threshold)[0]

        if len(walking_frames) == 0:
            return float(len(post_boundary) / fps)  # Never started walking
        else:
            return float(walking_frames[0] / fps)

    def _max_consecutive(self, bool_array: np.ndarray) -> int:
        """Find maximum consecutive True values."""
        if len(bool_array) == 0:
            return 0

        max_count = 0
        current_count = 0

        for val in bool_array:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _determine_severity(self, metrics: Dict) -> tuple:
        """Determine FOG severity based on combined metrics."""
        hesitation_ratio = metrics.get('hesitation_ratio', 0)
        festination_power = metrics.get('festination_power', 0)
        akinesia_ratio = metrics.get('akinesia_ratio', 0)
        initiation_delay = metrics.get('initiation_delay_sec', 0)
        max_freeze = metrics.get('max_freeze_duration_sec', 0)
        freeze_index = metrics.get('freeze_index', 0)
        step_asymmetry = metrics.get('step_asymmetry', 0)

        # Scoring based on severity indicators
        score = 0
        severity_reasons = []

        # Freeze Index scoring (Moore et al. 2008) - PRIMARY INDICATOR
        if freeze_index >= self.FI_SEVERE:
            score += 4
            severity_reasons.append(f'severe_FI_{freeze_index:.2f}')
        elif freeze_index >= self.FI_MODERATE:
            score += 3
            severity_reasons.append(f'moderate_FI_{freeze_index:.2f}')
        elif freeze_index >= self.FI_MILD:
            score += 2
            severity_reasons.append(f'mild_FI_{freeze_index:.2f}')
        elif freeze_index >= self.FI_NORMAL:
            score += 1

        # Hesitation scoring
        if hesitation_ratio > 0.5:
            score += 3
            severity_reasons.append(f'high_hesitation_{hesitation_ratio:.0%}')
        elif hesitation_ratio > 0.3:
            score += 2
            severity_reasons.append(f'hesitation_{hesitation_ratio:.0%}')
        elif hesitation_ratio > 0.1:
            score += 1

        # Festination scoring
        if festination_power > 0.3:
            score += 3
            severity_reasons.append(f'festination_{festination_power:.0%}')
        elif festination_power > 0.15:
            score += 2
            severity_reasons.append(f'festination_{festination_power:.0%}')
        elif festination_power > 0.05:
            score += 1

        # Akinesia scoring
        if akinesia_ratio > 0.3 or max_freeze > 2.0:
            score += 3
            severity_reasons.append(f'akinesia_{akinesia_ratio:.0%}')
        elif akinesia_ratio > 0.15 or max_freeze > 1.0:
            score += 2
            severity_reasons.append(f'akinesia_{akinesia_ratio:.0%}')
        elif akinesia_ratio > 0.05 or max_freeze > 0.5:
            score += 1

        # Initiation delay scoring
        if initiation_delay > 3.0:
            score += 3
            severity_reasons.append(f'init_delay_{initiation_delay:.1f}s')
        elif initiation_delay > 1.5:
            score += 2
            severity_reasons.append(f'init_delay_{initiation_delay:.1f}s')
        elif initiation_delay > 0.5:
            score += 1

        # Step asymmetry scoring
        if step_asymmetry >= self.ABNORMAL_ASYMMETRY:
            score += 1
            severity_reasons.append(f'asymmetry_{step_asymmetry:.0%}')

        # Determine severity
        if score >= 10:
            severity = 'severe'
        elif score >= 6:
            severity = 'moderate'
        elif score >= 3:
            severity = 'mild'
        else:
            severity = 'normal'

        return severity, severity_reasons

    def _empty_result(self, segment: AnalysisSegment) -> SymptomResult:
        return SymptomResult(
            symptom_type='fog_transition',
            person_id=segment.person_id,
            segment_id=f"fog_trans_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={},
            severity='unknown',
            confidence=0
        )


# =============================================================================
# STATISTICAL AGGREGATOR
# =============================================================================

class SymptomStatisticalAggregator:
    """Aggregate symptom results across segments with statistical analysis."""

    def aggregate(self, results: List[SymptomResult]) -> PersonSymptomSummary:
        """Aggregate results for a single person and symptom type."""
        if not results:
            return self._empty_summary()

        person_id = results[0].person_id
        symptom_type = results[0].symptom_type

        # Collect all metric values
        all_metrics: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            for metric_name, value in r.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    all_metrics[metric_name].append(value)

        # Calculate statistics for each metric
        metrics_stats = {}
        for metric_name, values in all_metrics.items():
            if len(values) >= 2:
                arr = np.array(values)
                mean_val = np.mean(arr)
                std_val = np.std(arr, ddof=1)

                # 95% CI
                n = len(arr)
                se = std_val / np.sqrt(n)
                t_val = stats.t.ppf(0.975, n - 1)
                ci_lower = mean_val - t_val * se
                ci_upper = mean_val + t_val * se

                metrics_stats[metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'n': n,
                    'values': [float(v) for v in values]
                }
            elif len(values) == 1:
                metrics_stats[metric_name] = {
                    'mean': float(values[0]),
                    'std': 0.0,
                    'ci_lower': float(values[0]),
                    'ci_upper': float(values[0]),
                    'n': 1,
                    'values': [float(values[0])]
                }

        # Overall severity (mode of individual severities)
        severities = [r.severity for r in results if r.severity != 'unknown']
        if severities:
            severity_counts = defaultdict(int)
            for s in severities:
                severity_counts[s] += 1
            overall_severity = max(severity_counts.items(), key=lambda x: x[1])[0]
        else:
            overall_severity = 'unknown'

        # Average confidence
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0

        return PersonSymptomSummary(
            person_id=person_id,
            symptom_type=symptom_type,
            n_segments=len(results),
            total_duration=sum(r.duration for r in results),
            metrics_stats=metrics_stats,
            overall_severity=overall_severity,
            confidence=float(avg_confidence)
        )

    def _empty_summary(self) -> PersonSymptomSummary:
        return PersonSymptomSummary(
            person_id='unknown',
            symptom_type='unknown',
            n_segments=0,
            total_duration=0,
            metrics_stats={},
            overall_severity='unknown',
            confidence=0
        )


# =============================================================================
# GAIT ANALYZER (BASELINE FEATURES)
# =============================================================================

class GaitAnalyzer:
    """Analyze gait segments using baseline-style handcrafted features."""

    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def analyze(self, track: PersonTrack, segment: AnalysisSegment, fps: float) -> SymptomResult:
        start_rel = max(0, segment.start_frame - track.start_frame)
        end_rel = min(track.duration_frames, segment.end_frame - track.start_frame)

        if end_rel - start_rel < max(8, int(fps * 0.5)):
            return self._empty_result(segment)

        keypoints = track.keypoints[start_rel:end_rel]
        metrics = self._extract_features(keypoints, fps)
        severity = self._classify_severity(metrics)

        return SymptomResult(
            symptom_type='gait',
            person_id=segment.person_id,
            segment_id=f"gait_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics=metrics,
            severity=severity,
            confidence=segment.confidence
        )

    def _extract_features(self, keypoints: np.ndarray, fps: float) -> Dict[str, float]:
        hip_center = (keypoints[:, self.LEFT_HIP, :2] + keypoints[:, self.RIGHT_HIP, :2]) / 2
        trans = np.column_stack([hip_center[:, 0], hip_center[:, 1], np.zeros(len(hip_center))])

        trans_diff = np.diff(trans, axis=0)
        velocity = trans_diff * fps
        speed = np.linalg.norm(velocity, axis=1)
        speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

        if len(speed) > 0:
            speed_mean = float(np.mean(speed))
            speed_std = float(np.std(speed))
            speed_max = float(np.max(speed))
            speed_min = float(np.min(speed))
        else:
            speed_mean = speed_std = speed_max = speed_min = 0.0

        left_knee = keypoints[:, self.LEFT_KNEE, :2]
        right_knee = keypoints[:, self.RIGHT_KNEE, :2]
        left_ankle = keypoints[:, self.LEFT_ANKLE, :2]
        right_ankle = keypoints[:, self.RIGHT_ANKLE, :2]

        knee_asymmetry = float(np.mean(np.linalg.norm(left_knee - right_knee, axis=1)))
        ankle_asymmetry = float(np.mean(np.linalg.norm(left_ankle - right_ankle, axis=1)))
        hip_asymmetry = float(np.mean(np.linalg.norm(
            keypoints[:, self.LEFT_HIP, :2] - keypoints[:, self.RIGHT_HIP, :2], axis=1
        )))

        gait_frequency = 0.0
        gait_regularity = 0.0
        if len(trans) > 10:
            pelvis_signal = trans[:, 1] - np.mean(trans[:, 1])
            fft_vals = np.fft.fft(pelvis_signal)
            freqs = np.fft.fftfreq(len(pelvis_signal), d=1 / fps)
            pos_mask = (freqs > 0.3) & (freqs < 3.0)
            if np.any(pos_mask):
                mags = np.abs(fft_vals)[pos_mask]
                gait_frequency = float(freqs[pos_mask][np.argmax(mags)])
                gait_regularity = float(np.max(mags) / (np.mean(mags) + 1e-6))

        accel_mean = 0.0
        accel_std = 0.0
        if len(velocity) > 1:
            accel = np.diff(velocity, axis=0) * fps
            accel_mag = np.linalg.norm(accel, axis=1)
            accel_mean = float(np.mean(accel_mag))
            accel_std = float(np.std(accel_mag))

        jerk_mean = 0.0
        if len(velocity) > 2:
            jerk = np.diff(np.diff(velocity, axis=0), axis=0) * fps * fps
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_mean = float(np.mean(jerk_mag))

        step_distance = np.linalg.norm(left_ankle - right_ankle, axis=1)
        stride_cv = float((np.std(step_distance) / (np.mean(step_distance) + 1e-6)) * 100)

        return {
            'speed_mean': speed_mean,
            'speed_std': speed_std,
            'speed_max': speed_max,
            'speed_min': speed_min,
            'hip_asymmetry': hip_asymmetry,
            'knee_asymmetry': knee_asymmetry,
            'ankle_asymmetry': ankle_asymmetry,
            'gait_frequency': gait_frequency,
            'gait_regularity': gait_regularity,
            'accel_mean': accel_mean,
            'accel_std': accel_std,
            'jerk_mean': jerk_mean,
            'stride_cv': stride_cv,
        }

    def _classify_severity(self, metrics: Dict[str, float]) -> str:
        risk_points = 0
        if metrics.get('speed_mean', 0) < 40:
            risk_points += 2
        if metrics.get('stride_cv', 0) > 20:
            risk_points += 2
        if metrics.get('gait_regularity', 0) < 2.0:
            risk_points += 1
        if metrics.get('jerk_mean', 0) > 1500:
            risk_points += 1

        if risk_points >= 5:
            return 'severe'
        if risk_points >= 3:
            return 'moderate'
        if risk_points >= 1:
            return 'mild'
        return 'normal'

    def _empty_result(self, segment: AnalysisSegment) -> SymptomResult:
        return SymptomResult(
            symptom_type='gait',
            person_id=segment.person_id,
            segment_id=f"gait_{segment.start_frame}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            metrics={},
            severity='unknown',
            confidence=0,
        )


# =============================================================================
# MAIN ANALYZER
# =============================================================================

class PDSymptomsAnalyzer:
    """
    Main analyzer for all PD motor symptoms.

    Analysis workflow:
    1. Extract person tracks from video
    2. Detect and classify activity segments:
       - Walking (보행구간) → Gait analysis, FOG detection
       - Resting (안정구간) → Tremor analysis
       - Task (작업구간) → Bradykinesia, rigidity analysis
       - Standing (서있는 구간) → Posture analysis
    3. Apply appropriate analyses to each segment type
    4. Aggregate statistics per person
    """

    # Mapping from activity type to analysis functions
    ACTIVITY_ANALYSIS_MAP = {
        'walking': ['gait'],  # FOG is analyzed at standing↔walking transitions
        'resting': ['tremor'],
        'task': ['bradykinesia'],
        'standing': ['posture']
    }

    def __init__(self):
        self.tracker = MultiPersonTracker()
        self.activity_detector = UnifiedActivityDetector()
        self.fog_transition_detector = FOGTransitionDetector()

        # Symptom analyzers
        self.symptom_analyzers = {
            'tremor': TremorAnalyzer(),
            'bradykinesia': BradykinesiaAnalyzer(),
            'posture': PostureAnalyzer(),
            'fog': None,  # Alias populated from fog_transition results
            'fog_transition': FOGTransitionAnalyzer(),
            'gait': GaitAnalyzer(),
        }

        self.aggregator = SymptomStatisticalAggregator()

    def _build_skeleton_track_payload(self, track: PersonTrack, frame_stride: int = 2) -> Dict:
        """Serialize per-frame pose keypoints for UI skeleton overlay."""
        stride = max(1, int(frame_stride))
        sampled = np.round(track.keypoints[::stride], 4)
        return {
            'start_frame': track.start_frame,
            'end_frame': track.end_frame,
            'frame_stride': stride,
            'n_frames': int(sampled.shape[0]),
            'n_landmarks': int(sampled.shape[1]),
            'keypoints': sampled.tolist(),
        }

    def analyze_video(self, video_path: str,
                      symptoms: List[str] = None,
                      progress_callback=None,
                      include_skeleton: bool = False,
                      skeleton_frame_stride: int = 2) -> Dict:
        """
        Analyze video for PD symptoms with activity-based segmentation.

        Workflow:
        1. Extract person tracks
        2. Classify into activity segments (walking/resting/task/standing)
        3. Apply appropriate analyses based on activity type
        4. Aggregate statistics

        Args:
            video_path: Path to video file
            symptoms: List of symptoms to analyze (default: all)
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with activity segments and per-person symptom analysis
        """
        if symptoms is None:
            symptoms = ['tremor', 'bradykinesia', 'posture', 'fog', 'fog_transition', 'gait']

        # Step 1: Extract person tracks
        if progress_callback:
            progress_callback(0, "Extracting person tracks...")

        tracks, video_info = self.tracker.extract_tracks(
            video_path,
            extract_hands=True,
            extract_face=True,
            progress_callback=lambda p: progress_callback(p * 0.3, "Tracking...") if progress_callback else None
        )

        fps = video_info['fps']

        if not tracks:
            return {
                'video_info': video_info,
                'persons': [],
                'activity_segments': [],
                'error': 'No persons detected in video'
            }

        # Step 2: Detect activity segments for each person
        if progress_callback:
            progress_callback(30, "Classifying activity segments...")

        all_activities: Dict[str, List[ActivitySegment]] = {}
        activity_summary = {'walking': 0, 'resting': 0, 'task': 0, 'standing': 0}

        for track in tracks:
            activities = self.activity_detector.detect_activities(track, fps)
            all_activities[track.person_id] = activities

            # Count durations
            for act in activities:
                if act.activity_type in activity_summary:
                    activity_summary[act.activity_type] += act.duration

        # Step 3: Convert activity segments to analysis segments and analyze
        if progress_callback:
            progress_callback(40, "Analyzing symptoms by activity...")

        all_results: Dict[str, Dict[str, List[SymptomResult]]] = defaultdict(lambda: defaultdict(list))
        all_analysis_segments: Dict[str, Dict[str, List[AnalysisSegment]]] = defaultdict(lambda: defaultdict(list))

        total_activities = sum(len(acts) for acts in all_activities.values())
        processed = 0

        for track in tracks:
            person_activities = all_activities[track.person_id]

            for activity in person_activities:
                # Get analyses for this activity type
                analysis_types = self.ACTIVITY_ANALYSIS_MAP.get(activity.activity_type, [])

                for analysis_type in analysis_types:
                    if analysis_type not in symptoms:
                        continue

                    # Create analysis segment from activity segment
                    analysis_segment = AnalysisSegment(
                        segment_type=analysis_type,
                        person_id=activity.person_id,
                        start_frame=activity.start_frame,
                        end_frame=activity.end_frame,
                        start_time=activity.start_time,
                        end_time=activity.end_time,
                        confidence=activity.confidence,
                        metadata={'activity_type': activity.activity_type}
                    )

                    all_analysis_segments[track.person_id][analysis_type].append(analysis_segment)

                    # Run analyzer if available
                    if analysis_type in self.symptom_analyzers and self.symptom_analyzers[analysis_type] is not None:
                        analyzer = self.symptom_analyzers[analysis_type]
                        result = analyzer.analyze(track, analysis_segment, fps)
                        all_results[track.person_id][analysis_type].append(result)

                processed += 1
                if progress_callback:
                    progress_callback(40 + (processed / max(1, total_activities)) * 40,
                                    f"Analyzing {activity.activity_type}...")

        # Step 3.5: Detect and analyze FOG at standing↔walking transitions
        if progress_callback:
            progress_callback(80, "Detecting FOG at transitions...")

        all_fog_transitions: Dict[str, List[AnalysisSegment]] = {}

        for track in tracks:
            person_activities = all_activities[track.person_id]

            # Detect FOG transition zones
            fog_transitions = self.fog_transition_detector.detect_transitions(
                person_activities, track, fps
            )
            all_fog_transitions[track.person_id] = fog_transitions

            # Analyze each FOG transition
            if 'fog_transition' in self.symptom_analyzers and self.symptom_analyzers['fog_transition'] is not None:
                analyzer = self.symptom_analyzers['fog_transition']
                for transition in fog_transitions:
                    all_analysis_segments[track.person_id]['fog_transition'].append(transition)
                    result = analyzer.analyze(track, transition, fps)
                    all_results[track.person_id]['fog_transition'].append(result)

        # Step 4: Aggregate results per person
        if progress_callback:
            progress_callback(90, "Aggregating statistics...")

        persons = []
        for track in tracks:
            # Keep fog alias synchronized with fog_transition for UI compatibility.
            if all_results[track.person_id]['fog_transition']:
                all_results[track.person_id]['fog'] = list(all_results[track.person_id]['fog_transition'])
                all_analysis_segments[track.person_id]['fog'] = list(all_analysis_segments[track.person_id]['fog_transition'])

            # Get activity segments for this person
            person_activities = all_activities.get(track.person_id, [])

            # Calculate activity breakdown
            activity_breakdown = {'walking': 0, 'resting': 0, 'task': 0, 'standing': 0}
            for act in person_activities:
                if act.activity_type in activity_breakdown:
                    activity_breakdown[act.activity_type] += act.duration

            # Get FOG transitions for this person
            fog_transitions = all_fog_transitions.get(track.person_id, [])

            person_data = {
                'person_id': track.person_id,
                'start_frame': track.start_frame,
                'end_frame': track.end_frame,
                'duration': (track.end_frame - track.start_frame) / fps,
                'activity_segments': [act.to_dict() for act in person_activities],
                'activity_breakdown': activity_breakdown,
                'fog_transitions': [t.to_dict() for t in fog_transitions],
                'n_fog_transitions': len(fog_transitions),
                'symptoms': {}
            }

            if include_skeleton:
                person_data['skeleton_track'] = self._build_skeleton_track_payload(
                    track, frame_stride=skeleton_frame_stride
                )

            for symptom in symptoms:
                results = all_results[track.person_id][symptom]
                segments = all_analysis_segments[track.person_id][symptom]

                if results:
                    summary = self.aggregator.aggregate(results)
                    person_data['symptoms'][symptom] = {
                        'summary': summary.to_dict(),
                        'segments': [s.to_dict() for s in segments],
                        'results': [r.to_dict() for r in results]
                    }
                else:
                    person_data['symptoms'][symptom] = {
                        'summary': None,
                        'segments': [s.to_dict() for s in segments],
                        'results': []
                    }

            persons.append(person_data)

        if progress_callback:
            progress_callback(100, "Complete")

        return _convert_numpy({
            'video_info': video_info,
            'n_persons': len(tracks),
            'activity_summary': activity_summary,
            'persons': persons,
            'analyzed_symptoms': symptoms
        })


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_pd_analyzer = None

def get_pd_symptoms_analyzer() -> PDSymptomsAnalyzer:
    """Get or create the singleton PD symptoms analyzer."""
    global _pd_analyzer
    if _pd_analyzer is None:
        _pd_analyzer = PDSymptomsAnalyzer()
    return _pd_analyzer
