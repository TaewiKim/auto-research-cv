"""
Parkinson's Gait Analysis Web Application
=========================================
Smart analysis with walking detection and face recognition.
"""

import os
import json
import uuid
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
DATA_FOLDER = BASE_DIR / 'data'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, DATA_FOLDER]:
    folder.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Try to import real analyzers, fall back to mock if MediaPipe not available
MOCK_MODE = False
try:
    from smart_analyzer import get_analyzer
    from pd_symptoms_analyzer import get_pd_symptoms_analyzer
    analyzer = get_analyzer(DATA_FOLDER)
    pd_symptoms_analyzer = get_pd_symptoms_analyzer()
    print("✓ MediaPipe loaded - Real analysis mode")
except ImportError as e:
    print(f"⚠️ MediaPipe not available: {e}")
    print("⚠️ Running in MOCK MODE - analysis results are simulated")
    MOCK_MODE = True
    from app_simple import generate_mock_analysis, generate_mock_symptoms
    analyzer = None
    pd_symptoms_analyzer = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def derive_fog_transitions(walking_segments, video_duration):
    """
    Derive FOG transition points from walking segments.

    FOG occurs at standing ↔ walking boundaries:
    - Standing → Walking: gait initiation
    - Walking → Standing: gait termination

    Args:
        walking_segments: List of walking segment dicts with start_time, end_time
        video_duration: Total video duration in seconds

    Returns:
        List of transition dicts with type and boundary info
    """
    if not walking_segments:
        return []

    transitions = []
    sorted_segments = sorted(walking_segments, key=lambda s: s.get('start_time', 0))

    for i, seg in enumerate(sorted_segments):
        start = seg.get('start_time', 0)
        end = seg.get('end_time', start)

        # Standing → Walking transition (gait initiation)
        # Look at time before walking starts
        if i == 0:
            pre_standing_duration = start
        else:
            prev_end = sorted_segments[i-1].get('end_time', 0)
            pre_standing_duration = start - prev_end

        if pre_standing_duration >= 1.0:  # At least 1.0s of standing before
            transitions.append({
                'type': 'initiation',
                'transition_type': 'standing_to_walking',
                'boundary_time': start,
                'standing_duration': pre_standing_duration,
                'walking_segment_idx': i,
                'analysis_window': {
                    'start': max(0, start - 2.0),
                    'end': min(video_duration, start + 2.0)
                }
            })

        # Walking → Standing transition (gait termination)
        # Look at time after walking ends
        if i == len(sorted_segments) - 1:
            post_standing_duration = video_duration - end
        else:
            next_start = sorted_segments[i+1].get('start_time', end)
            post_standing_duration = next_start - end

        if post_standing_duration >= 1.0:  # At least 1.0s of standing after
            transitions.append({
                'type': 'termination',
                'transition_type': 'walking_to_standing',
                'boundary_time': end,
                'standing_duration': post_standing_duration,
                'walking_segment_idx': i,
                'analysis_window': {
                    'start': max(0, end - 2.0),
                    'end': min(video_duration, end + 2.0)
                }
            })

    return transitions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        return jsonify({
            'success': True,
            'filename': filename,
            'video_url': url_for('serve_video', filename=filename)
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run smart analysis with walking detection and user identification."""
    data = request.json
    filename = data.get('filename')
    identify_user = data.get('identify_user', True)

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        if MOCK_MODE:
            # Use mock analysis
            results = generate_mock_analysis()
            results['filename'] = filename
            results['mock_mode'] = True
        else:
            # Run real smart analysis
            results = analyzer.analyze_video(
                str(filepath),
                identify_user=identify_user
            )

        # Save results
        result_file = RESULTS_FOLDER / f"{filename.rsplit('.', 1)[0]}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    """Run comprehensive PD motor symptom analysis with multi-person tracking."""
    data = request.json
    filename = data.get('filename')
    symptoms = data.get('symptoms', None)  # List of symptoms or None for all

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        if MOCK_MODE:
            # Use mock analysis
            results = generate_mock_symptoms()
            results['filename'] = filename
            results['mock_mode'] = True
        else:
            # Run real multi-symptom analysis
            results = pd_symptoms_analyzer.analyze_video(
                str(filepath),
                symptoms=symptoms,
                include_skeleton=True,
                skeleton_frame_stride=2
            )

            # Also run gait analysis and merge results
            try:
                gait_results = analyzer.analyze_video(str(filepath), identify_user=False)
                walking_segments = gait_results.get('walking_detection', {}).get('segments', [])
                video_duration = gait_results.get('video_info', {}).get('duration', 60)

                # Derive FOG transitions from walking segments
                fog_transitions = derive_fog_transitions(walking_segments, video_duration)

                # Add gait_analysis to results
                summary = gait_results.get('summary', {}) or {}
                walking_detection = gait_results.get('walking_detection', {}) or {}
                analysis_results = gait_results.get('analysis_results', []) or []

                estimated_steps = 0
                for seg in analysis_results:
                    cadence = seg.get('cadence')
                    duration = seg.get('duration')
                    if cadence is None or duration is None:
                        continue
                    estimated_steps += int(round((cadence * duration) / 60.0))

                results['gait_analysis'] = {
                    'success': True,
                    'walking_detection': walking_detection,
                    'analysis_results': analysis_results,
                    'statistical_analysis': gait_results.get('statistical_analysis', {}),
                    'summary': summary,
                    'biomarkers': {
                        'stride_cv': summary.get('avg_stride_time_cv'),
                        'arm_swing_asymmetry': summary.get('avg_arm_swing_asymmetry'),
                        'step_time_asymmetry': summary.get('avg_step_time_asymmetry'),
                        'pd_risk_score': summary.get('avg_pd_risk_score'),
                        'walk_ratio': walking_detection.get('walking_ratio'),
                    },
                    'gait_metrics': {
                        'walking_speed': summary.get('avg_speed'),
                        'stride_length': summary.get('avg_stride_length'),
                        'cadence': summary.get('avg_cadence'),
                        'step_count': estimated_steps,
                    },
                    'classification': summary.get('overall_classification', 'Unknown'),
                    'pd_risk_score': summary.get('avg_pd_risk_score', 0) * 100,
                    'segments': walking_segments,
                    'fog_transitions': fog_transitions,
                    'fog_transition_count': len(fog_transitions),
                }

                # Enhance FOG results in persons with derived transitions
                for person in results.get('persons', []):
                    if 'symptoms' in person and 'fog' in person['symptoms']:
                        fog_result = person['symptoms']['fog']
                        fog_result['transitions_detected'] = len(fog_transitions)
                        fog_result['transitions'] = fog_transitions

                        # Calculate FOG metrics based on transitions
                        if fog_transitions:
                            # Update summary with transition-based metrics
                            initiation_count = sum(1 for t in fog_transitions if t['type'] == 'initiation')
                            termination_count = sum(1 for t in fog_transitions if t['type'] == 'termination')

                            fog_result['summary'] = fog_result.get('summary', {})
                            fog_result['summary']['transition_count'] = len(fog_transitions)
                            fog_result['summary']['initiation_count'] = initiation_count
                            fog_result['summary']['termination_count'] = termination_count
                            fog_result['summary']['walking_segments'] = len(walking_segments)

            except Exception as e:
                print(f"Gait analysis integration error: {e}")
                results['gait_analysis'] = {'error': str(e)}

        # Save results
        result_file = RESULTS_FOLDER / f"{filename.rsplit('.', 1)[0]}_symptoms.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/register-user', methods=['POST'])
def register_user():
    """Register a new user with face image."""
    if MOCK_MODE:
        data = request.json
        name = data.get('name')
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        import uuid
        return jsonify({
            'success': True,
            'user': {'id': str(uuid.uuid4())[:8], 'name': name},
            'mock_mode': True
        })

    data = request.json
    name = data.get('name')
    image_data = data.get('image')  # Base64 encoded image

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    if not image_data:
        return jsonify({'error': 'Face image is required'}), 400

    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Register user
        profile = analyzer.register_user(name, image)

        if profile:
            return jsonify({'success': True, 'user': profile})
        else:
            return jsonify({'error': 'Face not detected in image'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/users', methods=['GET'])
def list_users():
    """List all registered users."""
    if MOCK_MODE:
        return jsonify([
            {'id': 'mock_user1', 'name': 'Test User 1'},
            {'id': 'mock_user2', 'name': 'Test User 2'},
        ])
    users = analyzer.list_users()
    return jsonify(users)


@app.route('/users/<user_id>/photo')
def get_user_photo(user_id):
    """Get user's face photo."""
    photo_path = DATA_FOLDER / 'profiles' / f"{user_id}_face.jpg"
    if photo_path.exists():
        return send_from_directory(str(DATA_FOLDER / 'profiles'), f"{user_id}_face.jpg")
    return jsonify({'error': 'Photo not found'}), 404


@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route('/favicon.ico')
def favicon():
    # Silence browser favicon requests when no favicon asset is provided.
    return ('', 204)


@app.route('/reference-data')
def get_reference_data():
    return jsonify({
        'healthy_young': {'mean': 1.24, 'std': 0.18, 'n': 24},
        'healthy_older': {'mean': 1.21, 'std': 0.19, 'n': 18},
        'pd_off': {'mean': 0.86, 'std': 0.30, 'n': 23},
        'pd_on': {'mean': 1.02, 'std': 0.28, 'n': 25},
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Smart Gait Analysis Web Server")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Automatic walking detection")
    print("  - Face recognition for user identification")
    print("  - Real-time gait analysis")
    print("\nOpen http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
