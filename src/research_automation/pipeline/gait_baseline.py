"""
CARE-PD Gait UPDRS Prediction Baseline
======================================

Simple baseline using handcrafted gait features + Random Forest
for predicting UPDRS gait scores from SMPL pose sequences.

Usage:
    from research_automation.pipeline.gait_baseline import (
        load_care_pd_data,
        extract_gait_features,
        train_updrs_classifier,
        evaluate_model,
    )

    # Load data
    walks, labels, subjects = load_care_pd_data("data/datasets/vida-adl_CARE-PD")

    # Extract features
    X, feature_names = extract_features_batch(walks)

    # Train and evaluate
    results = train_updrs_classifier(X, labels, n_splits=6)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


DATASETS_WITH_UPDRS = ["3DGait.pkl", "BMCLab.pkl", "PD-GaM.pkl", "T-SDU-PD.pkl"]

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]


def load_care_pd_data(
    dataset_dir: str | Path,
    datasets: list[str] | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Load CARE-PD data with UPDRS labels.

    Args:
        dataset_dir: Path to CARE-PD dataset directory
        datasets: List of dataset filenames to load (default: all with UPDRS)

    Returns:
        walks: List of walk dictionaries with pose, trans, etc.
        labels: UPDRS gait scores (0-3)
        subjects: Subject identifiers for grouping
    """
    dataset_dir = Path(dataset_dir)
    if datasets is None:
        datasets = DATASETS_WITH_UPDRS

    walks = []
    labels = []
    subjects = []

    for pkl_name in datasets:
        pkl_path = dataset_dir / pkl_name
        if not pkl_path.exists():
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for subj_id, subj_data in data.items():
            for walk_id, walk_data in subj_data.items():
                updrs = walk_data.get("UPDRS_GAIT")
                if updrs is not None:
                    walks.append(walk_data)
                    labels.append(updrs)
                    subjects.append(f"{pkl_name}_{subj_id}")

    return walks, np.array(labels), np.array(subjects)


def extract_gait_features(walk_data: dict[str, Any]) -> dict[str, float]:
    """
    Extract handcrafted gait features from a single walk.

    Args:
        walk_data: Dictionary with 'pose', 'trans', 'fps' keys

    Returns:
        Dictionary of feature name -> value
    """
    pose = walk_data["pose"]  # (T, 72) - 24 joints × 3 axis-angle
    trans = walk_data["trans"]  # (T, 3) - translation
    fps = walk_data.get("fps", 30)

    features = {}

    # --- Temporal features ---
    features["duration"] = pose.shape[0] / fps
    features["n_frames"] = pose.shape[0]

    # --- Translation/velocity features ---
    trans_diff = np.diff(trans, axis=0)
    velocity = trans_diff * fps
    speed = np.linalg.norm(velocity, axis=1)

    features["speed_mean"] = speed.mean()
    features["speed_std"] = speed.std()
    features["speed_max"] = speed.max()
    features["speed_min"] = speed.min() if len(speed) > 0 else 0

    features["vel_x_mean"] = np.abs(velocity[:, 0]).mean()
    features["vel_y_mean"] = np.abs(velocity[:, 1]).mean()
    features["vel_z_mean"] = np.abs(velocity[:, 2]).mean()

    # --- Pose statistics ---
    pose_reshaped = pose.reshape(-1, 24, 3)

    features["pose_std_mean"] = pose.std(axis=0).mean()
    features["pose_range_mean"] = (pose.max(axis=0) - pose.min(axis=0)).mean()

    # Per-joint variability (key joints for gait)
    joint_indices = {
        "pelvis": 0, "left_hip": 1, "right_hip": 2,
        "left_knee": 4, "right_knee": 5,
        "left_ankle": 7, "right_ankle": 8,
        "spine1": 3, "spine2": 6, "spine3": 9,
    }

    for joint_name, idx in joint_indices.items():
        joint_pose = pose_reshaped[:, idx, :]
        features[f"{joint_name}_std"] = joint_pose.std()
        features[f"{joint_name}_range"] = joint_pose.max() - joint_pose.min()

    # --- Symmetry features ---
    features["hip_asymmetry"] = np.abs(
        pose_reshaped[:, 1, :] - pose_reshaped[:, 2, :]
    ).mean()
    features["knee_asymmetry"] = np.abs(
        pose_reshaped[:, 4, :] - pose_reshaped[:, 5, :]
    ).mean()
    features["ankle_asymmetry"] = np.abs(
        pose_reshaped[:, 7, :] - pose_reshaped[:, 8, :]
    ).mean()

    # --- Frequency domain features ---
    pelvis_z = trans[:, 2] if trans.shape[1] > 2 else trans[:, 1]
    if len(pelvis_z) > 10:
        fft = np.fft.fft(pelvis_z - pelvis_z.mean())
        freqs = np.fft.fftfreq(len(pelvis_z), 1 / fps)
        pos_mask = (freqs > 0.3) & (freqs < 3)
        if pos_mask.any():
            fft_mag = np.abs(fft)[pos_mask]
            features["gait_frequency"] = freqs[pos_mask][np.argmax(fft_mag)]
            features["gait_regularity"] = fft_mag.max() / (fft_mag.mean() + 1e-6)
        else:
            features["gait_frequency"] = 0
            features["gait_regularity"] = 0
    else:
        features["gait_frequency"] = 0
        features["gait_regularity"] = 0

    # --- Acceleration features ---
    if len(velocity) > 1:
        accel = np.diff(velocity, axis=0) * fps
        accel_mag = np.linalg.norm(accel, axis=1)
        features["accel_mean"] = accel_mag.mean()
        features["accel_std"] = accel_mag.std()
        features["accel_max"] = accel_mag.max()
    else:
        features["accel_mean"] = 0
        features["accel_std"] = 0
        features["accel_max"] = 0

    # --- Jerk (smoothness) ---
    if len(velocity) > 2:
        jerk = np.diff(np.diff(velocity, axis=0), axis=0) * fps * fps
        jerk_mag = np.linalg.norm(jerk, axis=1)
        features["jerk_mean"] = jerk_mag.mean()
        features["jerk_std"] = jerk_mag.std()
    else:
        features["jerk_mean"] = 0
        features["jerk_std"] = 0

    return features


def extract_features_batch(
    walks: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features from multiple walks.

    Returns:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
    """
    feature_list = [extract_gait_features(w) for w in walks]
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, feature_names


def train_updrs_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 6,
    binary: bool = False,
) -> dict[str, Any]:
    """
    Train and evaluate UPDRS classifier with cross-validation.

    Args:
        X: Feature matrix
        y: Labels (UPDRS 0-3)
        n_splits: Number of CV folds
        binary: If True, convert to binary (Normal vs Impaired)

    Returns:
        Dictionary with accuracy, predictions, and model
    """
    if binary:
        y = (y > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)

    # Fit final model on all data
    clf.fit(X_scaled, y)

    results = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "y_true": y,
        "y_pred": y_pred,
        "model": clf,
        "scaler": scaler,
        "confusion_matrix": confusion_matrix(y, y_pred),
    }

    if binary:
        y_prob = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        results["roc_auc"] = roc_auc_score(y, y_prob)
        results["y_prob"] = y_prob

    return results


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_k: int = 15,
) -> list[tuple[str, float]]:
    """Get top-k most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[i], importances[i]) for i in indices]


def print_results(results: dict[str, Any], feature_names: list[str] | None = None):
    """Print evaluation results."""
    print(f"Accuracy:          {results['accuracy']:.3f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.3f}")

    if "roc_auc" in results:
        print(f"ROC-AUC:           {results['roc_auc']:.3f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    if feature_names:
        print("\nTop 10 Important Features:")
        for name, imp in get_feature_importance(results["model"], feature_names, 10):
            print(f"  {name:<20s} {imp:.4f}")


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "data/datasets/vida-adl_CARE-PD"

    print("Loading CARE-PD data...")
    walks, labels, subjects = load_care_pd_data(dataset_dir)
    print(f"Loaded {len(walks)} walks")

    print("\nExtracting features...")
    X, feature_names = extract_features_batch(walks)
    print(f"Feature matrix: {X.shape}")

    print("\n" + "=" * 50)
    print("4-Class UPDRS Prediction")
    print("=" * 50)
    results = train_updrs_classifier(X, labels, binary=False)
    print_results(results, feature_names)

    print("\n" + "=" * 50)
    print("Binary: Normal vs Impaired")
    print("=" * 50)
    results_binary = train_updrs_classifier(X, labels, binary=True)
    print_results(results_binary, feature_names)
