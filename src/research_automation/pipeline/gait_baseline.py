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
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
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
        "macro_f1": f1_score(y, y_pred, average="macro"),
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
    print(f"Macro-F1:          {results['macro_f1']:.3f}")

    if "roc_auc" in results:
        print(f"ROC-AUC:           {results['roc_auc']:.3f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    if feature_names:
        print("\nTop 10 Important Features:")
        for name, imp in get_feature_importance(results["model"], feature_names, 10):
            print(f"  {name:<20s} {imp:.4f}")


def _make_rf_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _load_dataset_with_subject_groups(
    dataset_dir: str | Path,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one CARE-PD dataset with subject-level groups."""
    walks, labels, subjects = load_care_pd_data(dataset_dir, datasets=[dataset_name])
    if len(walks) == 0:
        return np.empty((0, 0)), np.array([]), np.array([])
    X, _ = extract_features_batch(walks)
    # Strip dataset prefix (e.g., BMCLab.pkl_subjectA -> subjectA) for LOSO grouping.
    groups = np.array([s.split("_", 1)[1] if "_" in s else s for s in subjects])
    return X, labels, groups


def evaluate_within_dataset_loso(
    dataset_dir: str | Path,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Literature-style within-dataset evaluation (LOSO, subject-wise).

    Returns per-dataset metrics and macro-F1 mean across datasets.
    """
    if datasets is None:
        datasets = DATASETS_WITH_UPDRS

    results: dict[str, dict[str, float]] = {}
    macro_values = []

    for dataset_name in datasets:
        X, y, groups = _load_dataset_with_subject_groups(dataset_dir, dataset_name)
        if len(y) == 0:
            continue

        logo = LeaveOneGroupOut()
        y_pred = np.empty_like(y)

        for train_idx, test_idx in logo.split(X, y, groups=groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = _make_rf_classifier()
            clf.fit(X_train, y[train_idx])
            y_pred[test_idx] = clf.predict(X_test)

        metrics = {
            "macro_f1": float(f1_score(y, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "n_subjects": int(len(np.unique(groups))),
            "n_walks": int(len(y)),
        }
        results[dataset_name] = metrics
        macro_values.append(metrics["macro_f1"])

    if macro_values:
        results["summary"] = {"macro_f1_mean": float(np.mean(macro_values))}
    return results


def evaluate_lodo(
    dataset_dir: str | Path,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Literature-style Leave-One-Dataset-Out (LODO) evaluation.
    """
    if datasets is None:
        datasets = DATASETS_WITH_UPDRS

    cached: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset_name in datasets:
        cached[dataset_name] = _load_dataset_with_subject_groups(dataset_dir, dataset_name)

    results: dict[str, dict[str, float]] = {}
    macro_values = []

    for test_dataset in datasets:
        X_test, y_test, _ = cached[test_dataset]
        if len(y_test) == 0:
            continue

        train_parts = [cached[name] for name in datasets if name != test_dataset and len(cached[name][1]) > 0]
        X_train = np.concatenate([part[0] for part in train_parts], axis=0)
        y_train = np.concatenate([part[1] for part in train_parts], axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = _make_rf_classifier()
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        metrics = {
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "n_walks_test": int(len(y_test)),
        }
        results[test_dataset] = metrics
        macro_values.append(metrics["macro_f1"])

    if macro_values:
        results["summary"] = {"macro_f1_mean": float(np.mean(macro_values))}
    return results


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="CARE-PD baseline evaluation")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/CARE-PD",
        help="Directory containing CARE-PD .pkl files",
    )
    parser.add_argument(
        "--protocol",
        choices=["literature", "all", "legacy"],
        default="literature",
        help="Evaluation protocol. literature = LOSO + LODO (recommended).",
    )
    parser.add_argument(
        "--method",
        choices=["rf", "carepd_official", "auto"],
        default="auto",
        help="rf: handcrafted baseline, carepd_official: paper code path, auto: prefer official if ready.",
    )
    parser.add_argument(
        "--carepd-code-dir",
        default="data/datasets/CARE-PD-code",
        help="Path to official CARE-PD code repository.",
    )
    parser.add_argument(
        "--backbone",
        default="motionbert",
        help="Official CARE-PD backbone for run.py (e.g. motionbert, mixste, motionagformer, poseformerv2, momask).",
    )
    parser.add_argument(
        "--config",
        default="BMCLab_backright.json",
        help="Official CARE-PD config file for selected backbone.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    selected_method = args.method
    if selected_method == "auto":
        try:
            from carepd_official import official_env_ready
        except Exception:
            selected_method = "rf"
        else:
            ready, _ = official_env_ready(args.carepd_code_dir)
            selected_method = "carepd_official" if ready else "rf"

    if selected_method == "carepd_official":
        from carepd_official import official_env_ready, run_official_eval

        ready, issues = official_env_ready(args.carepd_code_dir)
        if not ready:
            print("CARE-PD official method is not ready:")
            for issue in issues:
                print(f"  - {issue}")
            print("Falling back to RF baseline method.")
            selected_method = "rf"
        else:
            protocol = "within" if args.protocol in {"literature", "all"} else "within"
            print("\n" + "=" * 50)
            print("CARE-PD Official Method")
            print("=" * 50)
            print(f"code_dir:  {args.carepd_code_dir}")
            print(f"protocol:  {protocol}")
            print(f"backbone:  {args.backbone}")
            print(f"config:    {args.config}")
            run = run_official_eval(
                code_dir=args.carepd_code_dir,
                backbone=args.backbone,
                config=args.config,
                protocol=protocol,
                python_executable=sys.executable,
            )
            print("\n[stdout]")
            print(run.stdout[-8000:] if run.stdout else "(empty)")
            print("\n[stderr]")
            print(run.stderr[-8000:] if run.stderr else "(empty)")
            print(f"\nexit_code: {run.returncode}")
            summary = {
                "method": "carepd_official",
                "protocol": protocol,
                "backbone": args.backbone,
                "config": args.config,
                "exit_code": run.returncode,
            }
            print("\nsummary:", json.dumps(summary, ensure_ascii=False))
            raise SystemExit(run.returncode)

    if selected_method == "rf" and args.protocol in {"literature", "all"}:
        print("\n" + "=" * 50)
        print("Within-Dataset LOSO (Literature Protocol)")
        print("=" * 50)
        loso_results = evaluate_within_dataset_loso(dataset_dir)
        for ds in DATASETS_WITH_UPDRS:
            if ds not in loso_results:
                continue
            r = loso_results[ds]
            print(
                f"{ds:<12s} Macro-F1={r['macro_f1']:.3f}  "
                f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}  "
                f"Subjects={r['n_subjects']}  Walks={r['n_walks']}"
            )
        if "summary" in loso_results:
            print(f"LOSO Macro-F1 mean: {loso_results['summary']['macro_f1_mean']:.3f}")

        print("\n" + "=" * 50)
        print("LODO (Literature Protocol)")
        print("=" * 50)
        lodo_results = evaluate_lodo(dataset_dir)
        for ds in DATASETS_WITH_UPDRS:
            if ds not in lodo_results:
                continue
            r = lodo_results[ds]
            print(
                f"Test={ds:<12s} Macro-F1={r['macro_f1']:.3f}  "
                f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}  "
                f"N_test={r['n_walks_test']}"
            )
        if "summary" in lodo_results:
            print(f"LODO Macro-F1 mean: {lodo_results['summary']['macro_f1_mean']:.3f}")

    if selected_method == "rf" and args.protocol in {"legacy", "all"}:
        print("\n" + "=" * 50)
        print("Legacy Pooled CV (Not Literature Protocol)")
        print("=" * 50)
        print("Loading CARE-PD data...")
        walks, labels, _ = load_care_pd_data(dataset_dir)
        print(f"Loaded {len(walks)} walks")
        X, feature_names = extract_features_batch(walks)
        print(f"Feature matrix: {X.shape}")

        print("\n4-Class UPDRS Prediction")
        results = train_updrs_classifier(X, labels, binary=False)
        print_results(results, feature_names)

        print("\nBinary: Normal vs Impaired")
        results_binary = train_updrs_classifier(X, labels, binary=True)
        print_results(results_binary, feature_names)
