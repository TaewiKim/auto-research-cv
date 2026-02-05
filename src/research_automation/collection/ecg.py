"""ECG signal loading and basic processing.

Requires optional dependencies: pip install research-automation[ecg]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Check for optional dependencies
try:
    import neurokit2 as nk

    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    nk = None

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    wfdb = None


def _check_neurokit() -> None:
    """Check if neurokit2 is available."""
    if not NEUROKIT_AVAILABLE:
        raise ImportError(
            "neurokit2 is required for ECG processing. "
            "Install with: pip install research-automation[ecg]"
        )


def _check_wfdb() -> None:
    """Check if wfdb is available."""
    if not WFDB_AVAILABLE:
        raise ImportError(
            "wfdb is required for PhysioNet ECG loading. "
            "Install with: pip install research-automation[ecg]"
        )


@dataclass
class ECGSignal:
    """ECG signal data."""

    data: np.ndarray
    sampling_rate: float
    duration: float  # seconds
    leads: list[str]
    metadata: dict


@dataclass
class ECGFeatures:
    """Extracted ECG features."""

    heart_rate: float  # BPM
    heart_rate_variability: dict[str, float]  # HRV metrics
    r_peaks: np.ndarray  # R-peak indices
    rr_intervals: np.ndarray  # RR intervals in ms
    quality_score: float  # Signal quality 0-1


def load_ecg_wfdb(record_path: str | Path) -> ECGSignal:
    """Load ECG from WFDB format (PhysioNet)."""
    _check_wfdb()

    record_path = Path(record_path)
    record_name = str(record_path.with_suffix(""))

    record = wfdb.rdrecord(record_name)

    return ECGSignal(
        data=record.p_signal,
        sampling_rate=record.fs,
        duration=len(record.p_signal) / record.fs,
        leads=record.sig_name,
        metadata={
            "units": record.units,
            "comments": record.comments,
        },
    )


def load_ecg_csv(
    file_path: str | Path,
    sampling_rate: float,
    time_col: str | None = None,
    signal_cols: list[str] | None = None,
) -> ECGSignal:
    """Load ECG from CSV file."""
    import pandas as pd

    df = pd.read_csv(file_path)

    if signal_cols:
        leads = signal_cols
        data = df[signal_cols].values
    else:
        # Assume all numeric columns except time are signals
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col and time_col in numeric_cols:
            numeric_cols.remove(time_col)
        leads = numeric_cols
        data = df[numeric_cols].values

    return ECGSignal(
        data=data,
        sampling_rate=sampling_rate,
        duration=len(data) / sampling_rate,
        leads=leads,
        metadata={"source": str(file_path)},
    )


def process_ecg(signal: ECGSignal, lead_idx: int = 0) -> ECGFeatures:
    """Extract features from ECG signal using NeuroKit2."""
    _check_neurokit()

    # Get single lead
    ecg_data = signal.data[:, lead_idx] if signal.data.ndim > 1 else signal.data

    # Clean signal
    ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=signal.sampling_rate)

    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=signal.sampling_rate)
    r_peak_indices = rpeaks["ECG_R_Peaks"]

    # Calculate RR intervals
    rr_intervals = np.diff(r_peak_indices) / signal.sampling_rate * 1000  # ms

    # Heart rate
    if len(rr_intervals) > 0:
        heart_rate = 60000 / np.mean(rr_intervals)  # BPM
    else:
        heart_rate = 0

    # HRV metrics
    hrv_metrics: dict[str, float] = {}
    if len(rr_intervals) > 2:
        try:
            hrv_time = nk.hrv_time(rpeaks, sampling_rate=signal.sampling_rate)
            hrv_metrics["rmssd"] = float(hrv_time["HRV_RMSSD"].iloc[0])
            hrv_metrics["sdnn"] = float(hrv_time["HRV_SDNN"].iloc[0])
            hrv_metrics["pnn50"] = float(hrv_time["HRV_pNN50"].iloc[0])
        except Exception:
            pass

    # Signal quality
    try:
        quality = nk.ecg_quality(ecg_cleaned, sampling_rate=signal.sampling_rate)
        quality_score = np.mean(quality == "Acceptable")
    except Exception:
        quality_score = 0.5

    return ECGFeatures(
        heart_rate=heart_rate,
        heart_rate_variability=hrv_metrics,
        r_peaks=np.array(r_peak_indices),
        rr_intervals=rr_intervals,
        quality_score=quality_score,
    )


def detect_arrhythmia(signal: ECGSignal, lead_idx: int = 0) -> dict[str, bool]:
    """Basic arrhythmia detection."""
    features = process_ecg(signal, lead_idx)

    results = {
        "bradycardia": features.heart_rate < 60,
        "tachycardia": features.heart_rate > 100,
        "irregular_rhythm": False,
        "ectopic_beats": False,
    }

    # Check rhythm regularity
    if len(features.rr_intervals) > 5:
        rr_std = np.std(features.rr_intervals)
        rr_mean = np.mean(features.rr_intervals)
        cv = rr_std / rr_mean if rr_mean > 0 else 0

        results["irregular_rhythm"] = cv > 0.15

        # Check for ectopic beats (RR intervals differing >25% from mean)
        if rr_mean > 0:
            deviation = np.abs(features.rr_intervals - rr_mean) / rr_mean
            results["ectopic_beats"] = np.any(deviation > 0.25)

    return results


def format_ecg_report(features: ECGFeatures) -> str:
    """Format ECG features as readable report."""
    lines = [
        "# ECG Analysis Report\n",
        "## Heart Rate",
        f"- BPM: {features.heart_rate:.1f}",
        f"- R-peaks detected: {len(features.r_peaks)}",
        f"- Signal quality: {features.quality_score*100:.1f}%",
        "",
        "## Heart Rate Variability",
    ]

    if features.heart_rate_variability:
        for metric, value in features.heart_rate_variability.items():
            lines.append(f"- {metric.upper()}: {value:.2f}")
    else:
        lines.append("- Insufficient data for HRV analysis")

    return "\n".join(lines)


def is_ecg_available() -> bool:
    """Check if ECG processing dependencies are available."""
    return NEUROKIT_AVAILABLE and WFDB_AVAILABLE
