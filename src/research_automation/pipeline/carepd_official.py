from __future__ import annotations

import subprocess
import os
from pathlib import Path
from typing import Sequence


def official_env_ready(code_dir: str | Path) -> tuple[bool, list[str]]:
    """
    Check minimal prerequisites for CARE-PD official pipeline.
    """
    code_root = Path(code_dir)
    issues: list[str] = []

    if not (code_root / "run.py").exists():
        issues.append("`run.py` not found in CARE-PD code directory.")
    if not (code_root / "eval_only.py").exists():
        issues.append("`eval_only.py` not found in CARE-PD code directory.")
    if not (code_root / "assets" / "datasets").exists():
        issues.append("`assets/datasets` missing (dataset not linked/copied).")
    ckpt_root = code_root / "assets" / "Pretrained_checkpoints"
    if not ckpt_root.exists():
        issues.append("`assets/Pretrained_checkpoints` missing (download_models.sh not run).")

    return len(issues) == 0, issues


def run_official_eval(
    code_dir: str | Path,
    backbone: str,
    config: str,
    protocol: str = "within",
    python_executable: str = "python",
    extra_args: Sequence[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run CARE-PD official eval script for a single setting.

    protocol:
      - `within`: within-dataset LOSO
      - `lodo`: leave-one-dataset-out
    """
    code_root = Path(code_dir)
    cmd = [
        python_executable,
        "run.py",
        "--backbone",
        backbone,
        "--config",
        config,
        "--hypertune",
        "0",
        "--this_run_num",
        "0",
    ]
    if protocol == "within":
        cmd.extend(["--num_folds", "-1"])
    elif protocol == "lodo":
        cmd.extend(["--cross_dataset_test", "1", "--force_LODO", "1", "--exp_name_rigid", "LODO"])
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    if extra_args:
        cmd.extend(list(extra_args))

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.run(
        cmd,
        cwd=str(code_root),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
