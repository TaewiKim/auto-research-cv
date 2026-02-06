"""
Experiment Tracking with MLflow
================================

Track experiments, log metrics, and manage model artifacts.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    tracking_uri: str = "mlruns"
    artifact_location: str | None = None

    def __post_init__(self):
        if not self.tracking_uri.startswith(("http://", "https://", "file://")):
            # Local directory
            self.tracking_uri = f"file://{Path(self.tracking_uri).absolute()}"


@dataclass
class RunConfig:
    """Configuration for a single run."""

    name: str
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """
    MLflow-based experiment tracker.

    Usage:
        tracker = ExperimentTracker("my-experiment")

        with tracker.start_run("baseline-v1") as run:
            tracker.log_params({"model": "RandomForest", "n_estimators": 100})
            tracker.log_metrics({"accuracy": 0.85, "f1": 0.82})
            tracker.log_artifact("model.pkl")
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "mlruns",
        description: str = "",
    ):
        self.config = ExperimentConfig(
            name=experiment_name,
            description=description,
            tracking_uri=tracking_uri,
        )
        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None
        self._active_run: mlflow.ActiveRun | None = None

        self._setup()

    def _setup(self):
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.config.name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(
                self.config.name,
                artifact_location=self.config.artifact_location,
            )
        else:
            self._experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.config.name)
        self._client = MlflowClient()

    @property
    def experiment_id(self) -> str:
        """Get current experiment ID."""
        return self._experiment_id

    @property
    def active_run(self) -> mlflow.ActiveRun | None:
        """Get active run if any."""
        return self._active_run

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            description: Run description
            tags: Additional tags
            nested: Allow nested runs

        Yields:
            Active MLflow run
        """
        all_tags = {"description": description}
        if tags:
            all_tags.update(tags)

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=self._experiment_id,
            tags=all_tags,
            nested=nested,
        ) as run:
            self._active_run = run
            try:
                yield run
            finally:
                self._active_run = None

    def log_params(self, params: dict[str, Any]):
        """Log parameters."""
        for key, value in params.items():
            # Convert to string for MLflow
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None):
        """Log single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None):
        """Log artifact file."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None):
        """Log all files in directory as artifacts."""
        mlflow.log_artifacts(str(local_dir), artifact_path)

    def log_dict(self, dictionary: dict, artifact_file: str):
        """Log dictionary as JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ):
        """Log sklearn model."""
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name,
        )

    def log_dataclass(self, obj: Any, name: str = "config"):
        """Log dataclass as JSON artifact."""
        if hasattr(obj, "__dataclass_fields__"):
            data = asdict(obj)
            self.log_dict(data, f"{name}.json")

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, str]):
        """Set multiple tags."""
        mlflow.set_tags(tags)

    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """Get run by ID."""
        return self._client.get_run(run_id)

    def list_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: list[str] | None = None,
    ) -> list[mlflow.entities.Run]:
        """List runs in the experiment."""
        return self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or ["start_time DESC"],
        )

    def get_best_run(
        self,
        metric: str,
        maximize: bool = True,
    ) -> mlflow.entities.Run | None:
        """Get best run by metric."""
        order = "DESC" if maximize else "ASC"
        runs = self.list_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        return runs[0] if runs else None

    def compare_runs(
        self,
        run_ids: list[str],
        metrics: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compare metrics across runs."""
        results = {}
        for run_id in run_ids:
            run = self.get_run(run_id)
            run_metrics = {m: run.data.metrics.get(m) for m in metrics}
            results[run_id] = run_metrics
        return results


@dataclass
class ExperimentResult:
    """Result of an experiment run."""

    run_id: str
    run_name: str
    experiment_name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, str]
    start_time: datetime
    end_time: datetime | None
    status: str
    artifact_uri: str

    @classmethod
    def from_run(cls, run: mlflow.entities.Run, experiment_name: str) -> "ExperimentResult":
        """Create from MLflow run."""
        return cls(
            run_id=run.info.run_id,
            run_name=run.info.run_name or "",
            experiment_name=experiment_name,
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
            tags=dict(run.data.tags),
            start_time=datetime.fromtimestamp(run.info.start_time / 1000),
            end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            status=run.info.status,
            artifact_uri=run.info.artifact_uri,
        )


def run_experiment(
    experiment_name: str,
    run_name: str,
    train_fn: callable,
    params: dict[str, Any],
    tracking_uri: str = "mlruns",
) -> ExperimentResult:
    """
    Run an experiment with automatic tracking.

    Args:
        experiment_name: Name of the experiment
        run_name: Name for this run
        train_fn: Training function that returns dict of metrics
        params: Parameters to log
        tracking_uri: MLflow tracking URI

    Returns:
        ExperimentResult with run details
    """
    tracker = ExperimentTracker(experiment_name, tracking_uri)

    with tracker.start_run(run_name) as run:
        tracker.log_params(params)

        # Run training
        metrics = train_fn(params)

        # Log metrics
        if isinstance(metrics, dict):
            tracker.log_metrics(metrics)

    return ExperimentResult.from_run(
        tracker.get_run(run.info.run_id),
        experiment_name,
    )
