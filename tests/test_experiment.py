"""Tests for experiment tracking module."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from research_automation.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentTracker,
    RunConfig,
    run_experiment,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = ExperimentConfig(name="test-experiment")
        assert config.name == "test-experiment"
        assert config.description == ""
        assert config.tracking_uri.startswith("file://")

    def test_config_with_description(self):
        """Test config with description."""
        config = ExperimentConfig(
            name="my-exp",
            description="Test experiment description",
        )
        assert config.description == "Test experiment description"

    def test_config_with_tags(self):
        """Test config with tags."""
        config = ExperimentConfig(
            name="tagged-exp",
            tags={"version": "1.0", "type": "baseline"},
        )
        assert config.tags["version"] == "1.0"
        assert config.tags["type"] == "baseline"


class TestRunConfig:
    """Tests for RunConfig."""

    def test_run_config_defaults(self):
        """Test default run configuration."""
        config = RunConfig(name="run-1")
        assert config.name == "run-1"
        assert config.description == ""
        assert config.params == {}

    def test_run_config_with_params(self):
        """Test run config with parameters."""
        config = RunConfig(
            name="run-2",
            params={"model": "RandomForest", "n_estimators": 100},
        )
        assert config.params["model"] == "RandomForest"
        assert config.params["n_estimators"] == 100


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    @pytest.fixture
    def temp_tracking_dir(self):
        """Create temporary tracking directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "mlruns"

    def test_tracker_init(self, temp_tracking_dir):
        """Test tracker initialization."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        assert tracker.config.name == "test-exp"
        assert tracker.experiment_id is not None

    def test_start_run(self, temp_tracking_dir):
        """Test starting a run."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        with tracker.start_run("test-run") as run:
            assert run is not None
            assert run.info.run_id is not None

    def test_log_params(self, temp_tracking_dir):
        """Test logging parameters."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        with tracker.start_run("param-run") as run:
            tracker.log_params({"model": "SVM", "C": 1.0})
            # Params are logged asynchronously, just verify no error

    def test_log_metrics(self, temp_tracking_dir):
        """Test logging metrics."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        with tracker.start_run("metric-run") as run:
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
            tracker.log_metric("precision", 0.94)
            # Verify no error

    def test_list_runs(self, temp_tracking_dir):
        """Test listing runs."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        # Create some runs
        for i in range(3):
            with tracker.start_run(f"run-{i}"):
                tracker.log_metrics({"accuracy": 0.8 + i * 0.05})

        runs = tracker.list_runs()
        assert len(runs) == 3

    def test_get_best_run(self, temp_tracking_dir):
        """Test getting best run."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        # Create runs with different metrics
        for i, acc in enumerate([0.7, 0.9, 0.8]):
            with tracker.start_run(f"run-{i}"):
                tracker.log_metrics({"accuracy": acc})

        best = tracker.get_best_run("accuracy", maximize=True)
        assert best is not None
        assert abs(best.data.metrics["accuracy"] - 0.9) < 0.01

    def test_compare_runs(self, temp_tracking_dir):
        """Test comparing runs."""
        tracker = ExperimentTracker(
            experiment_name="test-exp",
            tracking_uri=str(temp_tracking_dir),
        )
        run_ids = []
        for i in range(2):
            with tracker.start_run(f"run-{i}") as run:
                tracker.log_metrics({"accuracy": 0.8 + i * 0.1, "loss": 0.3 - i * 0.1})
                run_ids.append(run.info.run_id)

        comparison = tracker.compare_runs(run_ids, ["accuracy", "loss"])
        assert len(comparison) == 2
        for run_id in run_ids:
            assert run_id in comparison
            assert "accuracy" in comparison[run_id]


class TestExperimentResult:
    """Tests for ExperimentResult."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result."""
        return ExperimentResult(
            run_id="abc123",
            run_name="test-run",
            experiment_name="test-exp",
            params={"model": "RF", "n_trees": 100},
            metrics={"accuracy": 0.92, "f1": 0.89},
            tags={"version": "1.0"},
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="FINISHED",
            artifact_uri="mlruns/0/abc123/artifacts",
        )

    def test_result_attributes(self, sample_result):
        """Test result attributes."""
        assert sample_result.run_id == "abc123"
        assert sample_result.run_name == "test-run"
        assert sample_result.params["model"] == "RF"
        assert sample_result.metrics["accuracy"] == 0.92

    def test_result_from_run(self):
        """Test creating result from MLflow run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test-exp",
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )
            with tracker.start_run("test") as run:
                tracker.log_params({"x": 1})
                tracker.log_metrics({"y": 0.5})

            mlflow_run = tracker.get_run(run.info.run_id)
            result = ExperimentResult.from_run(mlflow_run, "test-exp")

            assert result.run_id == run.info.run_id
            assert result.experiment_name == "test-exp"


class TestRunExperiment:
    """Tests for run_experiment convenience function."""

    def test_run_experiment_basic(self):
        """Test basic experiment run."""
        with tempfile.TemporaryDirectory() as tmpdir:

            def train_fn(params):
                return {"accuracy": params["lr"] * 10}

            result = run_experiment(
                experiment_name="quick-exp",
                run_name="test",
                train_fn=train_fn,
                params={"lr": 0.01},
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )

            assert result.status == "FINISHED"
            assert "accuracy" in result.metrics
            assert abs(result.metrics["accuracy"] - 0.1) < 0.01
