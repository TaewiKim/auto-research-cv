"""Tests for report generation module."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from research_automation.report import (
    ExperimentReport,
    LiteratureReport,
    ReportGenerator,
    ReportSection,
    generate_experiment_report,
    generate_literature_report,
)


class TestReportSection:
    """Tests for ReportSection."""

    def test_section_basic(self):
        """Test basic section creation."""
        section = ReportSection(
            title="Introduction",
            content="This is the introduction.",
        )
        assert section.title == "Introduction"
        assert section.content == "This is the introduction."
        assert section.subsections == []

    def test_section_with_subsections(self):
        """Test section with subsections."""
        subsection = ReportSection(
            title="Background",
            content="Background information.",
        )
        section = ReportSection(
            title="Introduction",
            content="Main introduction.",
            subsections=[subsection],
        )
        assert len(section.subsections) == 1
        assert section.subsections[0].title == "Background"

    def test_section_with_figures(self):
        """Test section with figures."""
        section = ReportSection(
            title="Results",
            content="Result description.",
            figures=["fig1.png", "fig2.png"],
        )
        assert len(section.figures) == 2

    def test_section_with_tables(self):
        """Test section with tables."""
        section = ReportSection(
            title="Data",
            content="Data description.",
            tables=[{"header": ["A", "B"], "rows": [[1, 2], [3, 4]]}],
        )
        assert len(section.tables) == 1


class TestExperimentReport:
    """Tests for ExperimentReport."""

    @pytest.fixture
    def sample_experiment_report(self):
        """Create sample experiment report."""
        return ExperimentReport(
            name="UPDRS Baseline",
            description="Baseline experiment for UPDRS prediction",
            date=datetime(2024, 1, 15),
            dataset="CARE-PD",
            methods="Random Forest classifier with 100 trees",
            results={"confusion_matrix": [[10, 2], [1, 15]]},
            metrics={"accuracy": 0.89, "f1": 0.87, "precision": 0.88},
            figures=["confusion_matrix.png"],
            conclusions="The baseline achieves good performance.",
        )

    def test_report_attributes(self, sample_experiment_report):
        """Test report attributes."""
        report = sample_experiment_report
        assert report.name == "UPDRS Baseline"
        assert report.dataset == "CARE-PD"
        assert report.metrics["accuracy"] == 0.89

    def test_report_defaults(self):
        """Test report defaults."""
        report = ExperimentReport(
            name="Test",
            description="Test desc",
            date=datetime.now(),
            dataset="Test data",
            methods="Test methods",
            results={},
            metrics={"acc": 0.5},
        )
        assert report.figures == []
        assert report.conclusions == ""


class TestLiteratureReport:
    """Tests for LiteratureReport."""

    @pytest.fixture
    def sample_lit_report(self):
        """Create sample literature report."""
        return LiteratureReport(
            title="Video-based Parkinson's Assessment Review",
            date=datetime(2024, 2, 1),
            query="Parkinson gait video analysis",
            n_papers=25,
            papers=[
                {
                    "title": "Deep Learning for PD",
                    "authors": ["Smith, J.", "Doe, A."],
                    "source": "Nature Medicine",
                    "doi": "10.1038/s41591-024-00001",
                    "abstract": "Novel deep learning approach...",
                },
                {
                    "title": "Video Gait Analysis",
                    "authors": ["Brown, B."],
                    "source": "IEEE TMI",
                    "doi": "10.1109/tmi.2024.00002",
                    "abstract": "Automated gait analysis...",
                },
            ],
            summary="This review covers recent advances in video-based PD assessment.",
            key_findings=[
                "Deep learning outperforms traditional methods",
                "Smartphone videos are sufficient for assessment",
            ],
        )

    def test_lit_report_attributes(self, sample_lit_report):
        """Test literature report attributes."""
        report = sample_lit_report
        assert report.title == "Video-based Parkinson's Assessment Review"
        assert report.n_papers == 25
        assert len(report.papers) == 2
        assert len(report.key_findings) == 2

    def test_lit_report_defaults(self):
        """Test literature report defaults."""
        report = LiteratureReport(
            title="Test Review",
            date=datetime.now(),
            query="test query",
            n_papers=0,
            papers=[],
        )
        assert report.summary == ""
        assert report.key_findings == []


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create report generator."""
        return ReportGenerator()

    def test_generator_init(self, generator):
        """Test generator initialization."""
        assert generator.env is not None
        assert "format_number" in generator.env.filters
        assert "format_date" in generator.env.filters

    def test_format_number_filter(self, generator):
        """Test number formatting filter."""
        result = generator._format_number(3.14159, 2)
        assert result == "3.14"

    def test_format_date_filter(self, generator):
        """Test date formatting filter."""
        dt = datetime(2024, 6, 15)
        result = generator._format_date(dt)
        assert result == "2024-06-15"

    def test_render_string(self, generator):
        """Test rendering template string."""
        template = "Hello, {{ name }}!"
        result = generator.render_string(template, {"name": "World"})
        assert result == "Hello, World!"

    def test_generate_experiment_report(self, generator):
        """Test experiment report generation."""
        report = ExperimentReport(
            name="Test Experiment",
            description="Testing report generation",
            date=datetime(2024, 3, 1),
            dataset="Test Dataset",
            methods="Test methods",
            results={"key": "value"},
            metrics={"accuracy": 0.95},
        )
        content = generator.generate_experiment_report(report)
        assert "Test Experiment" in content
        assert "accuracy" in content
        assert "0.950" in content

    def test_generate_literature_report(self, generator):
        """Test literature report generation."""
        report = LiteratureReport(
            title="Test Literature Review",
            date=datetime(2024, 3, 1),
            query="test query",
            n_papers=1,
            papers=[
                {
                    "title": "Test Paper",
                    "authors": ["Author A", "Author B"],
                    "source": "Test Journal",
                    "doi": "10.1234/test",
                    "abstract": "Test abstract content.",
                }
            ],
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
        )
        content = generator.generate_literature_report(report)
        assert "Test Literature Review" in content
        assert "Test Paper" in content
        assert "Author A" in content

    def test_generate_comparison_report(self, generator):
        """Test comparison report generation."""
        experiments = [
            ExperimentReport(
                name="Experiment 1",
                description="First experiment",
                date=datetime(2024, 1, 1),
                dataset="Dataset A",
                methods="Method 1",
                results={},
                metrics={"accuracy": 0.85, "f1": 0.83},
            ),
            ExperimentReport(
                name="Experiment 2",
                description="Second experiment",
                date=datetime(2024, 1, 2),
                dataset="Dataset A",
                methods="Method 2",
                results={},
                metrics={"accuracy": 0.90, "f1": 0.88},
            ),
        ]
        content = generator.generate_comparison_report(experiments)
        assert "Experiment 1" in content
        assert "Experiment 2" in content
        assert "0.850" in content
        assert "0.900" in content

    def test_save_report_to_file(self, generator):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            report = ExperimentReport(
                name="File Test",
                description="Test saving",
                date=datetime.now(),
                dataset="Test",
                methods="Test",
                results={},
                metrics={"acc": 0.5},
            )
            content = generator.generate_experiment_report(report, output_path)
            assert output_path.exists()
            assert output_path.read_text() == content


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_experiment_report_func(self):
        """Test generate_experiment_report convenience function."""
        content = generate_experiment_report(
            name="Quick Test",
            description="Testing convenience function",
            dataset="Test Data",
            methods="Simple method",
            metrics={"accuracy": 0.75},
        )
        assert "Quick Test" in content
        assert "accuracy" in content

    def test_generate_literature_report_func(self):
        """Test generate_literature_report convenience function."""
        content = generate_literature_report(
            title="Quick Review",
            query="test search",
            papers=[
                {
                    "title": "Paper 1",
                    "authors": ["A. Author"],
                    "source": "Journal",
                    "abstract": "Abstract text.",
                }
            ],
            summary="Quick summary",
            key_findings=["Key finding"],
        )
        assert "Quick Review" in content
        assert "Paper 1" in content

    def test_generate_experiment_report_with_output(self):
        """Test saving experiment report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "exp_report.md"
            content = generate_experiment_report(
                name="Save Test",
                description="Test",
                dataset="Data",
                methods="Method",
                metrics={"m": 0.5},
                output_path=output_path,
            )
            assert output_path.exists()
            saved_content = output_path.read_text()
            assert saved_content == content

    def test_generate_literature_report_with_output(self):
        """Test saving literature report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lit_report.md"
            content = generate_literature_report(
                title="Save Test",
                query="query",
                papers=[],
                output_path=output_path,
            )
            assert output_path.exists()
            saved_content = output_path.read_text()
            assert saved_content == content
