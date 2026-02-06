"""Report generation for research outputs."""

from .generator import (
    ExperimentReport,
    LiteratureReport,
    ReportGenerator,
    ReportSection,
    generate_experiment_report,
    generate_literature_report,
)

__all__ = [
    "ExperimentReport",
    "LiteratureReport",
    "ReportGenerator",
    "ReportSection",
    "generate_experiment_report",
    "generate_literature_report",
]
