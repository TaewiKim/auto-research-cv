"""
Research Report Generator
=========================

Generate research reports from experiment results and literature reviews.
Uses Jinja2 templates for flexible output formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class ReportSection:
    """A section in the report."""

    title: str
    content: str
    subsections: list["ReportSection"] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)


@dataclass
class ExperimentReport:
    """Report data for an experiment."""

    name: str
    description: str
    date: datetime
    dataset: str
    methods: str
    results: dict[str, Any]
    metrics: dict[str, float]
    figures: list[str] = field(default_factory=list)
    conclusions: str = ""


@dataclass
class LiteratureReport:
    """Report data for literature review."""

    title: str
    date: datetime
    query: str
    n_papers: int
    papers: list[dict[str, Any]]
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)


class ReportGenerator:
    """Generate reports from templates."""

    def __init__(self, template_dir: str | Path | None = None):
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent.parent / "config" / "templates"

        self.template_dir = Path(template_dir)

        if self.template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self.env = Environment(autoescape=select_autoescape(["html", "xml"]))

        # Register custom filters
        self.env.filters["format_number"] = self._format_number
        self.env.filters["format_date"] = self._format_date

    @staticmethod
    def _format_number(value: float, decimals: int = 3) -> str:
        """Format number with specified decimals."""
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"
        return str(value)

    @staticmethod
    def _format_date(value: datetime, fmt: str = "%Y-%m-%d") -> str:
        """Format datetime."""
        if isinstance(value, datetime):
            return value.strftime(fmt)
        return str(value)

    def render_template(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> str:
        """Render a template with context."""
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_string(
        self,
        template_str: str,
        context: dict[str, Any],
    ) -> str:
        """Render a template string with context."""
        template = self.env.from_string(template_str)
        return template.render(**context)

    def generate_experiment_report(
        self,
        report: ExperimentReport,
        output_path: str | Path | None = None,
        template: str | None = None,
    ) -> str:
        """
        Generate experiment report.

        Args:
            report: ExperimentReport data
            output_path: Optional path to save report
            template: Template name or string

        Returns:
            Rendered report string
        """
        if template is None:
            template = EXPERIMENT_REPORT_TEMPLATE

        context = {
            "report": report,
            "generated_at": datetime.now(),
        }

        content = self.render_string(template, context)

        if output_path:
            Path(output_path).write_text(content)

        return content

    def generate_literature_report(
        self,
        report: LiteratureReport,
        output_path: str | Path | None = None,
        template: str | None = None,
    ) -> str:
        """
        Generate literature review report.

        Args:
            report: LiteratureReport data
            output_path: Optional path to save report
            template: Template name or string

        Returns:
            Rendered report string
        """
        if template is None:
            template = LITERATURE_REPORT_TEMPLATE

        context = {
            "report": report,
            "generated_at": datetime.now(),
        }

        content = self.render_string(template, context)

        if output_path:
            Path(output_path).write_text(content)

        return content

    def generate_comparison_report(
        self,
        experiments: list[ExperimentReport],
        output_path: str | Path | None = None,
    ) -> str:
        """Generate comparison report across experiments."""
        template = COMPARISON_REPORT_TEMPLATE

        context = {
            "experiments": experiments,
            "generated_at": datetime.now(),
        }

        content = self.render_string(template, context)

        if output_path:
            Path(output_path).write_text(content)

        return content


# Default templates
EXPERIMENT_REPORT_TEMPLATE = """# {{ report.name }}

**Date:** {{ report.date | format_date }}
**Dataset:** {{ report.dataset }}

## Description

{{ report.description }}

## Methods

{{ report.methods }}

## Results

{% for key, value in report.metrics.items() %}
- **{{ key }}:** {{ value | format_number }}
{% endfor %}

{% if report.results %}
### Detailed Results

{% for key, value in report.results.items() %}
**{{ key }}:**
{% if value is mapping %}
{% for k, v in value.items() %}
  - {{ k }}: {{ v }}
{% endfor %}
{% else %}
{{ value }}
{% endif %}

{% endfor %}
{% endif %}

{% if report.figures %}
## Figures

{% for fig in report.figures %}
![Figure]({{ fig }})

{% endfor %}
{% endif %}

{% if report.conclusions %}
## Conclusions

{{ report.conclusions }}
{% endif %}

---
*Generated on {{ generated_at | format_date("%Y-%m-%d %H:%M") }}*
"""

LITERATURE_REPORT_TEMPLATE = """# Literature Review: {{ report.title }}

**Date:** {{ report.date | format_date }}
**Search Query:** {{ report.query }}
**Papers Found:** {{ report.n_papers }}

{% if report.summary %}
## Summary

{{ report.summary }}
{% endif %}

{% if report.key_findings %}
## Key Findings

{% for finding in report.key_findings %}
- {{ finding }}
{% endfor %}
{% endif %}

## Papers

{% for paper in report.papers %}
### {{ loop.index }}. {{ paper.title }}

**Authors:** {{ paper.authors | join(", ") if paper.authors is iterable else paper.authors }}
**Source:** {{ paper.source }}{% if paper.doi %} | DOI: {{ paper.doi }}{% endif %}

{% if paper.abstract %}
{{ paper.abstract[:500] }}{% if paper.abstract|length > 500 %}...{% endif %}
{% endif %}

{% endfor %}

---
*Generated on {{ generated_at | format_date("%Y-%m-%d %H:%M") }}*
"""

COMPARISON_REPORT_TEMPLATE = """# Experiment Comparison Report

**Generated:** {{ generated_at | format_date("%Y-%m-%d %H:%M") }}
**Experiments:** {{ experiments | length }}

## Summary Table

{% if experiments %}
{% set first_exp = experiments[0] %}
| Experiment | Dataset | {% for key in first_exp.metrics.keys() %}{{ key }} | {% endfor %}

|------------|---------|{% for _ in first_exp.metrics.keys() %}--------|{% endfor %}

{% for exp in experiments %}
| {{ exp.name }} | {{ exp.dataset }} | {% for value in exp.metrics.values() %}{{ value | format_number }} | {% endfor %}

{% endfor %}
{% endif %}

## Individual Experiments

{% for exp in experiments %}
### {{ exp.name }}

**Date:** {{ exp.date | format_date }}
**Dataset:** {{ exp.dataset }}

{{ exp.description }}

**Metrics:**
{% for key, value in exp.metrics.items() %}
- {{ key }}: {{ value | format_number }}
{% endfor %}

{% endfor %}

---
*Generated on {{ generated_at | format_date("%Y-%m-%d %H:%M") }}*
"""


def generate_experiment_report(
    name: str,
    description: str,
    dataset: str,
    methods: str,
    metrics: dict[str, float],
    results: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Convenience function to generate experiment report.

    Args:
        name: Experiment name
        description: Description
        dataset: Dataset used
        methods: Methods description
        metrics: Dict of metric name -> value
        results: Optional additional results
        output_path: Optional output file path

    Returns:
        Rendered report string
    """
    report = ExperimentReport(
        name=name,
        description=description,
        date=datetime.now(),
        dataset=dataset,
        methods=methods,
        results=results or {},
        metrics=metrics,
    )

    generator = ReportGenerator()
    return generator.generate_experiment_report(report, output_path)


def generate_literature_report(
    title: str,
    query: str,
    papers: list[dict[str, Any]],
    summary: str = "",
    key_findings: list[str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Convenience function to generate literature report.

    Args:
        title: Report title
        query: Search query used
        papers: List of paper dicts with title, authors, etc.
        summary: Optional summary
        key_findings: Optional list of key findings
        output_path: Optional output file path

    Returns:
        Rendered report string
    """
    report = LiteratureReport(
        title=title,
        date=datetime.now(),
        query=query,
        n_papers=len(papers),
        papers=papers,
        summary=summary,
        key_findings=key_findings or [],
    )

    generator = ReportGenerator()
    return generator.generate_literature_report(report, output_path)
