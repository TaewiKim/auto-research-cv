"""Paper summarization using Claude."""

from __future__ import annotations

import json
from pathlib import Path

from research_automation.core.claude import ClaudeClient, get_claude_client
from research_automation.core.database import get_session

from .extract import extract_text_from_pdf
from .models import Paper, PaperSummary


def summarize_paper(
    paper_id: int,
    focus_areas: list[str] | None = None,
    client: ClaudeClient | None = None,
) -> PaperSummary:
    """Summarize a paper and save to database."""
    client = client or get_claude_client()

    with get_session() as session:
        paper = session.query(Paper).filter(Paper.id == paper_id).first()
        if not paper:
            raise ValueError(f"Paper with id {paper_id} not found")

        # Get full text if available
        full_text = paper.full_text
        if not full_text and paper.pdf_path:
            pdf_path = Path(paper.pdf_path)
            if pdf_path.exists():
                extracted = extract_text_from_pdf(pdf_path)
                full_text = extracted.text
                paper.full_text = full_text

        # Generate summary
        result = client.summarize_paper(
            title=paper.title,
            abstract=paper.abstract or "",
            full_text=full_text,
            focus_areas=focus_areas,
        )

        # Create summary record
        summary = PaperSummary(
            paper_id=paper.id,
            summary=result.get("summary"),
            key_findings=result.get("key_findings"),
            methods=result.get("methods"),
            relevance=result.get("relevance"),
            limitations=result.get("limitations"),
            future_work=result.get("future_work"),
            model_used=client.config.model,
            focus_areas=json.dumps(focus_areas) if focus_areas else None,
        )

        session.add(summary)
        session.flush()

        # Refresh to get the ID
        session.refresh(summary)

        return summary


def summarize_pdf_file(
    pdf_path: str | Path,
    focus_areas: list[str] | None = None,
    client: ClaudeClient | None = None,
) -> dict[str, str]:
    """Summarize a PDF file directly without database."""
    client = client or get_claude_client()
    pdf_path = Path(pdf_path)

    # Extract text
    extracted = extract_text_from_pdf(pdf_path)

    # Generate summary
    return client.summarize_paper(
        title=extracted.title or pdf_path.stem,
        abstract="",  # Will be extracted from text by Claude
        full_text=extracted.text,
        focus_areas=focus_areas,
    )


def batch_summarize(
    paper_ids: list[int],
    focus_areas: list[str] | None = None,
) -> list[PaperSummary]:
    """Summarize multiple papers."""
    client = get_claude_client()
    summaries: list[PaperSummary] = []

    for paper_id in paper_ids:
        try:
            summary = summarize_paper(paper_id, focus_areas, client)
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing paper {paper_id}: {e}")
            continue

    return summaries


def get_paper_summary(paper_id: int) -> PaperSummary | None:
    """Get existing summary for a paper."""
    with get_session() as session:
        return (
            session.query(PaperSummary)
            .filter(PaperSummary.paper_id == paper_id)
            .order_by(PaperSummary.created_at.desc())
            .first()
        )


def format_summary_markdown(summary: PaperSummary) -> str:
    """Format summary as markdown."""
    lines = ["# Paper Summary\n"]

    if summary.summary:
        lines.append("## Overview\n")
        lines.append(summary.summary)
        lines.append("")

    if summary.key_findings:
        lines.append("## Key Findings\n")
        lines.append(summary.key_findings)
        lines.append("")

    if summary.methods:
        lines.append("## Methods\n")
        lines.append(summary.methods)
        lines.append("")

    if summary.relevance:
        lines.append("## Relevance\n")
        lines.append(summary.relevance)
        lines.append("")

    if summary.limitations:
        lines.append("## Limitations\n")
        lines.append(summary.limitations)
        lines.append("")

    if summary.future_work:
        lines.append("## Future Work\n")
        lines.append(summary.future_work)
        lines.append("")

    return "\n".join(lines)
