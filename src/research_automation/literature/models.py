"""SQLAlchemy models for literature management."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from research_automation.core.database import Base

if TYPE_CHECKING:
    pass


class Paper(Base):
    """Research paper model."""

    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[str | None] = mapped_column(Text)  # JSON array as string
    abstract: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(100), unique=True, index=True)
    pmid: Mapped[str | None] = mapped_column(String(20), unique=True, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(50), unique=True, index=True)
    url: Mapped[str | None] = mapped_column(String(500))
    pdf_path: Mapped[str | None] = mapped_column(String(500))
    source: Mapped[str | None] = mapped_column(String(50))  # pubmed, arxiv, semantic_scholar
    journal: Mapped[str | None] = mapped_column(String(300))
    publication_date: Mapped[datetime | None] = mapped_column(DateTime)
    citation_count: Mapped[int | None] = mapped_column(Integer, default=0)
    full_text: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[str | None] = mapped_column(Text)  # JSON array as string
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    summaries: Mapped[list["PaperSummary"]] = relationship(
        "PaperSummary", back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Paper(id={self.id}, title='{self.title[:50]}...')>"

    @property
    def identifier(self) -> str:
        """Get primary identifier for the paper."""
        return self.doi or self.pmid or self.arxiv_id or str(self.id)


class PaperSummary(Base):
    """AI-generated paper summary."""

    __tablename__ = "paper_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id"), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    key_findings: Mapped[str | None] = mapped_column(Text)
    methods: Mapped[str | None] = mapped_column(Text)
    relevance: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[str | None] = mapped_column(Text)
    future_work: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(100))
    focus_areas: Mapped[str | None] = mapped_column(Text)  # JSON array as string
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    paper: Mapped["Paper"] = relationship("Paper", back_populates="summaries")

    def __repr__(self) -> str:
        return f"<PaperSummary(id={self.id}, paper_id={self.paper_id})>"
