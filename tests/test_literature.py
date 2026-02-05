"""Tests for literature module."""

from __future__ import annotations

import json
from datetime import datetime

import pytest


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self, sample_search_result):
        """Test SearchResult creation."""
        assert sample_search_result.title.startswith("Video-based")
        assert len(sample_search_result.authors) == 2
        assert sample_search_result.source == "pubmed"

    def test_search_result_fields(self, sample_search_result):
        """Test all SearchResult fields are accessible."""
        assert sample_search_result.doi == "10.1234/example.2024"
        assert sample_search_result.pmid == "12345678"
        assert sample_search_result.arxiv_id is None
        assert sample_search_result.citation_count == 42


class TestPaperModel:
    """Tests for Paper SQLAlchemy model."""

    def test_paper_creation(self, test_db):
        """Test Paper model creation."""
        from research_automation.literature.models import Paper

        session = test_db()

        paper = Paper(
            title="Test Paper",
            authors=json.dumps(["Author One", "Author Two"]),
            abstract="Test abstract",
            doi="10.1234/test",
            source="pubmed",
        )

        session.add(paper)
        session.commit()

        assert paper.id is not None
        assert paper.title == "Test Paper"

        session.close()

    def test_paper_identifier(self, test_db):
        """Test Paper identifier property."""
        from research_automation.literature.models import Paper

        session = test_db()

        # Paper with DOI
        paper1 = Paper(title="Test 1", doi="10.1234/test1")
        session.add(paper1)
        session.flush()
        assert paper1.identifier == "10.1234/test1"

        # Paper with PMID only
        paper2 = Paper(title="Test 2", pmid="12345")
        session.add(paper2)
        session.flush()
        assert paper2.identifier == "12345"

        # Paper with arxiv only
        paper3 = Paper(title="Test 3", arxiv_id="2401.00001")
        session.add(paper3)
        session.flush()
        assert paper3.identifier == "2401.00001"

        session.close()


class TestPaperSummary:
    """Tests for PaperSummary model."""

    def test_summary_creation(self, test_db):
        """Test PaperSummary creation."""
        from research_automation.literature.models import Paper, PaperSummary

        session = test_db()

        paper = Paper(title="Test Paper", doi="10.1234/test")
        session.add(paper)
        session.flush()

        summary = PaperSummary(
            paper_id=paper.id,
            summary="This is a test summary.",
            key_findings="Finding 1\nFinding 2",
            model_used="claude-sonnet-4-20250514",
        )
        session.add(summary)
        session.commit()

        assert summary.id is not None
        assert summary.paper_id == paper.id
        assert summary.summary == "This is a test summary."

        session.close()


class TestExtraction:
    """Tests for PDF extraction."""

    def test_clean_text(self):
        """Test text cleaning function."""
        from research_automation.literature.extract import _clean_text

        text = "Line 1\n\n\n\n\nLine 2\n5\nLine 3"
        cleaned = _clean_text(text)

        assert "\n\n\n" not in cleaned

    def test_extract_sections(self):
        """Test section extraction."""
        from research_automation.literature.extract import extract_sections

        text = """
Some preamble text.

Abstract

This is the abstract of the paper.

Introduction

This introduces the topic.

Methods

We used these methods.

Results

Here are our results.

Discussion

We discuss our findings.

Conclusion

In conclusion.
"""
        sections = extract_sections(text)

        assert "abstract" in sections
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections
        assert "discussion" in sections

    def test_get_word_count(self):
        """Test word count function."""
        from research_automation.literature.extract import get_word_count

        assert get_word_count("one two three") == 3
        assert get_word_count("") == 0  # Empty string
        assert get_word_count("Hello, World!") == 2


class TestLiteratureSearch:
    """Tests for LiteratureSearch class."""

    def test_search_init(self):
        """Test LiteratureSearch initialization."""
        from research_automation.literature.search import LiteratureSearch

        searcher = LiteratureSearch()
        assert searcher.settings is not None
        searcher.close()

    def test_search_context_manager(self):
        """Test context manager protocol."""
        from research_automation.literature.search import LiteratureSearch

        with LiteratureSearch() as searcher:
            assert searcher is not None


class TestDownloader:
    """Tests for PaperDownloader."""

    def test_downloader_init(self, storage_manager):
        """Test PaperDownloader initialization."""
        from research_automation.literature.download import PaperDownloader

        downloader = PaperDownloader(storage_manager)
        assert downloader.storage == storage_manager
        downloader.close()

    def test_downloader_context_manager(self, storage_manager):
        """Test downloader context manager."""
        from research_automation.literature.download import PaperDownloader

        with PaperDownloader(storage_manager) as downloader:
            assert downloader is not None
