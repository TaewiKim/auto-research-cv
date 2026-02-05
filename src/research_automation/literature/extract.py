"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedContent:
    """Extracted content from a PDF."""

    text: str
    title: str | None
    metadata: dict
    page_count: int
    has_images: bool


def extract_text_from_pdf(pdf_path: str | Path) -> ExtractedContent:
    """Extract text and metadata from a PDF file."""
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    # Extract metadata
    metadata = doc.metadata or {}
    title = metadata.get("title") or _extract_title_from_first_page(doc)

    # Extract text from all pages
    text_parts: list[str] = []
    has_images = False

    for page in doc:
        text_parts.append(page.get_text())

        # Check for images
        if page.get_images():
            has_images = True

    full_text = "\n\n".join(text_parts)

    # Clean up text
    full_text = _clean_text(full_text)

    result = ExtractedContent(
        text=full_text,
        title=title,
        metadata=metadata,
        page_count=len(doc),
        has_images=has_images,
    )

    doc.close()
    return result


def _extract_title_from_first_page(doc) -> str | None:
    """Try to extract title from first page."""
    if len(doc) == 0:
        return None

    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    # Find the largest text on the first page (likely the title)
    max_size = 0
    title_text = None

    for block in blocks:
        if block.get("type") != 0:  # Text block
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                text = span.get("text", "").strip()

                if size > max_size and len(text) > 10:
                    max_size = size
                    title_text = text

    return title_text


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    import re

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove page numbers (common patterns)
    text = re.sub(r"\n\d+\n", "\n", text)

    # Remove header/footer artifacts
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip very short lines that might be page numbers or headers
        if len(line) < 3 and line.isdigit():
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_sections(text: str) -> dict[str, str]:
    """Extract common paper sections from text."""
    import re

    sections: dict[str, str] = {}

    # Common section headers
    section_patterns = [
        (r"(?i)^abstract\s*$", "abstract"),
        (r"(?i)^introduction\s*$", "introduction"),
        (r"(?i)^(related work|background)\s*$", "background"),
        (r"(?i)^(methods?|methodology|materials and methods)\s*$", "methods"),
        (r"(?i)^results?\s*$", "results"),
        (r"(?i)^discussion\s*$", "discussion"),
        (r"(?i)^conclusions?\s*$", "conclusion"),
        (r"(?i)^references?\s*$", "references"),
        (r"(?i)^acknowledgments?\s*$", "acknowledgments"),
    ]

    lines = text.split("\n")
    current_section = "preamble"
    current_content: list[str] = []

    for line in lines:
        line_stripped = line.strip()

        # Check if this line is a section header
        found_section = None
        for pattern, section_name in section_patterns:
            if re.match(pattern, line_stripped):
                found_section = section_name
                break

        if found_section:
            # Save previous section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = found_section
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def get_word_count(text: str) -> int:
    """Get word count of text."""
    return len(text.split())


def extract_references(text: str) -> list[str]:
    """Extract references from text."""
    import re

    sections = extract_sections(text)
    ref_text = sections.get("references", "")

    if not ref_text:
        return []

    # Split by common reference patterns
    # [1] Author et al...
    # 1. Author et al...
    refs = re.split(r"\n(?=\[\d+\]|\d+\.)", ref_text)

    # Clean up
    references = []
    for ref in refs:
        ref = ref.strip()
        if len(ref) > 20:  # Filter out noise
            references.append(ref)

    return references
