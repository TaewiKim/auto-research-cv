"""Anthropic Claude API wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from .config import ClaudeConfig


class ClaudeClient:
    """Wrapper for Anthropic Claude API."""

    def __init__(self, config: ClaudeConfig | None = None) -> None:
        """Initialize Claude client."""
        if config is None:
            from .config import get_settings

            config = get_settings().claude

        self.config = config
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(
                    "Anthropic API key not set. "
                    "Set ANTHROPIC_API_KEY environment variable or configure in settings."
                )
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate completion from Claude."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.config.temperature

        response = self.client.messages.create(**kwargs)

        # Extract text from response
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    def summarize_paper(
        self,
        title: str,
        abstract: str,
        full_text: str | None = None,
        focus_areas: list[str] | None = None,
    ) -> dict[str, str]:
        """Summarize a research paper."""
        focus_str = ""
        if focus_areas:
            focus_str = f"\n\nFocus especially on: {', '.join(focus_areas)}"

        content_parts = [f"Title: {title}", f"Abstract: {abstract}"]
        if full_text:
            # Truncate if too long
            max_text_len = 50000
            if len(full_text) > max_text_len:
                full_text = full_text[:max_text_len] + "\n\n[Truncated...]"
            content_parts.append(f"Full Text:\n{full_text}")

        prompt = f"""Analyze this research paper and provide a structured summary.

{chr(10).join(content_parts)}
{focus_str}

Provide your analysis in the following format:

SUMMARY:
[2-3 sentence overview of the paper]

KEY_FINDINGS:
[Bullet points of main findings/contributions]

METHODS:
[Brief description of methodology]

RELEVANCE:
[How this relates to video-based health monitoring, if applicable]

LIMITATIONS:
[Any noted limitations or concerns]

FUTURE_WORK:
[Suggested future directions mentioned]"""

        system = """You are a research assistant specializing in biomedical engineering
and computer vision. Provide concise, accurate summaries focusing on technical details
and clinical relevance."""

        response = self.complete(prompt, system=system)

        # Parse response into sections
        sections = {
            "summary": "",
            "key_findings": "",
            "methods": "",
            "relevance": "",
            "limitations": "",
            "future_work": "",
        }

        current_section = None
        current_content: list[str] = []

        for line in response.split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("SUMMARY:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "summary"
                current_content = [line.split(":", 1)[1].strip()] if ":" in line else []
            elif line_upper.startswith("KEY_FINDINGS:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "key_findings"
                current_content = []
            elif line_upper.startswith("METHODS:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "methods"
                current_content = []
            elif line_upper.startswith("RELEVANCE:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "relevance"
                current_content = []
            elif line_upper.startswith("LIMITATIONS:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "limitations"
                current_content = []
            elif line_upper.startswith("FUTURE_WORK:"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "future_work"
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def extract_clinical_scales(self, text: str) -> dict[str, list[str]]:
        """Extract clinical scale items from text."""
        prompt = f"""Extract clinical assessment scale items from this text.
Identify any standardized clinical scales (e.g., MDS-UPDRS, H&Y, MMSE, etc.)
and their individual items or subsections.

Text:
{text[:30000]}

Return in format:
SCALE_NAME:
- Item 1
- Item 2
...

If no clinical scales found, respond with "NO_SCALES_FOUND"."""

        response = self.complete(prompt)

        if "NO_SCALES_FOUND" in response:
            return {}

        scales: dict[str, list[str]] = {}
        current_scale = None
        items: list[str] = []

        for line in response.split("\n"):
            line = line.strip()
            if line.endswith(":") and not line.startswith("-"):
                if current_scale and items:
                    scales[current_scale] = items
                current_scale = line[:-1]
                items = []
            elif line.startswith("-") and current_scale:
                items.append(line[1:].strip())

        if current_scale and items:
            scales[current_scale] = items

        return scales


# Global client instance
_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    """Get or create global Claude client."""
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client
