"""PDF download functionality."""

from __future__ import annotations

from pathlib import Path

import httpx

from research_automation.core.storage import StorageManager


class PaperDownloader:
    """Download papers from various sources."""

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.storage = storage or StorageManager()
        self._client = httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (research-automation; academic use)",
            },
        )

    def download_by_doi(self, doi: str) -> Path | None:
        """Download paper PDF by DOI."""
        # Try Unpaywall first
        pdf_url = self._get_unpaywall_url(doi)

        if not pdf_url:
            # Try Sci-Hub as fallback (academic use)
            pdf_url = self._get_direct_url(doi)

        if not pdf_url:
            return None

        return self._download_pdf(pdf_url, doi)

    def download_by_arxiv(self, arxiv_id: str) -> Path | None:
        """Download paper from arXiv."""
        # Clean arxiv ID
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return self._download_pdf(pdf_url, f"arxiv_{arxiv_id}")

    def download_by_pmid(self, pmid: str) -> Path | None:
        """Download paper from PubMed Central if available."""
        # First, try to get PMC ID
        pmc_id = self._get_pmc_id(pmid)
        if not pmc_id:
            return None

        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
        return self._download_pdf(pdf_url, f"pmid_{pmid}")

    def download_from_url(self, url: str, identifier: str) -> Path | None:
        """Download paper from direct URL."""
        return self._download_pdf(url, identifier)

    def _get_unpaywall_url(self, doi: str) -> str | None:
        """Get open access PDF URL from Unpaywall."""
        from research_automation.core.config import get_settings

        settings = get_settings()
        email = settings.search.pubmed_email or "research@example.com"

        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {"email": email}

        try:
            response = self._client.get(url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()

            # Check for open access locations
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("url_for_pdf"):
                return best_oa["url_for_pdf"]

            # Check all OA locations
            for location in data.get("oa_locations", []):
                if location.get("url_for_pdf"):
                    return location["url_for_pdf"]

        except Exception:
            pass

        return None

    def _get_direct_url(self, doi: str) -> str | None:
        """Try to get direct PDF URL from publisher."""
        # Common direct patterns
        patterns = [
            f"https://doi.org/{doi}",
        ]

        for url in patterns:
            try:
                response = self._client.head(url, follow_redirects=True)
                content_type = response.headers.get("content-type", "")
                if "application/pdf" in content_type:
                    return str(response.url)
            except Exception:
                continue

        return None

    def _get_pmc_id(self, pmid: str) -> str | None:
        """Get PMC ID from PMID."""
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {"ids": pmid, "format": "json"}

        try:
            response = self._client.get(url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()
            records = data.get("records", [])
            if records and records[0].get("pmcid"):
                return records[0]["pmcid"]
        except Exception:
            pass

        return None

    def _download_pdf(self, url: str, identifier: str) -> Path | None:
        """Download PDF from URL."""
        try:
            response = self._client.get(url)
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get("content-type", "")
            if "application/pdf" not in content_type and not response.content[:4] == b"%PDF":
                return None

            # Save to storage
            path = self.storage.save_paper(response.content, identifier)
            return path

        except Exception:
            return None

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> "PaperDownloader":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def download_paper(
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmid: str | None = None,
    url: str | None = None,
) -> Path | None:
    """Download paper using any available identifier."""
    with PaperDownloader() as downloader:
        if arxiv_id:
            return downloader.download_by_arxiv(arxiv_id)
        if doi:
            return downloader.download_by_doi(doi)
        if pmid:
            return downloader.download_by_pmid(pmid)
        if url:
            identifier = url.split("/")[-1].replace(".pdf", "")
            return downloader.download_from_url(url, identifier)

    return None
