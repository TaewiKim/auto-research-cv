"""Literature search across multiple sources."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx

from research_automation.core.config import get_settings


@dataclass
class SearchResult:
    """Unified search result across sources."""

    title: str
    authors: list[str]
    abstract: str | None
    doi: str | None
    pmid: str | None
    arxiv_id: str | None
    url: str | None
    source: str
    journal: str | None
    publication_date: datetime | None
    citation_count: int | None = None


SourceType = Literal["pubmed", "arxiv", "semantic_scholar", "biorxiv"]


class LiteratureSearch:
    """Search multiple literature databases."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.Client(timeout=30.0)

    def search(
        self,
        query: str,
        sources: list[SourceType] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search across multiple sources."""
        if sources is None:
            sources = ["pubmed", "arxiv", "semantic_scholar"]
        if limit is None:
            limit = self.settings.search.default_limit

        results: list[SearchResult] = []

        for source in sources:
            try:
                if source == "pubmed":
                    results.extend(self.search_pubmed(query, limit))
                elif source == "arxiv":
                    results.extend(self.search_arxiv(query, limit))
                elif source == "semantic_scholar":
                    results.extend(self.search_semantic_scholar(query, limit))
                elif source == "biorxiv":
                    results.extend(self.search_biorxiv(query, limit))
            except Exception as e:
                # Log error but continue with other sources
                print(f"Error searching {source}: {e}")

        return results[:limit]

    def search_pubmed(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search PubMed."""
        results: list[SearchResult] = []

        # Step 1: Search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
        }
        if self.settings.search.pubmed_email:
            params["email"] = self.settings.search.pubmed_email

        response = self._client.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return results

        # Step 2: Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        }
        if self.settings.search.pubmed_email:
            params["email"] = self.settings.search.pubmed_email

        response = self._client.get(fetch_url, params=params)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.text)

        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find(".//MedlineCitation")
                article_data = medline.find(".//Article") if medline else None

                if article_data is None:
                    continue

                # Title
                title_elem = article_data.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "No title"

                # Authors
                authors = []
                for author in article_data.findall(".//Author"):
                    last = author.find("LastName")
                    first = author.find("ForeName")
                    if last is not None and first is not None:
                        authors.append(f"{last.text}, {first.text}")
                    elif last is not None:
                        authors.append(last.text)

                # Abstract
                abstract_elem = article_data.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else None

                # PMID
                pmid_elem = medline.find(".//PMID") if medline else None
                pmid = pmid_elem.text if pmid_elem is not None else None

                # DOI
                doi = None
                for id_elem in article.findall(".//ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text
                        break

                # Journal
                journal_elem = article_data.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else None

                # Publication date
                pub_date = None
                date_elem = article_data.find(".//PubDate")
                if date_elem is not None:
                    year = date_elem.find("Year")
                    month = date_elem.find("Month")
                    day = date_elem.find("Day")
                    if year is not None:
                        try:
                            pub_date = datetime(
                                int(year.text),
                                int(month.text) if month is not None else 1,
                                int(day.text) if day is not None else 1,
                            )
                        except (ValueError, TypeError):
                            pass

                results.append(
                    SearchResult(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        doi=doi,
                        pmid=pmid,
                        arxiv_id=None,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        source="pubmed",
                        journal=journal,
                        publication_date=pub_date,
                    )
                )
            except Exception:
                continue

        return results

    def search_arxiv(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search arXiv."""
        import arxiv

        results: list[SearchResult] = []

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for paper in client.results(search):
            arxiv_id = paper.entry_id.split("/")[-1]

            results.append(
                SearchResult(
                    title=paper.title,
                    authors=[a.name for a in paper.authors],
                    abstract=paper.summary,
                    doi=paper.doi,
                    pmid=None,
                    arxiv_id=arxiv_id,
                    url=paper.entry_id,
                    source="arxiv",
                    journal=None,
                    publication_date=paper.published,
                )
            )

        return results

    def search_semantic_scholar(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search Semantic Scholar."""
        results: list[SearchResult] = []

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,externalIds,url,venue,year,citationCount",
        }

        headers = {}
        if self.settings.search.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.search.semantic_scholar_api_key

        response = self._client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        for paper in data.get("data", []):
            external_ids = paper.get("externalIds", {})

            pub_date = None
            if paper.get("year"):
                try:
                    pub_date = datetime(paper["year"], 1, 1)
                except (ValueError, TypeError):
                    pass

            results.append(
                SearchResult(
                    title=paper.get("title", "No title"),
                    authors=[a.get("name", "") for a in paper.get("authors", [])],
                    abstract=paper.get("abstract"),
                    doi=external_ids.get("DOI"),
                    pmid=external_ids.get("PubMed"),
                    arxiv_id=external_ids.get("ArXiv"),
                    url=paper.get("url"),
                    source="semantic_scholar",
                    journal=paper.get("venue"),
                    publication_date=pub_date,
                    citation_count=paper.get("citationCount"),
                )
            )

        return results

    def search_biorxiv(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search bioRxiv/medRxiv."""
        results: list[SearchResult] = []

        # bioRxiv API - search recent preprints
        url = "https://api.biorxiv.org/details/biorxiv/2020-01-01/2030-01-01"
        response = self._client.get(url, params={"cursor": 0})

        if response.status_code != 200:
            return results

        data = response.json()

        # Filter by query terms
        query_terms = query.lower().split()

        for paper in data.get("collection", [])[:100]:  # Check first 100
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            combined = f"{title} {abstract}".lower()

            if all(term in combined for term in query_terms):
                pub_date = None
                date_str = paper.get("date")
                if date_str:
                    try:
                        pub_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass

                results.append(
                    SearchResult(
                        title=title,
                        authors=paper.get("authors", "").split("; "),
                        abstract=abstract,
                        doi=paper.get("doi"),
                        pmid=None,
                        arxiv_id=None,
                        url=f"https://www.biorxiv.org/content/{paper.get('doi')}",
                        source="biorxiv",
                        journal="bioRxiv",
                        publication_date=pub_date,
                    )
                )

                if len(results) >= limit:
                    break

        return results

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> "LiteratureSearch":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def save_results_to_db(results: list[SearchResult]) -> list[int]:
    """Save search results to database."""
    from research_automation.core.database import get_session

    from .models import Paper

    paper_ids: list[int] = []

    with get_session() as session:
        for result in results:
            # Check for existing paper
            existing = None
            if result.doi:
                existing = session.query(Paper).filter(Paper.doi == result.doi).first()
            if not existing and result.pmid:
                existing = session.query(Paper).filter(Paper.pmid == result.pmid).first()
            if not existing and result.arxiv_id:
                existing = session.query(Paper).filter(Paper.arxiv_id == result.arxiv_id).first()

            if existing:
                paper_ids.append(existing.id)
                continue

            paper = Paper(
                title=result.title,
                authors=json.dumps(result.authors),
                abstract=result.abstract,
                doi=result.doi,
                pmid=result.pmid,
                arxiv_id=result.arxiv_id,
                url=result.url,
                source=result.source,
                journal=result.journal,
                publication_date=result.publication_date,
                citation_count=result.citation_count,
            )
            session.add(paper)
            session.flush()
            paper_ids.append(paper.id)

    return paper_ids
