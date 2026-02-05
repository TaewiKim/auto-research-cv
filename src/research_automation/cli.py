"""CLI application using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="research",
    help="Research automation for video-based health monitoring",
    no_args_is_help=True,
)
console = Console()

# Sub-applications
config_app = typer.Typer(help="Configuration management")
lit_app = typer.Typer(help="Literature search and management")
collect_app = typer.Typer(help="Data collection utilities")

app.add_typer(config_app, name="config")
app.add_typer(lit_app, name="lit")
app.add_typer(collect_app, name="collect")


# ============================================================================
# Config commands
# ============================================================================


@config_app.command("show")
def config_show():
    """Show current configuration."""
    from research_automation.core.config import get_settings

    settings = get_settings()
    data = settings.to_dict()

    console.print("[bold]Current Configuration[/bold]\n")

    for section, values in data.items():
        console.print(f"[cyan]{section}:[/cyan]")
        if isinstance(values, dict):
            for key, value in values.items():
                # Mask sensitive values
                if "api_key" in key.lower() and value:
                    value = value[:8] + "..." if len(str(value)) > 8 else "***"
                console.print(f"  {key}: {value}")
        else:
            console.print(f"  {values}")
        console.print()


@config_app.command("init")
def config_init(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="Config file path")
    ] = Path("config/settings.yaml"),
):
    """Initialize configuration file."""
    import yaml

    from research_automation.core.config import Settings

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        console.print(f"[yellow]Config already exists at {path}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Abort()

    settings = Settings()
    data = settings.to_dict()

    # Add comments
    yaml_content = """# Research Automation Configuration

database:
  url: sqlite:///data/research.db
  echo: false

storage:
  base_dir: data
  papers_dir: data/papers
  videos_dir: data/videos
  raw_videos_dir: data/videos/raw
  processed_videos_dir: data/videos/processed
  datasets_dir: data/datasets
  cache_dir: data/cache

claude:
  # Set ANTHROPIC_API_KEY environment variable or add key here
  api_key: ""
  model: claude-sonnet-4-20250514
  max_tokens: 4096
  temperature: 0.3

youtube:
  max_duration: 600
  preferred_quality: 720p
  output_template: "%(id)s.%(ext)s"

search:
  default_limit: 20
  # Set PUBMED_EMAIL environment variable or add email here
  pubmed_email: ""
  # Optional: Set SEMANTIC_SCHOLAR_API_KEY for higher rate limits
  semantic_scholar_api_key: ""
"""

    path.write_text(yaml_content)
    console.print(f"[green]Created config at {path}[/green]")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g., claude.model)")],
    value: Annotated[str, typer.Argument(help="New value")],
):
    """Set a configuration value."""
    import yaml

    config_path = Path("config/settings.yaml")

    if not config_path.exists():
        console.print("[red]Config file not found. Run 'research config init' first.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Parse key path
    parts = key.split(".")
    current = data

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Convert value type
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            pass

    current[parts[-1]] = value

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    console.print(f"[green]Set {key} = {value}[/green]")


# ============================================================================
# Literature commands
# ============================================================================


@lit_app.command("search")
def lit_search(
    query: Annotated[str, typer.Argument(help="Search query")],
    source: Annotated[
        str, typer.Option("--source", "-s", help="Source: pubmed, arxiv, semantic_scholar")
    ] = "pubmed",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 10,
    save: Annotated[bool, typer.Option("--save", help="Save to database")] = False,
):
    """Search for papers."""
    from research_automation.core.database import init_db
    from research_automation.literature.search import LiteratureSearch, save_results_to_db

    if save:
        init_db()

    with LiteratureSearch() as searcher:
        results = searcher.search(query, sources=[source], limit=limit)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search Results: {query}")
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Authors", max_width=30)
    table.add_column("Year", justify="center")
    table.add_column("Source")
    table.add_column("ID")

    for r in results:
        year = r.publication_date.year if r.publication_date else "N/A"
        authors = ", ".join(r.authors[:2])
        if len(r.authors) > 2:
            authors += " et al."
        identifier = r.doi or r.pmid or r.arxiv_id or "-"

        table.add_row(
            r.title[:50] + "..." if len(r.title) > 50 else r.title,
            authors,
            str(year),
            r.source,
            identifier[:20] if len(identifier) > 20 else identifier,
        )

    console.print(table)

    if save:
        paper_ids = save_results_to_db(results)
        console.print(f"\n[green]Saved {len(paper_ids)} papers to database[/green]")


@lit_app.command("download")
def lit_download(
    doi: Annotated[Optional[str], typer.Option("--doi", help="Paper DOI")] = None,
    arxiv: Annotated[Optional[str], typer.Option("--arxiv", help="arXiv ID")] = None,
    pmid: Annotated[Optional[str], typer.Option("--pmid", help="PubMed ID")] = None,
    url: Annotated[Optional[str], typer.Option("--url", help="Direct PDF URL")] = None,
):
    """Download a paper PDF."""
    from research_automation.literature.download import download_paper

    if not any([doi, arxiv, pmid, url]):
        console.print("[red]Provide at least one identifier (--doi, --arxiv, --pmid, or --url)[/red]")
        raise typer.Exit(1)

    with console.status("Downloading..."):
        path = download_paper(doi=doi, arxiv_id=arxiv, pmid=pmid, url=url)

    if path:
        console.print(f"[green]Downloaded to: {path}[/green]")
    else:
        console.print("[red]Download failed. Paper may not be open access.[/red]")


@lit_app.command("summarize")
def lit_summarize(
    paper_id: Annotated[int, typer.Option("--paper-id", "-p", help="Paper ID from database")],
    focus: Annotated[
        Optional[str], typer.Option("--focus", "-f", help="Focus areas (comma-separated)")
    ] = None,
):
    """Summarize a paper using Claude."""
    from research_automation.core.database import init_db
    from research_automation.literature.summarize import format_summary_markdown, summarize_paper

    init_db()

    focus_areas = [f.strip() for f in focus.split(",")] if focus else None

    with console.status("Generating summary..."):
        try:
            summary = summarize_paper(paper_id, focus_areas)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    console.print(format_summary_markdown(summary))


@lit_app.command("list")
def lit_list(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max papers to show")] = 20,
):
    """List papers in database."""
    from research_automation.core.database import get_session, init_db
    from research_automation.literature.models import Paper

    init_db()

    with get_session() as session:
        papers = session.query(Paper).order_by(Paper.created_at.desc()).limit(limit).all()

    if not papers:
        console.print("[yellow]No papers in database[/yellow]")
        return

    table = Table(title="Papers in Database")
    table.add_column("ID", justify="right")
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Source")
    table.add_column("PDF")

    for p in papers:
        table.add_row(
            str(p.id),
            p.title[:50] + "..." if len(p.title) > 50 else p.title,
            p.source or "-",
            "Yes" if p.pdf_path else "No",
        )

    console.print(table)


# ============================================================================
# Collection commands
# ============================================================================


@collect_app.command("youtube")
def collect_youtube(
    query: Annotated[str, typer.Argument(help="Search query")],
    max_results: Annotated[int, typer.Option("--max", "-m", help="Max results")] = 5,
    download: Annotated[bool, typer.Option("--download", "-d", help="Download videos")] = False,
):
    """Search and optionally download YouTube videos."""
    from research_automation.collection.youtube import YouTubeCollector

    collector = YouTubeCollector()

    with console.status("Searching YouTube..."):
        results = collector.search(query, max_results)

    if not results:
        console.print("[yellow]No videos found[/yellow]")
        return

    table = Table(title=f"YouTube Results: {query}")
    table.add_column("ID", style="cyan")
    table.add_column("Title", max_width=40)
    table.add_column("Duration")
    table.add_column("Channel", max_width=20)
    table.add_column("Views")

    for v in results:
        duration = f"{v.duration // 60}:{v.duration % 60:02d}"
        views = f"{v.view_count:,}" if v.view_count else "N/A"

        table.add_row(
            v.video_id,
            v.title[:40] + "..." if len(v.title) > 40 else v.title,
            duration,
            v.channel[:20] if v.channel else "N/A",
            views,
        )

    console.print(table)

    if download:
        console.print("\n[bold]Downloading videos...[/bold]")
        for v in results:
            with console.status(f"Downloading {v.video_id}..."):
                result = collector.download(v.video_id)

            if result.success:
                console.print(f"[green]Downloaded: {result.path}[/green]")
            else:
                console.print(f"[red]Failed: {result.error}[/red]")


@collect_app.command("quality-check")
def collect_quality_check(
    path: Annotated[Path, typer.Argument(help="Video file or directory")],
    sample_rate: Annotated[
        int, typer.Option("--sample-rate", "-s", help="Check every Nth frame")
    ] = 30,
):
    """Check video quality for research use."""
    from research_automation.collection.quality import (
        VideoQualityChecker,
        format_quality_report,
    )

    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    with VideoQualityChecker(sample_rate=sample_rate) as checker:
        if path.is_file():
            with console.status("Analyzing video..."):
                metrics = checker.check_video(path)
            console.print(format_quality_report(metrics))
        else:
            with console.status("Analyzing directory..."):
                results = checker.check_directory(path)

            if not results:
                console.print("[yellow]No videos found[/yellow]")
                return

            table = Table(title="Quality Check Results")
            table.add_column("File", max_width=30)
            table.add_column("Duration")
            table.add_column("Resolution")
            table.add_column("Face %")
            table.add_column("Pose %")
            table.add_column("Overall")
            table.add_column("Usable")

            for filepath, m in results.items():
                filename = Path(filepath).name
                table.add_row(
                    filename[:30],
                    f"{m.duration:.1f}s",
                    f"{m.resolution[0]}x{m.resolution[1]}",
                    f"{m.face_detection_rate*100:.0f}%",
                    f"{m.pose_detection_rate*100:.0f}%",
                    f"{m.overall_score*100:.0f}%",
                    "[green]Yes[/green]" if m.is_usable else "[red]No[/red]",
                )

            console.print(table)


@collect_app.command("dataset")
def collect_dataset(
    action: Annotated[str, typer.Argument(help="Action: list, info, download")],
    name: Annotated[Optional[str], typer.Argument(help="Dataset name")] = None,
    category: Annotated[Optional[str], typer.Option("--category", "-c", help="Filter by category")] = None,
):
    """Manage research datasets."""
    from research_automation.collection.datasets import (
        AccessType,
        DatasetCategory,
        DatasetManager,
    )

    manager = DatasetManager()

    if action == "list":
        cat = DatasetCategory(category) if category else None
        datasets = manager.list_datasets(category=cat)

        table = Table(title="Available Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Category")
        table.add_column("Access")
        table.add_column("Size")
        table.add_column("Subjects")
        table.add_column("Downloaded")

        for d in datasets:
            is_downloaded = manager.is_downloaded(d.name.lower().replace(" ", "-"))
            table.add_row(
                d.name,
                d.category.value,
                d.access_type.value,
                d.size,
                str(d.subjects) if d.subjects else "-",
                "[green]Yes[/green]" if is_downloaded else "No",
            )

        console.print(table)

    elif action == "info":
        if not name:
            console.print("[red]Dataset name required[/red]")
            raise typer.Exit(1)

        dataset = manager.get_dataset(name)
        if not dataset:
            console.print(f"[red]Dataset not found: {name}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold cyan]{dataset.name}[/bold cyan]")
        console.print(f"\n{dataset.description}\n")
        console.print(f"Category: {dataset.category.value}")
        console.print(f"Access: {dataset.access_type.value}")
        console.print(f"Size: {dataset.size}")
        if dataset.subjects:
            console.print(f"Subjects: {dataset.subjects}")
        console.print(f"URL: {dataset.url}")
        if dataset.tags:
            console.print(f"Tags: {', '.join(dataset.tags)}")
        if dataset.notes:
            console.print(f"\nNotes: {dataset.notes}")

    elif action == "download":
        if not name:
            console.print("[red]Dataset name required[/red]")
            raise typer.Exit(1)

        dataset = manager.get_dataset(name)
        if not dataset:
            console.print(f"[red]Dataset not found: {name}[/red]")
            raise typer.Exit(1)

        if dataset.access_type != AccessType.OPEN:
            console.print(
                f"[yellow]This dataset requires {dataset.access_type.value} access.[/yellow]"
            )
            console.print(f"Visit: {dataset.url}")
            raise typer.Exit(1)

        if not dataset.huggingface_id:
            console.print("[yellow]Automatic download not supported for this dataset.[/yellow]")
            console.print(f"Visit: {dataset.download_url or dataset.url}")
            raise typer.Exit(1)

        with console.status(f"Downloading {dataset.name}..."):
            path = manager.download_huggingface(dataset.huggingface_id)

        console.print(f"[green]Downloaded to: {path}[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@collect_app.command("scale")
def collect_scale(
    action: Annotated[str, typer.Argument(help="Action: list, show")],
    name: Annotated[Optional[str], typer.Argument(help="Scale name or abbreviation")] = None,
):
    """View clinical assessment scales."""
    from research_automation.collection.questionnaire import format_scale, get_scale, list_scales

    if action == "list":
        scales = list_scales()

        table = Table(title="Clinical Assessment Scales")
        table.add_column("Abbreviation", style="cyan")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Items")
        table.add_column("Range")

        for s in scales:
            table.add_row(
                s.abbreviation,
                s.name[:50] + "..." if len(s.name) > 50 else s.name,
                s.scale_type.value,
                str(s.item_count),
                f"{s.total_min}-{s.total_max}",
            )

        console.print(table)

    elif action == "show":
        if not name:
            console.print("[red]Scale name required[/red]")
            raise typer.Exit(1)

        scale = get_scale(name)
        if not scale:
            console.print(f"[red]Scale not found: {name}[/red]")
            raise typer.Exit(1)

        console.print(format_scale(scale))

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


# ============================================================================
# Main entry point
# ============================================================================


@app.callback()
def main():
    """Research automation for video-based health monitoring."""
    pass


if __name__ == "__main__":
    app()
