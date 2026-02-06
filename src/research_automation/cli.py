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
pipeline_app = typer.Typer(help="Feature extraction pipeline")
experiment_app = typer.Typer(help="Experiment tracking")
report_app = typer.Typer(help="Report generation")

app.add_typer(config_app, name="config")
app.add_typer(lit_app, name="lit")
app.add_typer(collect_app, name="collect")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(experiment_app, name="experiment")
app.add_typer(report_app, name="report")


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
# Pipeline commands
# ============================================================================


@pipeline_app.command("extract-pose")
def pipeline_extract_pose(
    video_path: Annotated[Path, typer.Argument(help="Video file path")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output JSON path")] = None,
    sample_rate: Annotated[int, typer.Option("--sample-rate", "-s", help="Sample every Nth frame")] = 1,
    show_progress: Annotated[bool, typer.Option("--progress", "-p", help="Show progress bar")] = True,
):
    """Extract pose features from a video."""
    import json

    from research_automation.pipeline.extractors import PoseExtractor, extract_gait_features

    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/red]")
        raise typer.Exit(1)

    extractor = PoseExtractor(sample_rate=sample_rate)

    with console.status("Extracting pose..."):
        sequence = extractor.extract_from_video(str(video_path))

    if not sequence.frames:
        console.print("[yellow]No poses detected in video[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Extracted {len(sequence.frames)} pose frames[/green]")
    console.print(f"Duration: {sequence.duration_seconds:.2f}s")
    console.print(f"Avg confidence: {sequence.average_confidence:.3f}")

    # Extract gait features
    with console.status("Computing gait features..."):
        gait = extract_gait_features(sequence)

    if gait:
        console.print("\n[bold]Gait Features:[/bold]")
        console.print(f"  Speed: {gait.speed_mean:.3f} ± {gait.speed_std:.3f}")
        console.print(f"  Cadence: {gait.cadence:.2f} steps/min")
        console.print(f"  Step length: {gait.step_length_mean:.3f}")
        console.print(f"  Hip asymmetry: {gait.hip_asymmetry:.3f}")

    if output:
        data = {
            "video_path": str(video_path),
            "n_frames": len(sequence.frames),
            "duration": sequence.duration_seconds,
            "avg_confidence": sequence.average_confidence,
            "gait_features": gait.to_dict() if gait else None,
        }
        output.write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Saved to: {output}[/green]")


@pipeline_app.command("extract-face")
def pipeline_extract_face(
    video_path: Annotated[Path, typer.Argument(help="Video file path")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output JSON path")] = None,
    sample_rate: Annotated[int, typer.Option("--sample-rate", "-s", help="Sample every Nth frame")] = 1,
):
    """Extract facial features from a video."""
    import json

    from research_automation.pipeline.extractors import FaceExtractor, extract_facial_features

    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/red]")
        raise typer.Exit(1)

    extractor = FaceExtractor(sample_rate=sample_rate)

    with console.status("Extracting faces..."):
        sequence = extractor.extract_from_video(str(video_path))

    if not sequence.frames:
        console.print("[yellow]No faces detected in video[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Extracted {len(sequence.frames)} face frames[/green]")
    console.print(f"Duration: {sequence.duration_seconds:.2f}s")
    console.print(f"Avg confidence: {sequence.average_confidence:.3f}")

    # Extract facial features
    with console.status("Computing facial features..."):
        features = extract_facial_features(sequence)

    if features:
        console.print("\n[bold]Facial Features:[/bold]")
        console.print(f"  Blink rate: {features.blink_rate:.2f}/min")
        console.print(f"  Eye openness: {features.eye_openness_mean:.3f}")
        console.print(f"  Mouth openness: {features.mouth_openness_mean:.3f}")
        console.print(f"  Face asymmetry: {features.face_asymmetry:.3f}")

        if features.au_activations:
            console.print("\n[bold]Action Units:[/bold]")
            for au, val in sorted(features.au_activations.items())[:5]:
                console.print(f"  {au}: {val:.3f}")

    if output:
        data = {
            "video_path": str(video_path),
            "n_frames": len(sequence.frames),
            "duration": sequence.duration_seconds,
            "avg_confidence": sequence.average_confidence,
            "facial_features": features.to_dict() if features else None,
        }
        output.write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Saved to: {output}[/green]")


@pipeline_app.command("batch-extract")
def pipeline_batch_extract(
    video_dir: Annotated[Path, typer.Argument(help="Directory with video files")],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("data/features"),
    feature_type: Annotated[str, typer.Option("--type", "-t", help="Feature type: pose, face, both")] = "both",
    sample_rate: Annotated[int, typer.Option("--sample-rate", "-s", help="Sample every Nth frame")] = 2,
):
    """Batch extract features from multiple videos."""
    import json

    from research_automation.pipeline.extractors import (
        FaceExtractor,
        PoseExtractor,
        extract_facial_features,
        extract_gait_features,
    )

    if not video_dir.exists():
        console.print(f"[red]Directory not found: {video_dir}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))

    if not video_files:
        console.print("[yellow]No video files found[/yellow]")
        return

    console.print(f"Found {len(video_files)} videos")

    pose_extractor = PoseExtractor(sample_rate=sample_rate) if feature_type in ("pose", "both") else None
    face_extractor = FaceExtractor(sample_rate=sample_rate) if feature_type in ("face", "both") else None

    results = []
    for video_path in video_files:
        console.print(f"\nProcessing: {video_path.name}")
        result = {"video": video_path.name}

        if pose_extractor:
            try:
                sequence = pose_extractor.extract_from_video(str(video_path))
                gait = extract_gait_features(sequence) if sequence.frames else None
                result["pose_frames"] = len(sequence.frames)
                result["gait_features"] = gait.to_dict() if gait else None
                console.print(f"  [green]Pose: {len(sequence.frames)} frames[/green]")
            except Exception as e:
                console.print(f"  [red]Pose extraction failed: {e}[/red]")
                result["pose_error"] = str(e)

        if face_extractor:
            try:
                sequence = face_extractor.extract_from_video(str(video_path))
                features = extract_facial_features(sequence) if sequence.frames else None
                result["face_frames"] = len(sequence.frames)
                result["facial_features"] = features.to_dict() if features else None
                console.print(f"  [green]Face: {len(sequence.frames)} frames[/green]")
            except Exception as e:
                console.print(f"  [red]Face extraction failed: {e}[/red]")
                result["face_error"] = str(e)

        results.append(result)

    output_file = output_dir / "batch_features.json"
    output_file.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]Results saved to: {output_file}[/green]")


# ============================================================================
# Experiment commands
# ============================================================================


@experiment_app.command("list")
def experiment_list(
    experiment_name: Annotated[str, typer.Argument(help="Experiment name")],
    tracking_uri: Annotated[str, typer.Option("--uri", "-u", help="MLflow tracking URI")] = "mlruns",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max runs to show")] = 20,
):
    """List runs in an experiment."""
    from research_automation.experiment import ExperimentTracker

    tracker = ExperimentTracker(experiment_name, tracking_uri)
    runs = tracker.list_runs(max_results=limit)

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    table = Table(title=f"Runs: {experiment_name}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Metrics")

    for run in runs:
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in list(run.data.metrics.items())[:3])
        table.add_row(
            run.info.run_id[:8],
            run.info.run_name or "-",
            run.info.status,
            metrics_str or "-",
        )

    console.print(table)


@experiment_app.command("compare")
def experiment_compare(
    experiment_name: Annotated[str, typer.Argument(help="Experiment name")],
    metrics: Annotated[str, typer.Option("--metrics", "-m", help="Metrics to compare (comma-separated)")] = "accuracy,f1",
    tracking_uri: Annotated[str, typer.Option("--uri", "-u", help="MLflow tracking URI")] = "mlruns",
):
    """Compare runs in an experiment."""
    from research_automation.experiment import ExperimentTracker

    tracker = ExperimentTracker(experiment_name, tracking_uri)
    runs = tracker.list_runs(max_results=50)

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    metric_list = [m.strip() for m in metrics.split(",")]

    table = Table(title=f"Comparison: {experiment_name}")
    table.add_column("Run", style="cyan")
    table.add_column("Name")
    for m in metric_list:
        table.add_column(m, justify="right")

    for run in runs:
        row = [run.info.run_id[:8], run.info.run_name or "-"]
        for m in metric_list:
            val = run.data.metrics.get(m)
            row.append(f"{val:.4f}" if val is not None else "-")
        table.add_row(*row)

    console.print(table)


@experiment_app.command("best")
def experiment_best(
    experiment_name: Annotated[str, typer.Argument(help="Experiment name")],
    metric: Annotated[str, typer.Option("--metric", "-m", help="Metric to optimize")] = "accuracy",
    minimize: Annotated[bool, typer.Option("--minimize", help="Minimize metric instead of maximize")] = False,
    tracking_uri: Annotated[str, typer.Option("--uri", "-u", help="MLflow tracking URI")] = "mlruns",
):
    """Get the best run by a metric."""
    from research_automation.experiment import ExperimentResult, ExperimentTracker

    tracker = ExperimentTracker(experiment_name, tracking_uri)
    best_run = tracker.get_best_run(metric, maximize=not minimize)

    if not best_run:
        console.print("[yellow]No runs found[/yellow]")
        return

    result = ExperimentResult.from_run(best_run, experiment_name)

    console.print(f"\n[bold cyan]Best Run: {result.run_name or result.run_id[:8]}[/bold cyan]")
    console.print(f"\nRun ID: {result.run_id}")
    console.print(f"Status: {result.status}")
    console.print(f"Started: {result.start_time}")

    console.print("\n[bold]Parameters:[/bold]")
    for k, v in result.params.items():
        console.print(f"  {k}: {v}")

    console.print("\n[bold]Metrics:[/bold]")
    for k, v in sorted(result.metrics.items()):
        highlight = " [green]← best[/green]" if k == metric else ""
        console.print(f"  {k}: {v:.4f}{highlight}")


@experiment_app.command("run-baseline")
def experiment_run_baseline(
    dataset_path: Annotated[Path, typer.Argument(help="Path to CARE-PD dataset")],
    experiment_name: Annotated[str, typer.Option("--name", "-n", help="Experiment name")] = "updrs-baseline",
    tracking_uri: Annotated[str, typer.Option("--uri", "-u", help="MLflow tracking URI")] = "mlruns",
):
    """Run UPDRS prediction baseline on CARE-PD dataset."""
    from research_automation.experiment import ExperimentResult, ExperimentTracker
    from research_automation.pipeline.gait_baseline import (
        UPDRSBaseline,
        load_care_pd_features,
    )

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    console.print("[bold]Loading CARE-PD dataset...[/bold]")
    X, y_multiclass, y_binary = load_care_pd_features(dataset_path)
    console.print(f"Samples: {len(X)}, Features: {X.shape[1] if len(X) > 0 else 0}")

    tracker = ExperimentTracker(experiment_name, tracking_uri)
    baseline = UPDRSBaseline()

    with tracker.start_run("rf-baseline") as run:
        tracker.log_params({
            "model": "RandomForest",
            "n_estimators": 100,
            "dataset": str(dataset_path),
            "n_samples": len(X),
        })

        console.print("\n[bold]Training multiclass model...[/bold]")
        with console.status("Training..."):
            metrics = baseline.train(X, y_multiclass)

        tracker.log_metrics(metrics)
        console.print(f"Multiclass accuracy: {metrics['accuracy']:.3f}")

        console.print("\n[bold]Training binary model...[/bold]")
        with console.status("Training..."):
            binary_metrics = baseline.train_binary(X, y_binary)

        tracker.log_metrics({f"binary_{k}": v for k, v in binary_metrics.items()})
        console.print(f"Binary accuracy: {binary_metrics['accuracy']:.3f}")
        console.print(f"Binary ROC-AUC: {binary_metrics['roc_auc']:.3f}")

    result = ExperimentResult.from_run(tracker.get_run(run.info.run_id), experiment_name)
    console.print(f"\n[green]Run completed: {result.run_id[:8]}[/green]")


# ============================================================================
# Report commands
# ============================================================================


@report_app.command("experiment")
def report_experiment(
    name: Annotated[str, typer.Argument(help="Report name")],
    description: Annotated[str, typer.Option("--desc", "-d", help="Description")] = "",
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset name")] = "",
    methods: Annotated[str, typer.Option("--methods", "-m", help="Methods description")] = "",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file")] = None,
):
    """Generate an experiment report from MLflow."""
    from research_automation.experiment import ExperimentResult, ExperimentTracker
    from research_automation.report import generate_experiment_report

    tracker = ExperimentTracker(name)
    best_run = tracker.get_best_run("accuracy")

    if not best_run:
        console.print("[yellow]No runs found for this experiment[/yellow]")
        raise typer.Exit(1)

    result = ExperimentResult.from_run(best_run, name)

    report_md = generate_experiment_report(
        name=name,
        description=description or f"Experiment run: {result.run_name}",
        dataset=dataset or result.params.get("dataset", "Unknown"),
        methods=methods or f"Model: {result.params.get('model', 'Unknown')}",
        metrics=result.metrics,
        results=result.params,
        output_path=output,
    )

    if output:
        console.print(f"[green]Report saved to: {output}[/green]")
    else:
        console.print(report_md)


@report_app.command("literature")
def report_literature(
    title: Annotated[str, typer.Argument(help="Report title")],
    query: Annotated[str, typer.Option("--query", "-q", help="Search query used")] = "",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of papers")] = 20,
):
    """Generate a literature review report from database."""
    from research_automation.core.database import get_session, init_db
    from research_automation.literature.models import Paper
    from research_automation.report import generate_literature_report

    init_db()

    with get_session() as session:
        papers = session.query(Paper).order_by(Paper.created_at.desc()).limit(limit).all()

    if not papers:
        console.print("[yellow]No papers in database[/yellow]")
        raise typer.Exit(1)

    paper_data = [
        {
            "title": p.title,
            "authors": p.authors.split(", ") if p.authors else [],
            "source": p.source or "Unknown",
            "doi": p.doi,
            "abstract": p.abstract or "",
        }
        for p in papers
    ]

    report_md = generate_literature_report(
        title=title,
        query=query or "Database papers",
        papers=paper_data,
        output_path=output,
    )

    if output:
        console.print(f"[green]Report saved to: {output}[/green]")
    else:
        console.print(report_md)


# ============================================================================
# Main entry point
# ============================================================================


@app.callback()
def main():
    """Research automation for video-based health monitoring."""
    pass


if __name__ == "__main__":
    app()
