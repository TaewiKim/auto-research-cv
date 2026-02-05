# Research Automation

Video-based health monitoring research automation toolkit.

## Features

- **Literature Module**: Search, download, and summarize research papers from PubMed, arXiv, Semantic Scholar
- **Collection Module**: YouTube video search/download, quality assessment, dataset management
- **Clinical Scales**: MDS-UPDRS, Hoehn & Yahr, House-Brackmann and more
- **ECG Processing**: Load and analyze ECG signals (optional)

## Installation

```bash
# Using uv
uv sync

# With MediaPipe for full pose/face detection
uv sync --extra cv

# Development
uv sync --extra dev
```

## Quick Start

```bash
# Initialize configuration
research config init

# Search literature
research lit search "Parkinson gait video" --source pubmed --limit 10

# Download paper
research lit download --arxiv 2311.09890

# Search YouTube videos
research collect youtube "Parkinson tremor" --max 5

# Check video quality
research collect quality-check data/videos/raw/

# List available datasets
research collect dataset list

# View clinical scales
research collect scale list
research collect scale show MDS-UPDRS-III
```

## Configuration

Set environment variables or edit `config/settings.yaml`:

```bash
export ANTHROPIC_API_KEY="your-key"
export PUBMED_EMAIL="your@email.com"
```

## Project Structure

```
research-automation/
├── src/research_automation/
│   ├── core/           # Config, database, storage, Claude API
│   ├── literature/     # Paper search, download, summarization
│   ├── collection/     # Video collection, quality, datasets
│   └── cli.py          # Typer CLI application
├── config/
│   ├── settings.yaml
│   └── search_queries.yaml
└── tests/
```

## License

MIT
