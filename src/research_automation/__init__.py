"""Research Automation: Video-based Health Monitoring.

A comprehensive toolkit for automating research workflows in video-based
health monitoring, including literature search, video collection, and
quality assessment.
"""

__version__ = "0.1.0"

from research_automation.core import (
    Base,
    ClaudeClient,
    Settings,
    StorageManager,
    get_claude_client,
    get_session,
    get_settings,
    init_db,
)

__all__ = [
    "Base",
    "ClaudeClient",
    "Settings",
    "StorageManager",
    "__version__",
    "get_claude_client",
    "get_session",
    "get_settings",
    "init_db",
]
