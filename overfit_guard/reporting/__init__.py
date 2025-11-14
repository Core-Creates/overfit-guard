"""
Professional reporting module for research, marketing, and debugging.

This module provides:
- Multi-style comparison reporting (research, marketing, debug)
- Research paper-ready reports (LaTeX, PDF)
- Marketing materials (executive summaries, ROI analysis)
- Debug reports (detailed intervention logs)
- Model cards and reproducibility information
- Multi-format exports (JSON, CSV, HTML, Markdown)

Usage:
    from overfit_guard.reporting import (
        compute_overfit_guard_summary,
        print_overfit_guard_summary,
        ResearchReporter,
        MarketingReporter,
        DebugReporter,
        ModelCardGenerator,
        ReportExporter
    )
"""

from .comparison import (
    compute_overfit_guard_summary,
    print_overfit_guard_summary,
    export_summary_to_dict
)
from .research_reporter import ResearchReporter
from .marketing_reporter import MarketingReporter
from .debug_reporter import DebugReporter
from .model_card import ModelCardGenerator
from .export import ReportExporter

__all__ = [
    # Core comparison functions
    'compute_overfit_guard_summary',
    'print_overfit_guard_summary',
    'export_summary_to_dict',

    # Specialized reporters
    'ResearchReporter',
    'MarketingReporter',
    'DebugReporter',
    'ModelCardGenerator',
    'ReportExporter',
]

__version__ = '1.0.0'
