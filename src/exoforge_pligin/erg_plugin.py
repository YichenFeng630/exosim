"""ERG Plugin - Consolidated module for backward compatibility.

向后兼容的主模块 - 从独立模块汇总导入。

Module structure (now separated):
- erg_plugin_core.py: Core signal processing (ErgFilter, ErgConfig)
- erg_recorder.py: Data collection (ErgRecorder, MultiMuscleRecorder)
- erg_visualization.py: Plotting (plot_erg_signal, plot_multi_erg, etc.)
- viewer_with_erg.py: Interactive viewer (ErgViewerController, run_interactive_viewer)

This file re-exports for convenience:
from erg_plugin import ErgFilter, ErgRecorder, plot_erg_signal, run_interactive_viewer
"""
from __future__ import annotations

from .erg_plugin_core import ErgConfig, ErgFilter, build_erg_processor
from .erg_recorder import ErgRecorder, MultiMuscleRecorder
from .erg_visualization import (
    plot_comparison,
    plot_erg_signal,
    plot_multi_erg,
    quick_plot,
)
from .viewer_with_erg import ErgViewerController, run_interactive_viewer

__all__ = [
    # Core
    "ErgConfig",
    "ErgFilter",
    "build_erg_processor",
    # Recording
    "ErgRecorder",
    "MultiMuscleRecorder",
    # Visualization
    "plot_erg_signal",
    "plot_multi_erg",
    "plot_comparison",
    "quick_plot",
    # Viewer
    "ErgViewerController",
    "run_interactive_viewer",
]

__version__ = "2.0.0"

