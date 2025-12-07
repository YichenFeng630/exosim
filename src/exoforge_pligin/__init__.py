"""ExoForge ERG Plugin Package.

独立分离的模块组织：
- erg_plugin_core: 核心信号处理 (ErgFilter, ErgConfig)
- erg_recorder: 数据记录 (ErgRecorder, MultiMuscleRecorder)
- erg_visualization: 绘图和可视化 (plot_erg_signal, etc.)
- viewer_with_erg: 交互式 viewer (ErgViewerController, run_interactive_viewer)
- erg_plugin: 汇总模块，提供向后兼容接口

Usage:
    from src.exoforge_pligin import ErgFilter, ErgRecorder, plot_erg_signal
"""

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
