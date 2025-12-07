"""ERG visualization module (independent plotting).

═══════════════════════════════════════════════════════════════════
独立绘图模块 / Visualization module
═══════════════════════════════════════════════════════════════════

设计原则 / Design principles:
1. 独立性 (Independence):
   - 不依赖 ERG 处理核心或数据录制模块
   - 只需要 ErgRecorder 实例，即可进行任何绘图
   - Can plot any ErgRecorder data independently

2. 灵活性 (Flexibility):
   - 支持单肌肉、多肌肉、比较等多种绘图模式
   - Multiple plot types: single, multi, comparison, quick
   - Easy to combine plots for custom analysis

3. 可视化风格 (Visualization style):
   - 参考 sEMG-Sim 风格（简洁、专业）
   - Based on sEMG-Sim style (clean, professional)
   - 合理的颜色搭配和图表布局
   - Reasonable colors and layout

4. 导出选项 (Export options):
   - 所有图表都支持保存为 PNG（150 DPI）
   - All plots support saving as PNG (150 DPI)
   - 支持交互式查看（通过 plt.show()）
   - Interactive viewing support

提供的函数 / Provided functions:

1. plot_erg_signal():
   - 单肌肉详细分析（3 面板 + 可选缩放）
   - ERG envelope, activation, force
   - 常用于 offline analysis

2. plot_multi_erg():
   - N 行子图，每行一个肌肉
   - 快速对比多肌肉 ERG 信号
   - 常用于多肌肉实验分析

3. plot_comparison():
   - 并排对比两个 ErgRecorder
   - 显示 ERG 曲线 + 统计柱状图
   - 常用于评估两次记录的差异

4. quick_plot():
   - 简单快速的内联绘图
   - 支持双/三坐标轴（ERG, activation, force）
   - 常用于 Jupyter notebook 或实时反馈

工作流 / Typical workflow:

┌─────────────────────────────────────────────────┐
│ MuJoCo Viewer + 手动控制                        │
│ ↓                                               │
│ MultiMuscleRecorder                             │
│ ↓                                               │
│ rec.save_all()  ← 导出 NPZ 文件                 │
│ ↓                                               │
│ 重新加载或使用内存中的数据                      │
│ ↓                                               │
│ plot_multi_erg()  ← 快速查看所有肌肉            │
│ plot_erg_signal() ← 详细分析单个肌肉            │
│ plot_comparison() ← 比较两个肌肉或两次记录      │
└─────────────────────────────────────────────────┘

注意事项 / Important notes:

- 所有函数都使用 plt.show() 显示图表
  All functions use plt.show() to display plots
  在 Jupyter notebook 中会自动显示
  In Jupyter notebooks, displays automatically
  在脚本中可能阻塞 / May block in scripts (add plt.show() if needed)

- 如果指定 save_path，会在显示前保存
  If save_path is specified, saves before showing
  
- 支持中文标签和标题
  Supports Chinese labels and titles
  （需要系统安装中文字体）
  (requires Chinese fonts on system)

- 颜色约定 / Color conventions:
  ERG signal      → steelblue (钢蓝色)
  Activation      → orange (橙色)
  Force           → green (绿色)
  Fill area       → lighter version of main color
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 支持两种导入模式（包 vs 直接运行）
try:
    from .erg_recorder import ErgRecorder, MultiMuscleRecorder
except ImportError:
    from erg_recorder import ErgRecorder, MultiMuscleRecorder


def plot_erg_signal(
    recorder: ErgRecorder,
    window: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot ERG signal with optional zoom window / 绘制 ERG 信号及可选的缩放窗口.
    
    ───────────────────────────────────────────────────────────────
    单肌肉详细分析 / Single-muscle detailed analysis
    ───────────────────────────────────────────────────────────────
    
    这是最详细的分析图表。显示 ERG、激活和力的完整时间序列，
    可选地显示某个时间段的放大视图。
    
    This is the most detailed analysis plot showing ERG, activation,
    and force timeseries with optional zoomed view.
    
    参数 / Parameters:
        recorder (ErgRecorder): 包含记录数据的记录器实例
                               Recorder instance with recorded data
        
        window (Optional[Tuple[float, float]]): 可选的缩放窗口
                                               Optional zoom window (t_start, t_end)
            如果提供，图表会显示 2 列：
            - 第 1 列：完整时间序列
            - 第 2 列：zoom 到指定窗口的放大视图
            
            例 / e.g. window=(10.0, 15.0) 会放大 [10, 15] 秒区间
        
        save_path (Optional[str]): 保存路径
                                   如果提供，图表会保存为 PNG
                                   Path to save the plot (PNG format)
                                   例 / e.g. "analysis/masseter_L_detail.png"
        
        figsize (Tuple[int, int]): 图表尺寸（英寸）
                                   Figure size in inches
                                   默认 (12, 8)
                                   zoom 窗口会使用 (14, 8)
    
    图表结构 / Plot structure:
    
    不带 zoom 的版本 (window=None) / Without zoom:
    ┌──────────────────────┐
    │ Row 0: ERG Signal    │
    │        (steelblue)   │
    ├──────────────────────┤
    │ Row 1: Activation    │
    │        (orange)      │
    ├──────────────────────┤
    │ Row 2: Force         │
    │        (green)       │
    └──────────────────────┘
    
    带 zoom 的版本 (window=(t1, t2)) / With zoom:
    ┌─────────────────────┬─────────────────┐
    │ Full ERG (t=0..end) │ Zoom ERG [t1,t2]│
    ├─────────────────────┼─────────────────┤
    │ Full Act (t=0..end) │ Zoom Act [t1,t2]│
    ├─────────────────────┼─────────────────┤
    │ Full Force          │ Zoom Force      │
    └─────────────────────┴─────────────────┘
    
    每行显示的内容 / Content of each row:
    
    Row 0 - ERG Signal (ERG 包络信号):
        - 来自 ErgFilter.step() 的输出
        - 代表平滑的肌肉活动强度
        - 范围 [0, 1]，其中 0 = 完全放松，1 = 最大收缩
        - 这是最重要的信号，用于意图识别
    
    Row 1 - Activation (用户激活信号):
        - 来自 MuJoCo Viewer 手动控制
        - 用户在滑块上设置的值
        - 范围 [0, 1]，代表用户想要的力量
        - 用于验证 ERG 是否跟踪了用户的意图
    
    Row 2 - Force (执行器力):
        - 来自 MuJoCo 物理模拟
        - 单位：牛顿 (N)
        - 可以是正值（收缩）或负值（伸展）
        - 这里取绝对值显示，代表强度而不区分方向
        - 用于验证 ERG 是否反映了实际的力输出
    
    常见用法 / Common usage:
    
    1. 基本查看 / Basic view:
        >>> rec = ErgRecorder("masseter_L")
        >>> # ... record_step ...
        >>> plot_erg_signal(rec)
    
    2. 详细分析某个时间段 / Zoom into specific timespan:
        >>> plot_erg_signal(rec, window=(5.0, 10.0))
        # 显示完整信号 + [5, 10] 秒的放大图
    
    3. 保存为文件 / Save to file:
        >>> plot_erg_signal(rec, save_path="results/masseter_detail.png")
        # 保存为 PNG 后自动显示
    
    4. 结合多个操作 / Combined operations:
        >>> plot_erg_signal(
        ...     rec,
        ...     window=(15.0, 25.0),
        ...     save_path="analysis/day1_trial1.png",
        ...     figsize=(14, 10)
        ... )
    
    解读图表 / How to interpret the plot:
    
    好的 ERG 信号特征 / Good ERG signal characteristics:
    - ERG 跟随激活变化（activation 上升时 ERG 也上升）
    - ERG 的变化幅度与力的大小相关
    - ERG 信号平滑，没有过多噪声
    - 静止时 ERG 接近 0
    - ERG 在用户做动作时快速上升，释放时缓慢下降（LP 滤波器的作用）
    
    可能的问题 / Potential issues:
    - ERG 不随激活变化？可能激活没被记录正确
    - ERG 有很多尖刺？可能噪声参数 (noise_std) 太大
    - ERG 响应很慢？可能低通滤波系数 (a_lp) 太高
    - ERG 有漂移？可能高通滤波系数 (a_hp) 不合适
    
    参考 sEMG-Sim 绘图风格。
    References sEMG-Sim plotting style.
    
    侧效应 / Side effects:
    - 创建新的 matplotlib 图表
    - 如果指定 save_path，保存为 PNG（150 DPI）
    - 打印保存成功的消息
    - 调用 plt.show() 显示图表
    
    退出图表后 / After closing the plot:
    - 函数返回（无返回值）
    - 图表文件已保存（如果指定了 save_path）
    """
    # 提取数据 / Extract data from recorder
    data = recorder.to_numpy()
    t = data["time_array"]
    erg = data["erg_signal"]
    act = data["activation"]
    force = np.abs(data["force"])  # 取绝对值 / take absolute value

    # 确定列数：有 zoom 窗口则 2 列，否则 1 列
    # Determine number of columns
    cols = 2 if window else 1
    
    # 创建子图布局：3 行 × cols 列
    # Create subplot grid: 3 rows × cols columns
    # 如果 cols=1，则 axes 是 [ax0, ax1, ax2]
    # 如果 cols=2，则 axes 是 [[ax00, ax01], [ax10, ax11], [ax20, ax21]]
    fig, axes = plt.subplots(3, cols, figsize=(14 if cols == 2 else 8, 10))
    if cols == 1:
        # 转换为二维数组便于后续统一处理
        axes = [[axes[i]] for i in range(3)]

    fig.suptitle(f"ERG Analysis: {recorder.muscle_name}", fontsize=14, fontweight="bold")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 行 0: ERG 信号 / Row 0: ERG Signal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[0][0]
    ax.plot(t, erg, label="ERG (envelope)", lw=1.5, color="steelblue")
    ax.set_ylabel("ERG Envelope")
    ax.set_title(f"{recorder.muscle_name} - ERG Signal")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 如果有 zoom 窗口，绘制放大视图 / Draw zoomed view if window specified
    if window:
        ax_zoom = axes[0][1]
        ws, we = window
        mask = (t >= ws) & (t <= we)
        ax_zoom.plot(t[mask], erg[mask], lw=1.5, color="steelblue")
        ax_zoom.set_title(f"Zoom [{ws:.3f}, {we:.3f}] s")
        ax_zoom.grid(alpha=0.3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 行 1: 激活信号 / Row 1: Activation Signal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[1][0]
    ax.plot(t, act, label="Activation", lw=1.5, color="orange")
    ax.set_ylabel("Activation")
    ax.grid(alpha=0.3)
    ax.legend()
    if window:
        ax_zoom = axes[1][1]
        ax_zoom.plot(t[mask], act[mask], lw=1.5, color="orange")
        ax_zoom.grid(alpha=0.3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 行 2: 力信号 / Row 2: Force Signal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[2][0]
    ax.plot(t, force, label="Force (abs)", lw=1.5, color="green")
    ax.set_ylabel("Force")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    if window:
        ax_zoom = axes[2][1]
        ax_zoom.plot(t[mask], force[mask], lw=1.5, color="green")
        ax_zoom.set_xlabel("Time (s)")
        ax_zoom.grid(alpha=0.3)

    # 调整布局 / Adjust spacing
    plt.tight_layout()
    
    # 保存（如果指定了路径）/ Save if path specified
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved plot: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_multi_erg(
    recorders: dict[str, ErgRecorder],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Plot multiple muscle ERG signals in subplots / 在子图中绘制多个肌肉的 ERG 信号.
    
    ───────────────────────────────────────────────────────────────
    多肌肉并排显示 / Multi-muscle side-by-side view
    ───────────────────────────────────────────────────────────────
    
    这个函数为每个肌肉创建一行子图，显示其 ERG 信号和填充区域。
    非常适合快速比较多个肌肉的活动情况。
    
    This function creates one subplot row per muscle showing ERG signal
    with filled area underneath. Perfect for quick multi-muscle comparison.
    
    参数 / Parameters:
        recorders (dict[str, ErgRecorder]): 
            肌肉名称 → ErgRecorder 映射 / muscle name → ErgRecorder mapping
            例 / e.g. {
                "masseter_L": recorder_L,
                "masseter_R": recorder_R,
                "temporalis": recorder_T,
            }
        
        save_path (Optional[str]): 保存路径 / path to save PNG
                                   例 / e.g. "results/multi_erg.png"
        
        figsize (Tuple[int, int]): 图表尺寸 / figure size in inches
                                   默认 (14, 10)，可根据肌肉数量调整
                                   默认对 3-4 个肌肉较好
    
    图表结构 / Plot structure:
    
    示例（3 个肌肉）/ Example (3 muscles):
    ┌─────────────────────────────────────────────────────┐
    │ Row 0: masseter_L                                   │
    │        [ERG signal with filled area]               │
    ├─────────────────────────────────────────────────────┤
    │ Row 1: masseter_R                                   │
    │        [ERG signal with filled area]               │
    ├─────────────────────────────────────────────────────┤
    │ Row 2: temporalis                                   │
    │        [ERG signal with filled area]               │
    └─────────────────────────────────────────────────────┘
    
    每行特点 / Features of each row:
    - 线条 (line): ERG 信号的实时值，宽度 1.5 pts
      steelblue 颜色，易于区分
    - 填充区域 (fill): 从 x 轴到 ERG 线的半透明填充
      alpha=0.2，视觉上更易理解面积
    - 图例 (legend): 肌肉名称，位于右上角
    - 网格 (grid): 淡灰色背景网格，便于读取数值
    - 纵轴标签: "ERG Envelope"
    - 底行的横轴标签: "Time (s)"
    
    常见用法 / Common usage:
    
    1. 从 MultiMuscleRecorder 导出 / From MultiMuscleRecorder:
        >>> recorder = MultiMuscleRecorder(["m_L", "m_R", "t_L"])
        >>> # ... record_step ...
        >>> plot_multi_erg(recorder.recorders)
    
    2. 手动构建字典 / Manually build dict:
        >>> recorders = {
        ...     "masseter_L": rec_L,
        ...     "masseter_R": rec_R,
        ... }
        >>> plot_multi_erg(recorders)
    
    3. 保存结果 / Save result:
        >>> plot_multi_erg(recorder.recorders, save_path="day1_trial1.png")
    
    4. 自定义图表大小 / Custom figure size:
        >>> plot_multi_erg(recorder.recorders, figsize=(16, 12))
        # 对 5 个肌肉较好 / good for 5 muscles
    
    解读图表 / How to interpret:
    
    肌肉活动模式 / Muscle activation patterns:
    - 峰值高 → 该肌肉活动强度大 / High peaks = high activity
    - 多个峰值 → 多次肌肉收缩 / Multiple peaks = repeated contractions
    - 不同高度 → 肌肉强度不匹配 / Different heights = strength imbalance
    - 异步 → 肌肉激活时间不同步 / Asynchronous = timing difference
    
    同步性检查 / Synchronization check:
    - 如果是配对肌肉（如左右两侧），应该看起来相似
    - Paired muscles (L/R) should look similar
    - 如果明显不同，可能反映生物力学不对称
    - Significant difference may indicate biomechanical asymmetry
    
    建议 / Recommendations:
    - 肌肉数 ≤ 3: figsize=(14, 10) 足够 / sufficient
    - 肌肉数 = 4-5: figsize=(14, 12) 或 (16, 10)
    - 肌肉数 > 5: figsize=(14, 14) 或考虑分组显示
    
    侧效应 / Side effects:
    - 如果 recorders 为空，打印 "No recorders to plot" 并返回
    - Creates new matplotlib figure
    - Saves PNG if save_path specified
    - Calls plt.show()
    """
    # 获取肌肉数量 / Get number of muscles
    n_muscles = len(recorders)
    if n_muscles == 0:
        print("No recorders to plot")
        return

    # 创建子图：n_muscles 行，1 列
    # Create subplots: n_muscles rows, 1 column
    fig, axes = plt.subplots(n_muscles, 1, figsize=figsize)
    
    # 如果只有 1 个肌肉，axes 是 AxesSubplot，需要转换为列表
    # If only 1 muscle, axes is a single AxesSubplot, convert to list
    if n_muscles == 1:
        axes = [axes]

    fig.suptitle("Multi-Muscle ERG Analysis", fontsize=14, fontweight="bold")

    # 遍历每个肌肉，绘制其 ERG 信号 / Iterate through each muscle
    for idx, (muscle_name, recorder) in enumerate(recorders.items()):
        data = recorder.to_numpy()
        t = data["time_array"]
        erg = data["erg_signal"]

        ax = axes[idx]
        # 绘制 ERG 线条 / Plot ERG line
        ax.plot(t, erg, lw=1.5, color="steelblue", label=muscle_name)
        # 填充区域（从 0 到 ERG 信号）/ Fill area under curve
        ax.fill_between(t, 0, erg, alpha=0.2, color="steelblue")
        
        ax.set_ylabel("ERG Envelope")
        ax.set_title(f"{muscle_name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    # 为最后一行添加横轴标签 / Add x-label to last row
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved plot: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    recorder1: ErgRecorder,
    recorder2: ErgRecorder,
    label1: str = "Signal 1",
    label2: str = "Signal 2",
    save_path: Optional[str] = None,
) -> None:
    """Compare two ERG recordings side by side / 并排比较两个 ERG 记录.
    
    ───────────────────────────────────────────────────────────────
    双信号对比分析 / Dual-signal comparison analysis
    ───────────────────────────────────────────────────────────────
    
    这个函数用于比较两个 ErgRecorder 实例。
    常用场景：
    1. 同一肌肉的两次记录（评估重复性）
    2. 左右对称肌肉（评估平衡性）
    3. 不同参数设置下的结果（评估敏感性）
    
    This function compares two ErgRecorder instances side-by-side.
    Common scenarios:
    1. Two recordings of same muscle (evaluate repeatability)
    2. Left vs Right muscles (evaluate symmetry)
    3. Different parameter settings (evaluate sensitivity)
    
    参数 / Parameters:
        recorder1 (ErgRecorder): 第一个记录器 / first recorder
        recorder2 (ErgRecorder): 第二个记录器 / second recorder
        
        label1 (str): 第一个的标签 / label for first
                     默认 "Signal 1"
                     用于图表标题和图例
        
        label2 (str): 第二个的标签 / label for second
                     默认 "Signal 2"
                     
                     例 / e.g.
                     label1="Left Masseter", label2="Right Masseter"
        
        save_path (Optional[str]): 保存路径 / path to save PNG
    
    图表布局 / Plot layout (2 列):
    
    ┌──────────────────────────┬──────────────────────────┐
    │ 左: ERG 信号对比          │ 右: 统计柱状图          │
    │                         │                          │
    │ - Signal 1 (alpha=0.7) │ - 4 个指标的柱状图        │
    │ - Signal 2 (alpha=0.7) │   mean_erg               │
    │                         │   max_erg                │
    │ 叠加显示两个信号         │   mean_act               │
    │ (Overlaid view)         │   max_force              │
    │                         │                          │
    │ 时间軸: 从 0 到最后      │ 两种颜色的柱子           │
    │ (Time axis)             │ (Two bar colors)         │
    └──────────────────────────┴──────────────────────────┘
    
    常见用法 / Common usage:
    
    1. 对比左右肌肉 / Compare left vs right:
        >>> rec_L = ErgRecorder("masseter_L")
        >>> rec_R = ErgRecorder("masseter_R")
        >>> # ... record ...
        >>> plot_comparison(rec_L, rec_R, label1="Left", label2="Right")
    
    2. 对比两次试验 / Compare two trials:
        >>> rec_trial1 = ErgRecorder("trial_1")
        >>> rec_trial2 = ErgRecorder("trial_2")
        >>> plot_comparison(
        ...     rec_trial1, rec_trial2,
        ...     label1="Trial 1", label2="Trial 2",
        ...     save_path="comparison.png"
        ... )
    
    3. 对比不同参数 / Compare different parameters:
        >>> rec_k3 = ErgRecorder("k=3.0")
        >>> rec_k5 = ErgRecorder("k=5.0")
        >>> plot_comparison(rec_k3, rec_k5, label1="k=3.0", label2="k=5.0")
    
    左侧图 (ERG 信号对比) / Left plot (Signal comparison):
    - 两条线叠加显示，便于直观比较时间对齐
    - 两个信号用不同颜色区分
    - alpha=0.7 使得叠加部分可以看到两条线
    - 点越接近表示两个信号越相似
    - 点差异大表示存在差异
    
    右侧图 (统计对比) / Right plot (Statistics comparison):
    - 柱状图并排显示 4 个关键指标
    - 每个指标显示两条柱子（Signal 1 和 Signal 2）
    - 高度差异反映信号特性的差异
    
    统计指标说明 / Metrics explained:
    1. mean_erg: 平均 ERG 值（肌肉平均强度）
       Higher = stronger baseline activity
    
    2. max_erg: 最大 ERG 值（峰值强度）
       Higher = more forceful contractions
    
    3. mean_act: 平均激活（用户平均输入）
       Reflects how hard user was trying
    
    4. max_force: 最大力值（最强收缩）
       Higher = more powerful muscle
    
    何时出现不匹配？/ When to expect differences?
    
    左右肌肉对称 / Left-right symmetry:
    - 类似的高度 → 肌肉对称 / Similar heights → good symmetry
    - 左高 → 左侧更强 / Left higher → left dominant
    - 右高 → 右侧更强 / Right higher → right dominant
    - 差异 > 20% → 可能需要调整训练 / >20% diff → consider training
    
    两次试验重复性 / Two-trial repeatability:
    - 很相似 → 好的重复性 / Similar → good repeatability
    - 明显不同 → 可能存在外界干扰 / Different → possible noise/interference
    - 趋势相同但幅度不同 → 可能是激活程度不同 / Same trend but scaled → different effort level
    
    侧效应 / Side effects:
    - Creates new matplotlib figure with 2 subplots
    - Saves PNG if save_path specified
    - Calls plt.show()
    """
    # 提取两个记录的数据 / Extract data
    data1 = recorder1.to_numpy()
    data2 = recorder2.to_numpy()

    # 创建 1 行 2 列的子图 / Create 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"ERG Comparison: {label1} vs {label2}", fontsize=14, fontweight="bold")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 左侧图: ERG 信号对比 / Left plot: ERG signals
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[0]
    ax.plot(data1["time_array"], data1["erg_signal"], label=label1, lw=2, alpha=0.7)
    ax.plot(data2["time_array"], data2["erg_signal"], label=label2, lw=2, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERG Envelope")
    ax.set_title("ERG Signals")
    ax.grid(alpha=0.3)
    ax.legend()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 右侧图: 统计对比 / Right plot: Statistics comparison
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[1]
    
    # 获取两个记录的统计信息 / Get statistics
    stats1 = recorder1.summary_stats()
    stats2 = recorder2.summary_stats()
    
    # 选择要显示的指标 / Select metrics to display
    metrics = ["mean_erg", "max_erg", "mean_act", "max_force"]
    x = np.arange(len(metrics))  # 指标在 x 轴上的位置 / positions on x-axis
    width = 0.35  # 柱子宽度 / bar width
    
    # 提取每个指标的值 / Extract values for each metric
    values1 = [stats1.get(m, 0) for m in metrics]
    values2 = [stats2.get(m, 0) for m in metrics]
    
    # 绘制柱状图 / Draw bars
    ax.bar(x - width/2, values1, width, label=label1, alpha=0.7)
    ax.bar(x + width/2, values2, width, label=label2, alpha=0.7)
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Statistics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved plot: {save_path}")
        plt.close()
    else:
        plt.show()


def quick_plot(
    recorder: ErgRecorder,
    show_activation: bool = True,
    show_force: bool = True,
) -> None:
    """Quick inline plot for notebook or interactive use / 快速内联图表.
    
    ───────────────────────────────────────────────────────────────
    简洁、轻量级的绘图 / Compact, lightweight plotting
    ───────────────────────────────────────────────────────────────
    
    这个函数生成一个简单的图表，非常适合在 Jupyter notebook
    或实时交互环境中快速查看数据。
    
    This function creates a simple plot perfect for Jupyter notebooks
    or real-time interactive viewing.
    
    特点 / Features:
    - 单行图表（高度 3 英寸）
    - ERG 为主要信号（steelblue），宽线
    - 激活和力为辅助信号（右轴），细线
    - 支持双/三坐标轴（ERG, activation, force）
    - 快速生成，文件体积小
    
    参数 / Parameters:
        recorder (ErgRecorder): 要绘制的记录器 / recorder to plot
        
        show_activation (bool): 是否显示激活信号 / show activation signal
                               默认 True
                               如果 True，激活信号使用右侧第 1 个 Y 轴
        
        show_force (bool): 是否显示力信号 / show force signal
                          默认 True
                          如果 True，力信号使用右侧第 2 个 Y 轴
    
    图表配置 / Plot configuration:
    
    仅 ERG (show_activation=False, show_force=False):
    ┌───────────────────────┐
    │ ERG (steelblue)       │
    │                       │
    │ Y 轴标签: "ERG"       │
    └───────────────────────┘
    
    ERG + Activation (show_force=False):
    ┌──────────────────────────────┐
    │ ERG (steelblue, 厚)          │
    │ Activation (orange, 薄)      │
    │                              │
    │ 左 Y: "ERG" 右 Y: "Activation"│
    └──────────────────────────────┘
    
    全部信号 (show_activation=True, show_force=True):
    ┌────────────────────────────────────────────────────┐
    │ ERG (steelblue, 厚)                                │
    │ Activation (orange, 薄)                            │
    │ Force (green, 薄)                                  │
    │                                                    │
    │ 左 Y: "ERG"     中右 Y: "Activation"  右 Y: "Force"│
    └────────────────────────────────────────────────────┘
    
    常见用法 / Common usage:
    
    1. Jupyter notebook 中快速查看 / Quick view in Jupyter:
        >>> rec = ErgRecorder("masseter")
        >>> # ... record_step ...
        >>> quick_plot(rec)
        # 自动显示图表
    
    2. 只看 ERG，不要其他信号 / Only show ERG:
        >>> quick_plot(rec, show_activation=False, show_force=False)
    
    3. 看 ERG 和激活，不要力 / Show ERG and activation only:
        >>> quick_plot(rec, show_activation=True, show_force=False)
    
    4. 同时检查三个信号 / Check all three signals:
        >>> quick_plot(rec)  # 默认参数 / default parameters
    
    何时使用 quick_plot？/ When to use quick_plot?
    
    ✓ 适合用 quick_plot:
    - 快速探索数据 / Quick data exploration
    - Jupyter notebook 中的实时反馈 / Real-time feedback in Jupyter
    - 简单的数据验证 / Quick data validation
    - 不需要详细分析时 / When detailed analysis not needed
    
    ✗ 不适合用 quick_plot:
    - 需要放大视图 / Zoom view needed → use plot_erg_signal
    - 多肌肉对比 / Multi-muscle comparison → use plot_multi_erg
    - 两个信号比较 / Signal comparison → use plot_comparison
    - 需要高质量输出 / High-quality output → use other functions
    
    Y 轴配置说明 / Y-axis configuration:
    
    这个函数使用了 matplotlib 的"多轴"特性：
    - 主轴（左): ERG 信号
    - 右轴 1: Activation 信号（如果 show_activation=True）
    - 右轴 2: Force 信号（如果 show_force=True）
    
    好处 / Benefits:
    ✓ 在一个图表中显示多个量级不同的信号
    ✓ 易于视觉比较
    ✓ 轴标签清晰表示每条线的单位
    
    缺点 / Drawbacks:
    ✗ 右轴的刻度与左轴独立，可能误导
    ✗ 不如分组显示清晰
    
    何时要小心 / When to be careful:
    - ERG 范围 [0, 1]，Activation 范围 [0, 1]，Force 可能很大
    - 要确保轴刻度合理，不要被视觉欺骗
    
    侧效应 / Side effects:
    - Creates new matplotlib figure
    - Calls plt.show()
    """
    # 提取数据 / Extract data
    data = recorder.to_numpy()
    t = data["time_array"]
    erg = data["erg_signal"]

    # 创建图表（单行，高度 3 英寸）
    # Create single-row figure
    plt.figure(figsize=(12, 3))
    
    # 主轴：绘制 ERG 信号 / Main axis: plot ERG
    plt.plot(t, erg, lw=1.5, color="steelblue", label="ERG")
    
    # 如果需要显示激活信号，创建右轴 / If showing activation, create right axis
    if show_activation:
        ax2 = plt.gca().twinx()
        ax2.plot(t, data["activation"], lw=1, color="orange", alpha=0.5, label="Activation")
        ax2.set_ylabel("Activation")
    
    # 如果需要显示力信号，创建第二个右轴 / If showing force, create second right axis
    if show_force:
        ax3 = plt.gca().twinx()
        # 将第二个右轴向外移动，避免与第一个右轴重叠
        # Offset the second right axis to avoid overlap
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(t, np.abs(data["force"]), lw=1, color="green", alpha=0.5, label="Force")
        ax3.set_ylabel("Force")

    # 设置左轴标签 / Set left axis labels
    plt.xlabel("Time (s)")
    plt.ylabel("ERG Envelope")
    plt.title(recorder.muscle_name)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()