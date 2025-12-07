"""ERG data recording module (independent data collection).

独立数据记录模块 - 与 ERG 信号处理、可视化分离。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ErgRecorder:
    """Record ERG signals over simulation time / 记录 ERG 信号.
    
    ═══════════════════════════════════════════════════════════════════
    数据记录管理器 / Data recording manager
    ═══════════════════════════════════════════════════════════════════
    
    设计原则 / Design principles:
    1. 独立性 (Independence): 不依赖 ERG 处理逻辑或绘图模块
       只负责数据采集、存储、导出
    
    2. 简洁性 (Simplicity): 使用列表而非循环缓冲区
       Python 列表追加很快，内存也可控
       实际应用中数据量有限（最多几分钟 × 500Hz ≈ 100K 样本）
    
    3. 灵活性 (Flexibility): 支持多种导出格式
       - NPZ: 压缩二进制，适合科学计算（保留全精度）
       - CSV: 纯文本，易于 Excel/其他工具查看
       - 统计: 即时计算汇总指标（min/max/mean）
    
    数据流 / Data flow:
    ┌─────────────────┐
    │ MuJoCo 仿真     │
    │ 每 dt 时间步    │
    └────────┬────────┘
             │ activation, force
             ↓
    ┌─────────────────┐
    │ ErgFilter       │ → erg_signal
    │ .step()         │
    └────────┬────────┘
             │ (erg, act, force, t)
             ↓
    ┌─────────────────┐
    │ ErgRecorder     │
    │ .record_step()  │
    └────────┬────────┘
             │ 追加到列表
             ↓
    ┌─────────────────┐
    │ 数据数组        │
    │ time_array      │ [0.000, 0.002, 0.004, ...]
    │ erg_signal      │ [0.001, 0.005, 0.012, ...]
    │ activation      │ [0.100, 0.100, 0.102, ...]
    │ force           │ [1.234, 1.240, 1.260, ...]
    └────────┬────────┘
             │
             ├─→ save_npz() → .npz 文件（压缩）
             ├─→ save_csv() → .csv 文件（文本）
             └─→ summary_stats() → 统计指标
    
    属性 / Attributes:
        muscle_name (str): 肌肉名称，用于文件命名
        time_array (list[float]): 仿真时间戳，单位秒
        erg_signal (list[float]): ERG 包络值，范围 [0, 1]
        activation (list[float]): 用户激活（控制输入），范围 [0, 1]
        force (list[float]): 执行器输出力，单位 N
        dt (float): 仿真时间步长，默认 0.002s (500Hz)
    
    样本用法 / Usage example:
        >>> rec = ErgRecorder("masseter_L")
        >>> rec.record_step(0.000, erg=0.01, act=0.5, force=1.2)
        >>> rec.record_step(0.002, erg=0.02, act=0.5, force=1.3)
        >>> rec.record_step(0.004, erg=0.03, act=0.5, force=1.4)
        >>> rec.summary_stats()
        {'n_samples': 3, 'duration_s': 0.004, 'mean_erg': 0.02, ...}
        >>> rec.save_npz()  # → masseter_L_erg.npz
        >>> rec.save_csv()  # → masseter_L_erg.csv
    """
    muscle_name: str
    time_array: list[float] = field(default_factory=list)
    erg_signal: list[float] = field(default_factory=list)
    activation: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=list)
    dt: float = 0.002  # 默认时间步长 / default timestep (500 Hz)

    def record_step(self, t: float, erg: float, act: float, force: float) -> None:
        """Record one simulation step / 记录一个仿真步.
        
        ───────────────────────────────────────────────────────────────
        核心数据收集方法 / Core data collection method
        ───────────────────────────────────────────────────────────────
        
        这是最常被调用的方法（每个时间步调用一次）。
        This is called at every timestep during simulation.
        
        参数 / Parameters:
            t (float):     当前模拟时间，单位秒 / current simulation time
                          例 / example: 0.000, 0.002, 0.004, ...
                          通常是步数 × dt
            
            erg (float):   ERG 包络输出，范围 [0, 1]
                          来自 ErgFilter.step() 的返回值
                          from ErgFilter.step() output
                          表示当前肌肉活动强度 / represents muscle activation strength
                          
            act (float):   用户激活信号，范围 [0, 1]
                          从 MuJoCo Viewer 手动控制获得
                          from manual control slider in MuJoCo Viewer
                          用户想要施加多大的力 / how much force user wants to apply
                          
            force (float): 执行器实际产生的力，单位牛顿 (N)
                          来自 MuJoCo 物理模拟反馈
                          from MuJoCo physics simulation
                          可以是正值（收缩）或负值（伸展）
                          can be positive (contraction) or negative (extension)
        
        性能考虑 / Performance notes:
        - 列表 append() 是 O(1) 摊销时间
        - List append() is O(1) amortized
        - 不会发生显著的 GC 开销（Python 内部管理得很好）
        - No significant GC overhead (Python manages efficiently)
        
        存储效率 / Storage efficiency:
        - 每个样本 4 × 8 字节 = 32 字节（Python 对象开销另外计）
        - Per sample: 4 × 8 bytes = 32 bytes (Python overhead separate)
        - 60 秒 × 500Hz = 30,000 样本 ≈ 1 MB（不计 Python 开销）
        - 60s × 500Hz = 30,000 samples ≈ 1 MB (excluding Python overhead)
        
        为什么不用循环缓冲区？ / Why not use circular buffer?
        - 我们不需要实时约束（不是嵌入式系统）
        - 无需固定时间延迟保证
        - 完整数据对事后分析很重要
        - We need full historical data for post-analysis
        
        调用频率 / Calling frequency:
        - 推荐: 每个 MuJoCo 仿真步调用一次
        - Recommended: once per MuJoCo simulation step
        - 约 500 Hz (dt=0.002s)
        - ~500 Hz (dt=0.002s)
        """
        # 四个列表同步追加，保证索引对应
        # Append to all four lists in sync - indices correspond
        self.time_array.append(t)
        self.erg_signal.append(erg)
        self.activation.append(act)
        self.force.append(force)

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert to numpy arrays / 转换为 numpy 数组.
        
        ───────────────────────────────────────────────────────────────
        从 Python 列表转换为 NumPy 数组 / Convert Python lists to NumPy
        ───────────────────────────────────────────────────────────────
        
        为什么需要这个？/ Why do we need this?
        - Python 列表和 NumPy 数组各有优缺点
          Python lists: 快速追加 (append O(1)), 但计算慢
          NumPy arrays: 慢速创建，但计算很快 (向量化)
        - Lists good for appending, bad for computation
        - NumPy arrays bad for appending, great for computation
        
        本方法在导出时做一次转换，权衡两者优点
        This method converts once at export time
        
        返回值 / Returns:
            dict 包含 4 个键 / dict with 4 keys:
            {
                "time_array": np.ndarray    # shape (N,), dtype float64
                "erg_signal": np.ndarray    # shape (N,), dtype float64
                "activation": np.ndarray    # shape (N,), dtype float64
                "force": np.ndarray         # shape (N,), dtype float64
            }
        
        数组大小 / Array shapes:
            如果记录了 1000 个样本 / If we recorded 1000 samples:
            - time_array:   (1000,)  例 / e.g. [0.000, 0.002, ..., 1.998]
            - erg_signal:   (1000,)  例 / e.g. [0.001, 0.002, ..., 0.050]
            - activation:   (1000,)  例 / e.g. [0.500, 0.500, ..., 0.600]
            - force:        (1000,)  例 / e.g. [1.200, 1.250, ..., 1.500]
        
        用法 / Usage:
            >>> rec = ErgRecorder("masseter")
            >>> rec.record_step(0.000, 0.01, 0.5, 1.0)
            >>> rec.record_step(0.002, 0.02, 0.5, 1.1)
            >>> arrays = rec.to_numpy()
            >>> arrays["time_array"]
            array([0.000, 0.002])
            >>> arrays["erg_signal"]
            array([0.01, 0.02])
        """
        return {
            "time_array": np.array(self.time_array),
            "erg_signal": np.array(self.erg_signal),
            "activation": np.array(self.activation),
            "force": np.array(self.force),
        }

    def save_npz(self, out_dir: str = "erg_outputs") -> str:
        """Save as compressed NPZ file / 保存为 NPZ 压缩文件.
        
        ───────────────────────────────────────────────────────────────
        数据持久化到硬盘 (NPZ 格式) / Data persistence (NPZ format)
        ───────────────────────────────────────────────────────────────
        
        什么是 NPZ？/ What is NPZ?
        - NumPy Zipped 格式 / NumPy Zipped format
        - 是 ZIP 压缩包，包含多个 NumPy .npy 文件
        - A ZIP archive containing multiple NumPy .npy files
        - 通过 zlib 压缩减小文件大小 40-60%
        - Compression reduces file size by 40-60%
        
        为什么选 NPZ？/ Why NPZ?
        ✓ 快速: 直接二进制读写，无解析开销 / Direct binary I/O, no parsing
        ✓ 精确: 浮点数全精度保留 / Full float precision preserved
        ✓ 空间: 自动压缩，节省硬盘空间 / Auto-compression saves space
        ✓ 兼容: NumPy/SciPy/Matplotlib 原生支持 / Native support
        ✗ 不可读: 人类看不懂二进制格式 / Not human-readable
        
        文件大小对比 / File size comparison:
        1000 个样本 / 1000 samples:
        - CSV (文本):       ~20 KB (可读)
        - NPZ (压缩):       ~3 KB (不可读但快速)
        - 压缩率 / compression: 15% of CSV size
        
        导出后的结构 / Resulting NPZ structure:
        erg_outputs/masseter_L_erg.npz
            ├─ time_array       [0.000, 0.002, 0.004, ...]
            ├─ erg_signal       [0.001, 0.002, 0.005, ...]
            ├─ activation       [0.500, 0.500, 0.500, ...]
            └─ force            [1.200, 1.250, 1.300, ...]
        
        如何读取 / How to read:
            >>> data = np.load("masseter_L_erg.npz")
            >>> data.files  # 查看内部数组名
            ['time_array', 'erg_signal', 'activation', 'force']
            >>> data["erg_signal"]  # 访问特定数组
            array([0.001, 0.002, 0.005, ...])
        
        参数 / Parameters:
            out_dir (str): 输出目录，如果不存在会创建
                          Output directory, created if not exists
                          默认: "erg_outputs" / default: "erg_outputs"
        
        返回值 / Returns:
            str: 生成的文件完整路径 / full path to saved file
            
        侧效应 / Side effects:
            - 在终端打印 "✓ Saved: ..." 表示成功
            - Prints "✓ Saved: ..." to terminal
            - 在 out_dir 中创建新文件 / creates file in out_dir
        """
        # 确保输出目录存在，如果不存在则创建（包括所有父目录）
        # Ensure output directory exists, create if needed (including parents)
        # 例如: "erg_outputs/2024-01-15" 会创建所有中间目录
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # 构建输出文件名：{muscle_name}_erg.npz
        # Build output filename: {muscle_name}_erg.npz
        # 例如: "masseter_L_erg.npz"
        fname = os.path.join(out_dir, f"{self.muscle_name}_erg.npz")
        
        # 使用 NumPy 的 savez_compressed 将字典保存为压缩 NPZ
        # np.savez_compressed 会:
        # 1. 将所有数组转换为二进制 .npy 格式
        # 2. 用 zlib 压缩
        # 3. 打包成 ZIP 格式
        np.savez_compressed(fname, **self.to_numpy())
        
        # 输出成功消息（用户反馈）
        # Print success message for user feedback
        print(f"✓ Saved: {fname}")
        
        return fname

    def save_csv(self, out_dir: str = "erg_outputs") -> str:
        """Save as CSV file for easy inspection / 保存为 CSV 文件.
        
        ───────────────────────────────────────────────────────────────
        数据持久化到硬盘 (CSV 格式) / Data persistence (CSV format)
        ───────────────────────────────────────────────────────────────
        
        什么是 CSV？/ What is CSV?
        - Comma-Separated Values 逗号分隔值
        - 纯文本格式，任何编辑器都能打开
        - 可以在 Excel、Google Sheets 中打开
        - Can open in Excel, Google Sheets, any text editor
        
        为什么选 CSV？/ Why CSV?
        ✓ 可读: 人类可以直接看懂 / Human-readable
        ✓ 通用: Excel/Sheets/Python/R 都支持 / Universal support
        ✓ 检查: 快速验证数据是否正确 / Quick data validation
        ✗ 大: 文件比 NPZ 大 5-10 倍 / 5-10x larger than NPZ
        ✗ 慢: 文本解析比二进制读取慢 / Slower than binary
        ✗ 精度: 浮点数四舍五入到有限小数位 / Limited precision
        
        何时使用哪个？/ When to use which?
        - NPZ: 科学计算、数据分析、编程处理 / for computation
        - CSV: 快速查看、与非技术人员分享、Excel 分析
        
        文件格式 / File format:
        time,activation,force,erg_signal
        0.000000,0.500000,1.200000,0.001000
        0.002000,0.500000,1.250000,0.002000
        0.004000,0.500000,1.300000,0.005000
        ...
        
        参数 / Parameters:
            out_dir (str): 输出目录 / output directory
                          默认: "erg_outputs"
        
        返回值 / Returns:
            str: 生成的文件完整路径 / full path to saved file
        """
        # 创建输出目录
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # 构建输出文件名：{muscle_name}_erg.csv
        fname = os.path.join(out_dir, f"{self.muscle_name}_erg.csv")
        
        # 转换为 NumPy 数组，用于快速访问
        data = self.to_numpy()
        
        # 使用 numpy 写入 CSV
        # 手工写 CSV 而不用 pandas，减少依赖
        with open(fname, 'w') as f:
            # 写入表头 / Write header
            # 4 列: time, activation, force, erg_signal
            f.write("time,activation,force,erg_signal\n")
            
            # 逐行写入数据 / Write data row by row
            # 每行 4 个数据，用逗号分隔，保留 6 位小数
            for i in range(len(self.time_array)):
                f.write(f"{data['time_array'][i]:.6f},"
                       f"{data['activation'][i]:.6f},"
                       f"{data['force'][i]:.6f},"
                       f"{data['erg_signal'][i]:.6f}\n")
        
        print(f"✓ Saved: {fname}")
        return fname

    def summary_stats(self) -> dict[str, float]:
        """Return summary statistics / 返回汇总统计.
        
        ───────────────────────────────────────────────────────────────
        快速计算关键指标 / Quickly compute key metrics
        ───────────────────────────────────────────────────────────────
        
        作用 / Purpose:
        - 在不导出完整数据的情况下获取快速概览
        - Quick overview without exporting full data
        - 帮助用户评估录制的质量
        - Helps assess recording quality
        - 用于日志输出、进度报告
        - Used for logging, progress reporting
        
        返回的指标 / Returned metrics:
        
        1. n_samples (int): 记录的总样本数
           Total number of recorded samples
           例如 / e.g. 26947 samples
           用于判断录制时间长度 / indicates recording duration
        
        2. duration_s (float): 从开始到结束的总时间（秒）
           Total time from start to end (seconds)
           = time_array[-1] - time_array[0]
           例如 / e.g. 53.89 seconds
        
        3. mean_erg (float): ERG 信号的平均值
           Mean ERG value, range [0, 1]
           表示平均肌肉活动强度
           represents typical muscle activation level
           低值 (< 0.05) = 肌肉大部分时间放松
           高值 (> 0.2) = 肌肉经常收缩
        
        4. max_erg (float): ERG 信号的最大值
           Peak ERG value
           肌肉最强的收缩时刻 / strongest muscle contraction
           例如 / e.g. 0.856 = 85.6% 最大强度
        
        5. min_erg (float): ERG 信号的最小值
           Usually close to 0 (rest state)
        
        6. mean_act (float): 激活的平均值
           平均用户输入强度 / average user input
        
        7. max_act (float): 激活的最大值
           用户输入的峰值 / peak user input
        
        8. mean_force (float): 力的平均绝对值
           Average force magnitude (absolute value)
           无单位，但反映肌肉的平均力输出
        
        9. max_force (float): 力的最大绝对值
           Peak force magnitude
        
        实际例子 / Real example:
        >>> rec = ErgRecorder("masseter_L")
        >>> # ... record_step 1000 次 ...
        >>> stats = rec.summary_stats()
        >>> print(stats)
        {
            'n_samples': 26947,
            'duration_s': 53.89,
            'mean_erg': 0.0295,
            'max_erg': 0.8562,
            'min_erg': 0.0001,
            'mean_act': 0.5,
            'max_act': 1.0,
            'mean_force': 1.25,
            'max_force': 2.50
        }
        
        如何解释这个例子？
        - 录制了 26947 个样本，共 53.89 秒
        - 平均 ERG = 0.0295，说明肌肉大部分时间很放松
        - 最大 ERG = 0.8562，用户有过一次很强的收缩
        - 平均激活 = 0.5，用户保持了中等强度
        - 最大力 = 2.50，这是用户能施加的最大力
        
        返回值 / Returns:
            dict[str, float]: 包含 9 个关键指标的字典
                             Dictionary with 9 key metrics
        """
        # 转换为 NumPy 数组进行高效计算
        erg = np.array(self.erg_signal)
        act = np.array(self.activation)
        force = np.abs(np.array(self.force))  # 取绝对值，因为正负力都代表强度
        
        return {
            "n_samples": len(self.time_array),
            "duration_s": self.time_array[-1] if self.time_array else 0.0,
            "mean_erg": float(np.mean(erg)) if len(erg) > 0 else 0.0,
            "max_erg": float(np.max(erg)) if len(erg) > 0 else 0.0,
            "min_erg": float(np.min(erg)) if len(erg) > 0 else 0.0,
            "mean_act": float(np.mean(act)) if len(act) > 0 else 0.0,
            "max_act": float(np.max(act)) if len(act) > 0 else 0.0,
            "mean_force": float(np.mean(force)) if len(force) > 0 else 0.0,
            "max_force": float(np.max(force)) if len(force) > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all recorded data / 清除所有记录的数据.
        
        ───────────────────────────────────────────────────────────────
        重置录制器状态 / Reset recorder state
        ───────────────────────────────────────────────────────────────
        
        作用 / Purpose:
        - 用于新一轮录制前的清理
        - 在做多次录制时，避免数据混杂
        - 释放内存（虽然不多）
        
        用法 / Usage:
            >>> rec = ErgRecorder("masseter")
            >>> rec.record_step(0.000, 0.01, 0.5, 1.0)
            >>> rec.record_step(0.002, 0.02, 0.5, 1.1)
            >>> len(rec)
            2
            >>> rec.clear()
            >>> len(rec)
            0
        
        调用后 / After calling:
        - 所有 4 个数组都变成空列表
        - 可以立即开始新的录制
        - 统计信息会返回 0 / "无数据"
        """
        self.time_array.clear()
        self.erg_signal.clear()
        self.activation.clear()
        self.force.clear()

    def __len__(self) -> int:
        """Return number of recorded samples / 返回记录的样本数.
        
        ───────────────────────────────────────────────────────────────
        方便的长度查询 / Convenient length query
        ───────────────────────────────────────────────────────────────
        
        用法 / Usage:
            >>> rec = ErgRecorder("masseter")
            >>> rec.record_step(0.000, 0.01, 0.5, 1.0)
            >>> rec.record_step(0.002, 0.02, 0.5, 1.1)
            >>> len(rec)
            2
        
        等同于 / Equivalent to:
            >>> len(rec.time_array)
        """
        return len(self.time_array)

    def __repr__(self) -> str:
        """Return string representation / 返回字符串表示.
        
        ───────────────────────────────────────────────────────────────
        便利的调试输出 / Convenient debug representation
        ───────────────────────────────────────────────────────────────
        
        用法 / Usage:
            >>> rec = ErgRecorder("masseter_L")
            >>> rec.record_step(0.000, 0.01, 0.5, 1.0)
            >>> rec.record_step(0.002, 0.02, 0.5, 1.1)
            >>> print(rec)
            ErgRecorder(masseter_L, samples=2, duration=0.00s)
        
        输出中包含:
        1. 肌肉名称 / muscle name
        2. 样本数 / number of samples
        3. 录制时长（秒）/ duration in seconds
        
        这在日志输出中非常有用 / Useful for logging
        """
        stats = self.summary_stats()
        return (f"ErgRecorder({self.muscle_name}, "
                f"samples={stats['n_samples']}, "
                f"duration={stats['duration_s']:.2f}s)")


class MultiMuscleRecorder:
    """Record ERG signals for multiple muscles / 记录多个肌肉的 ERG 信号.
    
    ═══════════════════════════════════════════════════════════════════
    多肌肉并发数据管理 / Concurrent multi-muscle data management
    ═══════════════════════════════════════════════════════════════════
    
    设计 / Design:
    - 内部包含多个 ErgRecorder 实例，一个肌肉一个
    - 通过字典管理，使用肌肉名称为键
    - 提供统一的接口进行批量操作
    
    架构 / Architecture:
    
    ┌──────────────────────────────────────────────┐
    │  MultiMuscleRecorder                         │
    │  (recorders dict)                            │
    ├──────────────────────────────────────────────┤
    │ "masseter_L" → ErgRecorder                   │
    │                ├─ time_array: [0, 0.002, ..]│
    │                ├─ erg_signal: [0, 0.01, ..]│
    │                ├─ activation: [0.5, ..]    │
    │                └─ force: [1.2, ..]         │
    ├──────────────────────────────────────────────┤
    │ "masseter_R" → ErgRecorder                   │
    │                ├─ time_array: [0, 0.002, ..]│
    │                ├─ erg_signal: [0, 0.01, ..]│
    │                ├─ activation: [0.5, ..]    │
    │                └─ force: [1.2, ..]         │
    ├──────────────────────────────────────────────┤
    │ "temporalis_L" → ErgRecorder                 │
    │                ├─ ...                        │
    └──────────────────────────────────────────────┘
    
    用法 / Usage example:
        >>> recorder = MultiMuscleRecorder(
        ...     ["masseter_L", "masseter_R", "temporalis_L"],
        ...     dt=0.002
        ... )
        >>> 
        >>> # 在仿真循环中 / In simulation loop:
        >>> for step in range(1000):
        ...     t = step * 0.002
        ...     muscle_data = {
        ...         "masseter_L": (erg_L, act_L, force_L),
        ...         "masseter_R": (erg_R, act_R, force_R),
        ...         "temporalis_L": (erg_T, act_T, force_T),
        ...     }
        ...     recorder.record_step(t, muscle_data)
        >>> 
        >>> # 导出所有数据 / Export all data:
        >>> files = recorder.save_all()
        >>> # {"masseter_L": "masseter_L_erg.npz", ...}
        >>> 
        >>> # 查看统计 / View stats:
        >>> stats = recorder.summary_all()
        >>> print(stats["masseter_L"]["mean_erg"])
    
    为什么需要这个类？/ Why do we need this class?
    1. 代码简洁 / Code clarity:
       一行调用录制所有肌肉，而不是逐个调用每个 ErgRecorder
    
    2. 接口统一 / Unified interface:
       record_step() 处理所有肌肉，save_all() 导出所有数据
    
    3. 便于扩展 / Scalability:
       添加新肌肉很简单，无需改动录制逻辑
    
    4. 批量操作 / Batch operations:
       比较多个肌肉的统计、一次性导出等
    """
    
    def __init__(self, muscle_names: list[str], dt: float = 0.002):
        """Initialize recorder for multiple muscles / 初始化多肌肉记录器.
        
        ───────────────────────────────────────────────────────────────
        创建多肌肉管理器 / Create multi-muscle manager
        ───────────────────────────────────────────────────────────────
        
        参数 / Parameters:
            muscle_names (list[str]): 肌肉名称列表
                                     List of muscle names
                                     例 / e.g. ["masseter_L", "masseter_R"]
                                     这些名称会用作字典键 / used as dict keys
                                     和文件名前缀 / and file name prefixes
            
            dt (float): 仿真时间步长，单位秒
                       Simulation timestep, default 0.002s (500 Hz)
                       所有肌肉共享相同的 dt
                       All muscles share the same dt
        
        示例 / Example:
            >>> rec = MultiMuscleRecorder(
            ...     ["masseter_L", "masseter_R"],
            ...     dt=0.002
            ... )
            >>> rec.recorders.keys()
            dict_keys(['masseter_L', 'masseter_R'])
        
        内部状态 / Internal state:
            self.recorders:  dict[str, ErgRecorder]
            self.dt:         float (0.002)
        """
        # 创建一个字典，每个肌肉对应一个 ErgRecorder
        # Create a dict of ErgRecorder instances
        self.recorders = {
            name: ErgRecorder(name, dt=dt) for name in muscle_names
        }
        self.dt = dt

    def record_step(
        self,
        t: float,
        muscle_data: dict[str, tuple[float, float, float]],
    ) -> None:
        """Record one step for all muscles / 为所有肌肉记录一步.
        
        ───────────────────────────────────────────────────────────────
        批量录制单个时间步 / Batch record one timestep
        ───────────────────────────────────────────────────────────────
        
        设计模式 / Design pattern:
        这个方法简化了多肌肉录制的过程。不需要逐个调用每个 ErgRecorder：
        
        不推荐的做法 / Not recommended:
            for muscle_name in muscle_names:
                erg, act, force = muscle_data[muscle_name]
                recorder.recorders[muscle_name].record_step(t, erg, act, force)
        
        推荐的做法 / Recommended:
            recorder.record_step(t, muscle_data)
        
        参数 / Parameters:
            t (float): 当前仿真时间（秒）/ current simulation time
                      例 / e.g. 0.000, 0.002, 0.004, ...
            
            muscle_data (dict[str, tuple[float, float, float]]):
                肌肉名称 → (erg, activation, force)
                muscle_name → (erg_value, act_value, force_value)
                
                例 / Example:
                {
                    "masseter_L": (0.01, 0.5, 1.2),
                    "masseter_R": (0.02, 0.6, 1.3),
                }
                
                如果 muscle_data 包含了初始化时未指定的肌肉，
                会被默默忽略 (safe to pass extra data)
        
        性能 / Performance:
        - O(m) 时间，m = 肌肉数量 / time complexity O(m) where m = number of muscles
        - 对于通常的 4-6 个肌肉很快 / Fast for typical 4-6 muscles
        
        用法 / Usage:
            >>> rec = MultiMuscleRecorder(["masseter_L", "masseter_R"])
            >>> rec.record_step(0.000, {
            ...     "masseter_L": (0.01, 0.5, 1.2),
            ...     "masseter_R": (0.02, 0.6, 1.3),
            ... })
            >>> rec["masseter_L"].erg_signal
            [0.01]
            >>> rec["masseter_R"].erg_signal
            [0.02]
        """
        # 遍历 muscle_data 中的每个肌肉
        # Iterate through each muscle in muscle_data
        for muscle_name, (erg, act, force) in muscle_data.items():
            # 检查这个肌肉是否在我们管理的肌肉列表中
            # Only record if this muscle is in our recorders
            if muscle_name in self.recorders:
                # 调用该肌肉对应的 ErgRecorder 的 record_step
                self.recorders[muscle_name].record_step(t, erg, act, force)
            # 否则忽略（用户可能传了多余的肌肉数据）
            # Otherwise silently ignore (user might pass extra muscles)

    def save_all(self, out_dir: str = "erg_outputs") -> dict[str, dict]:
        """Save all muscle data (NPZ + CSV) / 保存所有肌肉数据 (NPZ + CSV).
        
        ───────────────────────────────────────────────────────────────
        批量导出所有肌肉的 NPZ 和 CSV 数据 / Batch export all muscle data
        ───────────────────────────────────────────────────────────────
        
        用法 / Usage:
            >>> rec = MultiMuscleRecorder(["masseter_L", "masseter_R"])
            >>> # ... record steps ...
            >>> files = rec.save_all()
            >>> print(files)
            {
                'masseter_L': {
                    'npz': '/path/to/masseter_L_erg.npz',
                    'csv': '/path/to/masseter_L_erg.csv'
                },
                'masseter_R': {
                    'npz': '/path/to/masseter_R_erg.npz',
                    'csv': '/path/to/masseter_R_erg.csv'
                }
            }
        
        参数 / Parameters:
            out_dir (str): 输出目录 / output directory
                          默认: "erg_outputs"
                          如果不存在会创建 / created if not exists
        
        返回值 / Returns:
            dict[str, dict]: 
                {
                    muscle_name: {
                        'npz': file_path,
                        'csv': file_path
                    }
                }
        
        侧效应 / Side effects:
        - 在终端打印每个文件的保存消息
        - Creates out_dir if not exists
        - Creates .npz files for each muscle
        - Creates .csv files for each muscle
        
        用法模式 / Usage pattern:
            >>> rec = MultiMuscleRecorder(["m_L", "m_R"])
            >>> # ... record ...
            >>> files = rec.save_all("data/2024-01-15")
            >>> # 输出:
            >>> # ✓ Saved: data/2024-01-15/m_L_erg.npz
            >>> # ✓ Saved: data/2024-01-15/m_L_erg.csv
            >>> # ✓ Saved: data/2024-01-15/m_R_erg.npz
            >>> # ✓ Saved: data/2024-01-15/m_R_erg.csv
        """
        result = {}
        # 逐个肌肉调用其 save_npz() 和 save_csv() 方法
        for muscle_name, recorder in self.recorders.items():
            result[muscle_name] = {
                'npz': recorder.save_npz(out_dir),
                'csv': recorder.save_csv(out_dir)
            }
        return result

    def summary_all(self) -> dict[str, dict]:
        """Get statistics for all muscles / 获取所有肌肉的统计.
        
        ───────────────────────────────────────────────────────────────
        批量获取统计信息 / Batch get statistics
        ───────────────────────────────────────────────────────────────
        
        返回值 / Returns:
            dict[str, dict]:
                外层键 = 肌肉名称 / outer key = muscle name
                内层字典 = 该肌肉的统计信息 / inner dict = stats
                
                返回的统计包含 9 个指标 / 9 metrics:
                n_samples, duration_s, mean_erg, max_erg, min_erg,
                mean_act, max_act, mean_force, max_force
        
        用法 / Usage:
            >>> rec = MultiMuscleRecorder(["masseter_L", "masseter_R"])
            >>> # ... record ...
            >>> all_stats = rec.summary_all()
            >>> print(all_stats["masseter_L"])
            {
                'n_samples': 26947,
                'duration_s': 53.89,
                'mean_erg': 0.0295,
                'max_erg': 0.8562,
                ...
            }
        
        比较肌肉 / Comparing muscles:
            >>> all_stats = rec.summary_all()
            >>> left_erg = all_stats["masseter_L"]["mean_erg"]
            >>> right_erg = all_stats["masseter_R"]["mean_erg"]
            >>> print(f"Left mean ERG: {left_erg:.4f}")
            >>> print(f"Right mean ERG: {right_erg:.4f}")
            >>> print(f"Difference: {abs(left_erg - right_erg):.4f}")
        """
        return {
            muscle_name: recorder.summary_stats()
            for muscle_name, recorder in self.recorders.items()
        }

    def __getitem__(self, muscle_name: str) -> ErgRecorder:
        """Get recorder for specific muscle / 获取特定肌肉的记录器.
        
        ───────────────────────────────────────────────────────────────
        便利的索引访问 / Convenient indexing access
        ───────────────────────────────────────────────────────────────
        
        用法 / Usage:
            >>> rec = MultiMuscleRecorder(["masseter_L", "masseter_R"])
            >>> rec["masseter_L"]
            ErgRecorder(masseter_L, samples=0, duration=0.00s)
            >>> rec["masseter_L"].time_array
            []
        
        等同于 / Equivalent to:
            >>> rec.recorders["masseter_L"]
        
        这个方法使代码更简洁 / This method makes code more concise:
            rec["masseter_L"] vs rec.recorders["masseter_L"]
        """
        return self.recorders[muscle_name]
