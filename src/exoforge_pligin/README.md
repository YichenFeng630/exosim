# ERG Plugin - 项目文档

## 📋 项目概述

ERG (Electrogoniogram) 插件系统，用于肌肉信号处理、数据记录和可视化。

项目已完全重构为三层架构：
- **插件核心** (4 模块) - ERG 信号处理
- **演示模块** (1 文件) - 实时 MuJoCo 演示
- **测试模块** (1 文件) - 数据生成和测试


## 🏗️ 核心文件说明

### 插件核心 (Plugin Core)

| 文件 | 功能 | 说明 |
|------|------|------|
| **erg_plugin_core.py** | 核心算法 | 5 阶 ERG 信号处理：激活 → 非线性 → 高通 → 整流 → 低通 |
| **erg_plugin.py** | 主接口 | ErgPlugin, ErgBatch, ErgActuatorGroup 等类 |
| **erg_recorder.py** | 数据录制 | ErgRecorder, MultiMuscleRecorder, NPZ/CSV 导出 |
| **erg_visualization.py** | 离线绘图 | plot_erg_signal, plot_multi_erg, plot_comparison 等 |

### 演示模块 (Demonstration)

| 文件 | 功能 |
|------|------|
| **viewer_realtime_plot.py** | 实时 MuJoCo Viewer + matplotlib 演示 |

### 测试模块 (Testing)

| 文件 | 功能 |
|------|------|
| **test.py** | 完整测试框架，支持肌肉选择、数据生成 |


## 🚀 快速开始

### 1. 运行测试
```bash
python test.py
```

### 2. 选择肌肉（多种方式）
```
0                          # 单肌肉
0,2,4                      # 多肌肉
superficial_masseter_left  # 按名称
```

### 3. 查看结果
```bash
ls erg_test/
# 查看生成的数据和图表
```


## 📊 输出文件结构

```
erg_test/
└── superficial_20251207_165939/           # 时间戳文件夹
    ├── *.npz                              # 压缩数据（NumPy）
    ├── *.csv                              # CSV 文本数据
    └── plots/
        ├── *_erg_signal.png               # ERG 详细分析
        ├── *_time_series.png              # 时间序列
        └── multi_muscle_comparison.png    # 多肌肉对比（可选）
```

**数据格式说明：**
- **NPZ**: 压缩二进制，包含 time_array, erg_signal, activation, force
- **CSV**: 文本格式，可直接用 Excel 打开
- **PNG**: 高质量图表（150 DPI）


## 💻 编程 API 使用

### 基础用法
```python
from erg_plugin_core import ErgFilter
from erg_recorder import ErgRecorder
from erg_visualization import plot_erg_signal

# 创建滤波器和录制器
filter = ErgFilter()
recorder = ErgRecorder("superficial_masseter_left")

# 处理一步数据
erg = filter.step(activation=0.5, force=1.0)
recorder.record_step(t=0.002, erg=erg, act=0.5, force=1.0)

# 保存和绘图
recorder.save_npz("erg_test")
recorder.save_csv("erg_test")
plot_erg_signal(recorder, save_path="erg_test/plot.png")
```

### 批量处理
```python
from erg_recorder import MultiMuscleRecorder

recorder = MultiMuscleRecorder(["muscle1", "muscle2"])

for step in range(15000):
    muscle_data = {
        "muscle1": (erg1, act1, force1),
        "muscle2": (erg2, act2, force2),
    }
    recorder.record_step(t=step*0.002, muscle_data=muscle_data)

files = recorder.save_all("erg_test")  # 返回 NPZ + CSV 路径
```


## ⚙️ 配置参数

### ErgFilter 参数

```python
filter = ErgFilter(
    k=3.0,              # 非线性强度 (1.0-5.0)
    alpha=0.2,          # 力权重 (0.0-1.0)
    noise_std=0.02,     # 噪声强度 (0.0-0.1)
    a_hp=0.995,         # 高通强度 (0.9-0.999)
    a_lp=0.90,          # 低通平滑 (0.5-0.99)
    fmax=1.0            # 力归一化 (>0)
)
```

### 仿真参数

在 test.py 中修改：
```python
controller.run_simulated_session(duration=60.0)  # 改为 60 秒
```


## 🔍 数据检查

### 查看 NPZ 内容
```python
import numpy as np
data = np.load("erg_test/muscle_name_erg.npz")
print(data.files)  # ['time_array', 'erg_signal', 'activation', 'force']
```

### 查看 CSV 内容
```python
import pandas as pd
df = pd.read_csv("erg_test/muscle_name_erg.csv")
print(df.head())
```


## 📈 数据规格

| 参数 | 值 |
|------|-----|
| 采样率 | 500 Hz (dt = 0.002 s) |
| 默认时长 | 30 秒 |
| 采样点数 | 15000 |
| 激活范围 | [0.0, 1.0] |
| 力范围 | ~[0.8, 1.2] N (模拟) |
| NPZ 压缩率 | 40-60% |


## ❓ 常见问题

**Q: 如何选择多个肌肉?**
A: 输入 `0,2,4` 或 `0,1,2,3,4,5` 等逗号分隔的索引

**Q: 如何改变仿真时长?**
A: 修改 test.py 中的 `controller.run_simulated_session(duration=X)`

**Q: NPZ vs CSV 应该用哪个?**
A: NPZ 更小更快（处理用），CSV 更易读（分析用）

**Q: 如何离线使用（无图形显示）?**
A: erg_visualization 已支持离线模式，设置 save_path 即可

**Q: 支持哪些肌肉?**
A: 查看 test.py 中的 `AVAILABLE_MUSCLES` 列表


## 📚 详细文档

- **ARCHITECTURE.md** - 详细的系统架构说明
- **QUICK_REFERENCE.txt** - 快速参考卡


## 🛠️ 环境要求

- Python 3.8+
- numpy
- matplotlib
- mujoco


## 📞 快速链接

| 任务 | 命令 |
|------|------|
| 运行测试 | `python test.py` |
| 实时演示 | `python viewer_realtime_plot.py` |
| 查看肌肉列表 | 查看 test.py 中 `AVAILABLE_MUSCLES` |
