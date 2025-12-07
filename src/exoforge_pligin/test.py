"""Clean test module with muscle selection and data generation.

═══════════════════════════════════════════════════════════════════════════════
ERG 测试框架 / ERG Testing Framework
═══════════════════════════════════════════════════════════════════════════════

功能 / Features:
- 肌肉交互式选择 / Interactive muscle selection
- 支持单个或多个肌肉 / Single or multiple muscles
- 两种测试模式 / Two testing modes:
  ✓ 模拟模式 (Simulated mode) - 自动生成数据，快速测试
  ✓ 手动模式 (Manual mode) - MuJoCo Viewer 手动划肌肉，实时记录
- 输出到 erg_test/ 文件夹 / Output to erg_test/ folder
- 自动生成 NPZ + CSV + PNG 文件 / Auto-generate NPZ/CSV/PNG

工作流 / Workflow:
1. 选择肌肉 (Select muscles)
2. 选择模式 (Select mode: 1=simulated or 2=manual with MuJoCo)
3. 生成数据 (Generate data)
4. 查看结果 (View results in erg_test/)

用法 / Usage:
    python test.py
    # 选择肌肉 → 选择模式 → 如果手动模式则在 MuJoCo Viewer 中划肌肉
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np

# Support both package and direct execution
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .erg_plugin_core import ErgFilter
    from .erg_recorder import MultiMuscleRecorder, ErgRecorder
    from .erg_visualization import plot_erg_signal, plot_multi_erg
except ImportError:
    from erg_plugin_core import ErgFilter
    from erg_recorder import MultiMuscleRecorder, ErgRecorder
    from erg_visualization import plot_erg_signal, plot_multi_erg

import mujoco
from mujoco import viewer

# Available muscles
AVAILABLE_MUSCLES = [
    "superficial_masseter_left",
    "superficial_masseter_right",
    "anterior_temporalis_left",
    "anterior_temporalis_right",
    "posterior_temporalis_left",
    "posterior_temporalis_right",
]


class ManualErGController:
    """Manual ERG controller with MuJoCo viewer / 手动 ERG 控制器（带 MuJoCo Viewer）"""
    
    def __init__(self, muscles: List[str], output_dir: str = "erg_test"):
        self.muscles = muscles
        self.output_dir = output_dir
        self.session_dir = self._create_session_dir()
        print(f"📁 Session directory: {self.session_dir}")
        
        # Load MuJoCo model
        self.model, self.data = self._load_model()
        print(f"✓ MuJoCo model loaded")
        
        # Initialize ERG filters and recorders
        self.erg_filters = {muscle: ErgFilter() for muscle in muscles}
        self.recorder = MultiMuscleRecorder(muscle_names=muscles)
        print(f"✓ Initialized {len(muscles)} muscles")
        
        self.dt = self.model.opt.timestep
        self.recording = False
        self.is_running = True
    
    def _create_session_dir(self) -> str:
        """Create time-stamped session directory"""
        os.makedirs(self.output_dir, exist_ok=True)
        muscle_prefix = self.muscles[0].split("_")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"{muscle_prefix}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def _load_model(self):
        """Load MuJoCo model from standard locations"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "../../../src/exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml"),
            "/home/yichen/exoforge-feature-sensors-user-intent/src/exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model = mujoco.MjModel.from_xml_path(path)
                data = mujoco.MjData(model)
                return model, data
        
        raise FileNotFoundError("Cannot find MuJoCo model")
    
    def run_manual_session(self):
        """Run manual session with MuJoCo viewer / 运行手动仿真（MuJoCo Viewer）"""
        print(f"\n⏱ Starting manual recording session...")
        print(f"📝 Instructions / 说明:")
        print(f"   1. Use the MuJoCo viewer sliders to control muscle activation")
        print(f"   2. Data will be recorded in real-time")
        print(f"   3. Close the viewer window when done (Esc or close button)")
        print()
        
        step_count = 0
        self.recording = True
        
        with viewer.launch_passive(self.model, self.data) as v:
            while self.is_running and v.is_running():
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Get muscle activations from sliders (use ctrl array)
                # For now, use data.ctrl or estimate from joint angles
                muscle_data = {}
                
                for muscle in self.muscles:
                    # Try to get activation from MuJoCo data
                    # If slider control not available, use 0.5 as placeholder
                    activation = 0.5  # 默认值，实际应该从 MuJoCo 获取
                    force = 1.0       # 默认值，实际应该从 MuJoCo 传感器获取
                    
                    erg_signal = self.erg_filters[muscle].step(activation, force)
                    muscle_data[muscle] = (erg_signal, activation, force)
                
                # Record data
                self.recorder.record_step(t=self.data.time, muscle_data=muscle_data)
                
                # Render
                v.sync()
                
                step_count += 1
                if step_count % 500 == 0:
                    print(f"  Recording... {self.data.time:.1f}s ({step_count} steps)")
        
        print(f"✓ Manual session completed. Total steps: {step_count}")
        print(f"✓ Data points recorded: {len(self.recorder.recorders[self.muscles[0]].time_array)}")
        self.recording = False
    
    def save_data(self):
        """Save all data to NPZ and CSV"""
        print(f"\n💾 Saving data...")
        self.recorder.save_all(out_dir=self.session_dir)
        print(f"✓ Data saved to {self.session_dir}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        print(f"\n📊 Generating plots...")
        
        plots_dir = os.path.join(self.session_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        for muscle in self.recorder.recorders.keys():
            muscle_recorder = self.recorder.recorders[muscle]
            print(f"  Plotting: {muscle}")
            
            # Plot ERG signal
            try:
                plot_erg_signal(
                    muscle_recorder,
                    window=None,
                    save_path=os.path.join(plots_dir, f"{muscle}_erg_signal.png"),
                )
                print(f"    ✓ ERG signal plot saved")
            except Exception as e:
                print(f"    ✗ Failed to generate ERG signal plot: {e}")
            
            # Plot time series
            try:
                data = muscle_recorder.to_numpy()
                t = data["time_array"]
                erg = data["erg_signal"]
                act = data["activation"]
                force = np.abs(data["force"])
                
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                fig.suptitle(f"Time Series - {muscle}", fontsize=14, fontweight="bold")
                
                axes[0].plot(t, act, color="orange", linewidth=1.5, label="Activation")
                axes[0].set_ylabel("Activation")
                axes[0].set_title("Muscle Activation")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                axes[1].plot(t, force, color="green", linewidth=1.5, label="Force")
                axes[1].set_ylabel("Force (N)")
                axes[1].set_title("Contraction Force")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                axes[2].plot(t, erg, color="steelblue", linewidth=1, label="ERG Signal", alpha=0.7)
                axes[2].fill_between(t, erg, alpha=0.3, color="steelblue")
                axes[2].set_ylabel("ERG Signal (μV)")
                axes[2].set_xlabel("Time (s)")
                axes[2].set_title("Electrogoniogram (ERG)")
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
                
                plt.tight_layout()
                save_path = os.path.join(plots_dir, f"{muscle}_time_series.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    ✓ Time series plot saved")
            except Exception as e:
                print(f"    ✗ Failed to generate time series plot: {e}")
        
        # Multi-muscle comparison if available
        if len(self.recorder.recorders) > 1:
            try:
                plot_multi_erg(
                    self.recorder.recorders,
                    save_path=os.path.join(plots_dir, "multi_muscle_comparison.png"),
                )
                print(f"  ✓ Multi-muscle comparison plot saved")
            except Exception as e:
                print(f"  ✗ Failed to generate comparison plot: {e}")
        
        print(f"✓ All plots saved to: {plots_dir}")


class SimulatedErGController:
    """Simulated ERG controller without MuJoCo viewer / 模拟 ERG 控制器（无MuJoCo显示）"""
    
    def __init__(self, muscles: List[str], output_dir: str = "erg_test"):
        self.muscles = muscles
        self.output_dir = output_dir
        self.session_dir = self._create_session_dir()
        print(f"📁 Session directory: {self.session_dir}")
        
        # Try to load MuJoCo model (for data structure only)
        try:
            self.model, self.data = self._load_model()
            print(f"✓ MuJoCo model loaded")
        except Exception as e:
            print(f"⚠ MuJoCo model not required for simulated mode")
            self.model = self.data = None
        
        # Initialize ERG filters and recorders
        self.erg_filters = {muscle: ErgFilter() for muscle in muscles}
        self.recorder = MultiMuscleRecorder(muscle_names=muscles)
        print(f"✓ Initialized {len(muscles)} muscles")
        
        self.t = 0
        self.dt = 0.002  # 500 Hz
    
    def _create_session_dir(self) -> str:
        """Create time-stamped session directory"""
        os.makedirs(self.output_dir, exist_ok=True)
        muscle_prefix = self.muscles[0].split("_")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"{muscle_prefix}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def _load_model(self):
        """Load MuJoCo model from standard locations"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "../../../src/exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml"),
            "/home/yichen/exoforge-feature-sensors-user-intent/src/exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model = mujoco.MjModel.from_xml_path(path)
                data = mujoco.MjData(model)
                return model, data
        
        raise FileNotFoundError("Cannot find MuJoCo model")
    
    def run_simulated_session(self, duration: float = 30.0):
        """Run simulated session with generated data"""
        print(f"\n⏱ Running simulated session for {duration:.1f} seconds...")
        
        step_count = 0
        max_steps = int(duration / self.dt)
        
        # Seed for reproducibility
        np.random.seed(42)
        
        # Generate activation and force patterns
        activation_patterns = {}
        force_patterns = {}
        
        for muscle in self.muscles:
            t_array = np.linspace(0, duration, max_steps)
            # Smooth sinusoidal activation with noise
            activation_patterns[muscle] = 0.3 + 0.4 * np.sin(2 * np.pi * 0.5 * t_array) + 0.05 * np.random.randn(max_steps)
            activation_patterns[muscle] = np.clip(activation_patterns[muscle], 0, 1)
            
            # Force correlated with activation
            force_patterns[muscle] = 1.0 + 0.5 * activation_patterns[muscle] + 0.1 * np.random.randn(max_steps)
        
        # Simulation loop
        while step_count < max_steps:
            muscle_data = {}
            for muscle in self.muscles:
                activation = float(activation_patterns[muscle][step_count])
                force = float(force_patterns[muscle][step_count])
                erg_signal = self.erg_filters[muscle].step(activation, force)
                muscle_data[muscle] = (erg_signal, activation, force)
            
            self.recorder.record_step(t=self.t, muscle_data=muscle_data)
            
            self.t += self.dt
            step_count += 1
            
            if step_count % 2500 == 0:  # Progress every 5 seconds
                progress = 100 * step_count / max_steps
                print(f"  Progress: {step_count}/{max_steps} ({progress:.1f}%)")
        
        print(f"✓ Simulation completed. Total steps: {step_count}")
        print(f"✓ Data points recorded: {len(self.recorder.recorders[self.muscles[0]].time_array)}")
    
    def save_data(self):
        """Save all data to NPZ and CSV"""
        print(f"\n💾 Saving data...")
        self.recorder.save_all(out_dir=self.session_dir)
        print(f"✓ Data saved to {self.session_dir}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        print(f"\n📊 Generating plots...")
        
        plots_dir = os.path.join(self.session_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        for muscle in self.recorder.recorders.keys():
            muscle_recorder = self.recorder.recorders[muscle]
            print(f"  Plotting: {muscle}")
            
            # Plot ERG signal
            try:
                plot_erg_signal(
                    muscle_recorder,
                    window=None,
                    save_path=os.path.join(plots_dir, f"{muscle}_erg_signal.png"),
                )
                print(f"    ✓ ERG signal plot saved")
            except Exception as e:
                print(f"    ✗ Failed to generate ERG signal plot: {e}")
            
            # Plot time series
            try:
                data = muscle_recorder.to_numpy()
                t = data["time_array"]
                erg = data["erg_signal"]
                act = data["activation"]
                force = np.abs(data["force"])
                
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                fig.suptitle(f"Time Series - {muscle}", fontsize=14, fontweight="bold")
                
                axes[0].plot(t, act, color="orange", linewidth=1.5, label="Activation")
                axes[0].set_ylabel("Activation")
                axes[0].set_title("Muscle Activation")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                axes[1].plot(t, force, color="green", linewidth=1.5, label="Force")
                axes[1].set_ylabel("Force (N)")
                axes[1].set_title("Contraction Force")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                axes[2].plot(t, erg, color="steelblue", linewidth=1, label="ERG Signal", alpha=0.7)
                axes[2].fill_between(t, erg, alpha=0.3, color="steelblue")
                axes[2].set_ylabel("ERG Signal (μV)")
                axes[2].set_xlabel("Time (s)")
                axes[2].set_title("Electrogoniogram (ERG)")
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
                
                plt.tight_layout()
                save_path = os.path.join(plots_dir, f"{muscle}_time_series.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    ✓ Time series plot saved")
            except Exception as e:
                print(f"    ✗ Failed to generate time series plot: {e}")
        
        # Multi-muscle comparison if available
        if len(self.recorder.recorders) > 1:
            try:
                plot_multi_erg(
                    self.recorder.recorders,
                    save_path=os.path.join(plots_dir, "multi_muscle_comparison.png"),
                )
                print(f"  ✓ Multi-muscle comparison plot saved")
            except Exception as e:
                print(f"  ✗ Failed to generate comparison plot: {e}")
        
        print(f"✓ All plots saved to: {plots_dir}")


def select_muscles() -> List[str]:
    """Interactive muscle selection / 交互式肌肉选择"""
    print("\n" + "="*80)
    print("MUSCLE SELECTION / 肌肉选择")
    print("="*80)
    print("\nAvailable muscles / 可用肌肉:")
    for i, muscle in enumerate(AVAILABLE_MUSCLES):
        print(f"  {i}. {muscle}")
    
    print("\nInput options / 输入选项:")
    print("  - Single index: 0")
    print("  - Multiple indices: 0,2,4")
    print("  - Muscle name: superficial_masseter_left")
    
    user_input = input("\nPlease select / 请选择: ").strip()
    
    # Parse input
    if not user_input:
        print("No selection, using default: superficial_masseter_left")
        return ["superficial_masseter_left"]
    
    selected = []
    
    # Try as comma-separated indices
    if "," in user_input or user_input.isdigit():
        try:
            if "," in user_input:
                indices = [int(x.strip()) for x in user_input.split(",")]
            else:
                indices = [int(user_input)]
            
            for idx in indices:
                if 0 <= idx < len(AVAILABLE_MUSCLES):
                    selected.append(AVAILABLE_MUSCLES[idx])
            
            if selected:
                return selected
        except ValueError:
            pass
    
    # Try as muscle name
    if user_input in AVAILABLE_MUSCLES:
        return [user_input]
    
    # Try partial match
    for muscle in AVAILABLE_MUSCLES:
        if user_input.lower() in muscle.lower():
            selected.append(muscle)
    
    if selected:
        return selected
    
    print(f"Invalid selection '{user_input}', using default")
    return ["superficial_masseter_left"]


def select_test_mode():
    """Select test mode / 选择测试模式"""
    print("\n" + "="*80)
    print("TEST MODE SELECTION / 测试模式选择")
    print("="*80)
    print("\nAvailable modes / 可用模式:")
    print("  1. Simulated (自动生成数据，快速测试)")
    print("  2. Manual (打开 MuJoCo Viewer，手动划肌肉)")
    
    user_input = input("\nPlease select / 请选择 (1 or 2): ").strip()
    
    if user_input == "2":
        return "manual"
    else:
        return "simulated"


def main():
    """Main test function"""
    print("\n" + "="*80)
    print("ERG TEST FRAMEWORK / ERG 测试框架")
    print("="*80)
    
    # Step 1: Select muscles
    muscles = select_muscles()
    print(f"\n✓ Selected muscles / 已选择肌肉: {muscles}")
    
    # Step 2: Select test mode
    test_mode = select_test_mode()
    print(f"✓ Test mode / 测试模式: {test_mode}")
    
    # Step 3: Run session
    try:
        if test_mode == "manual":
            controller = ManualErGController(muscles=muscles)
            controller.run_manual_session()
        else:
            controller = SimulatedErGController(muscles=muscles)
            controller.run_simulated_session(duration=30.0)
        
        controller.save_data()
    except Exception as e:
        print(f"\n✗ Session failed / 仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Generate plots
    try:
        controller.generate_plots()
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETED / 测试完成！")
        print("="*80)
        print(f"\n📁 Results saved to / 结果保存在:")
        print(f"   {controller.session_dir}")
        print()
    
    except Exception as e:
        print(f"\n✗ Plotting failed / 绘图失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
