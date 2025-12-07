#!/usr/bin/env python3
"""
Interactive MuJoCo Viewer with Real-time Signal Plotting.

交互式 MuJoCo Viewer，配合实时信号绘图（激活、力、ERG）。

这个脚本展示了：
1. 启动 MuJoCo Viewer for manual muscle control
2. 实时显示 3 条信号（激活、力、ERG）的动态图表
3. 用户可以通过滑块控制肌肉激活
4. 信号实时更新，窗口显示最近 10 秒的数据

Usage:
    python viewer_realtime_plot.py
    
    Then use matplotlib slider to control muscle activation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import mujoco
import mujoco.viewer
from threading import Thread, Lock
import time

from erg_plugin_core import ErgFilter
from erg_recorder import ErgRecorder


class RealtimeSignalPlotter:
    """Real-time signal plotting with mujoco viewer.
    
    实时信号绘图，与 MuJoCo Viewer 集成。
    """
    
    def __init__(
        self,
        model_path: str,
        muscle_name: str = "superficial_masseter_left",
        window_size: float = 10.0,  # 显示最近 10 秒
        dt: float = 0.002,
    ):
        """Initialize real-time plotter.
        
        参数 / Parameters:
            model_path: MJCF 模型路径
            muscle_name: 肌肉名称
            window_size: 时间窗口大小（秒）
            dt: 仿真时间步长
        """
        self.model_path = model_path
        self.muscle_name = muscle_name
        self.window_size = window_size
        self.dt = dt
        
        # Load model
        print(f"Loading model: {model_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model path exists: {os.path.exists(model_path)}")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            n_actuators = len(self.data.ctrl)
            print(f"✓ Model loaded: {self.model.nbody} bodies, {n_actuators} actuators")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
        
        # Get muscle actuator ID
        try:
            self.act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name
            )
            print(f"✓ Found muscle: {muscle_name} (act_id={self.act_id})")
        except Exception as e:
            print(f"✗ Muscle not found: {e}")
            raise
        
        # Initialize ERG filter
        self.erg_filter = ErgFilter()
        
        # Initialize recorder
        self.recorder = ErgRecorder(muscle_name=muscle_name, dt=dt)
        
        # Data buffers for plotting
        self.time_buffer = []
        self.activation_buffer = []
        self.force_buffer = []
        self.erg_buffer = []
        
        # Simulation state
        self.t = 0.0
        self.step_count = 0
        self.paused = False
        self.running = True
        
        # Manual activation control
        self.manual_activation = 0.0
    
    def step(self) -> None:
        """Execute one simulation step.
        
        注意: 这个方法仅由主线程调用，所以不需要额外的锁。
        """
        if self.paused:
            return
        
        # Set muscle activation from slider (thread-safe: float assignment is atomic)
        self.data.ctrl[self.act_id] = np.clip(self.manual_activation, 0.0, 1.0)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Extract signals
        activation = float(self.data.act[self.act_id])
        force = float(self.data.actuator_force[self.act_id])
        erg = self.erg_filter.step(activation, force)
        
        # Record
        self.recorder.record_step(self.t, erg, activation, force)
        
        # Add to buffers (no lock needed, only main thread accesses)
        self.time_buffer.append(self.t)
        self.activation_buffer.append(activation)
        self.force_buffer.append(force)
        self.erg_buffer.append(erg)
        
        # Trim buffers to window size
        max_samples = int(self.window_size / self.dt)
        if len(self.time_buffer) > max_samples:
            self.time_buffer.pop(0)
            self.activation_buffer.pop(0)
            self.force_buffer.pop(0)
            self.erg_buffer.pop(0)
        
        self.t += self.dt
        self.step_count += 1
        
        # Status
        if self.step_count % 500 == 0:
            print(f"Step {self.step_count} (t={self.t:.2f}s): "
                  f"act={self.manual_activation:.4f}, force={force:.4f}, erg={erg:.6f}")
    
    def get_buffer_time_offset(self) -> np.ndarray:
        """Get time array relative to current time (for plotting)."""
        if not self.time_buffer:
            return np.array([])
        # 返回相对时间，使得最新的数据点在 x=0
        current_time = self.time_buffer[-1]
        return np.array(self.time_buffer) - current_time
    


    
    def run_with_realtime_plot(self, with_viewer: bool = True) -> None:
        """Run simulation with real-time matplotlib plotting and optional MuJoCo Viewer.
        
        启动实时图表，可选启动 MuJoCo Viewer。
        简单方案：用 plt.pause() 的间隙让 Viewer 事件处理 (passive viewer)
        
        参数 / Parameters:
            with_viewer: 是否同时启动 MuJoCo Viewer
        """
        # Start MuJoCo Viewer if requested
        viewer = None
        if with_viewer:
            print("Starting MuJoCo Viewer (passive mode)...")
            # passive viewer doesn't block
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f"Real-time ERG Signals: {self.muscle_name}", 
                     fontsize=14, fontweight="bold")
        
        # Initialize line objects
        line_act, = axes[0].plot([], [], lw=2, color="orange", label="Activation")
        line_force, = axes[1].plot([], [], lw=2, color="green", label="Force")
        line_erg, = axes[2].plot([], [], lw=2, color="steelblue", label="ERG Envelope")
        
        # Setup axes
        for ax in axes:
            ax.set_xlim(-self.window_size, 0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
        
        axes[0].set_ylabel("Activation (0-1)")
        axes[0].set_ylim(0, 1.1)
        
        axes[1].set_ylabel("Force (N)")
        axes[1].set_ylim(-2, 5)
        
        axes[2].set_ylabel("ERG Envelope (0-1)")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Time (s, relative to now)")
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.96])
        
        # Add slider for manual control
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, "Activation", 0, 1, valinit=0, color='orange')
        
        def update_slider(val):
            self.manual_activation = val
        slider.on_changed(update_slider)
        
        # Enable interactive mode
        plt.ion()
        
        print("\n" + "="*70)
        print("Real-time Signal Plotter Started")
        print("="*70)
        if with_viewer:
            print("Windows:")
            print("  1. MuJoCo Viewer (3D visualization, passive)")
            print("  2. Matplotlib Plot (real-time signals)")
        print("\nControls:")
        print("  - Adjust slider to control muscle activation (0-1)")
        print("  - Close plot window to stop simulation")
        print("\nArchitecture:")
        print("  - Main thread: Drives simulation + updates plot")
        print("  - Each pause() gives Viewer a chance to render")
        print("="*70 + "\n")
        
        # Main loop - main thread drives everything
        update_interval = 0.02  # 50 Hz plot updates
        sim_steps_per_update = 10  # Run 10 sim steps (2ms each) per plot update (20ms)
        
        try:
            while self.running and plt.fignum_exists(fig.number):
                # Run multiple simulation steps
                for _ in range(sim_steps_per_update):
                    self.step()
                
                # Sync viewer if active
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                
                # Get current buffers
                time_data = self.time_buffer
                activation_data = self.activation_buffer
                force_data = self.force_buffer
                erg_data = self.erg_buffer
                
                # Calculate time offset
                if time_data:
                    current_time = time_data[-1]
                    time_offset = np.array(time_data) - current_time
                else:
                    time_offset = np.array([])
                
                # Update plot data
                line_act.set_data(time_offset, activation_data)
                line_force.set_data(time_offset, force_data)
                line_erg.set_data(time_offset, erg_data)
                
                # Auto-scale force axis
                if force_data:
                    force_min = min(force_data)
                    force_max = max(force_data)
                    margin = (force_max - force_min) * 0.1 + 0.5
                    axes[1].set_ylim(force_min - margin, force_max + margin)
                
                # Update title with current values
                if activation_data:
                    act = activation_data[-1]
                    force = force_data[-1]
                    erg = erg_data[-1]
                    viewer_status = " (+ MuJoCo Viewer)" if with_viewer else ""
                    fig.suptitle(
                        f"Real-time ERG Signals: {self.muscle_name}{viewer_status} | "
                        f"act={act:.3f} force={force:.2f}N erg={erg:.4f}",
                        fontsize=12
                    )
                
                # Draw and process events
                fig.canvas.draw_idle()
                plt.pause(update_interval)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            plt.ioff()
            plt.close(fig)
            if viewer is not None:
                viewer.close()
        
        # Stop simulation
        self.running = False
        
        # After plotting ends, save data
        print(f"\nSimulation ended at t={self.t:.2f}s (steps={self.step_count})")
        self.save_and_summarize()
    
    def run_with_mujoco_viewer(self) -> None:
        """Run simulation with MuJoCo Viewer (no real-time plot).
        
        启动 MuJoCo Viewer for manual control。
        """
        print("\n" + "="*70)
        print("MuJoCo Viewer Started")
        print("="*70)
        print("Controls:")
        print("  - Use mouse to rotate view")
        print("  - Scroll to zoom")
        print("  - Press 'H' for MuJoCo help")
        print("  - Close window to stop")
        print("="*70 + "\n")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                self.step()
                viewer.sync()
        
        print(f"\nSimulation ended at t={self.t:.2f}s (steps={self.step_count})")
        self.save_and_summarize()
    
    def save_and_summarize(self) -> None:
        """Save recordings and print summary."""
        print(f"\n{'='*70}")
        print("Simulation Summary")
        print(f"{'='*70}")
        
        stats = self.recorder.summary_stats()
        print(f"Duration: {stats['duration_s']:.2f} s")
        print(f"Samples: {stats['n_samples']}")
        print(f"Mean Activation: {stats['mean_act']:.4f}")
        print(f"Max Activation: {stats['max_act']:.4f}")
        print(f"Mean Force: {stats['mean_force']:.4f}")
        print(f"Max Force: {stats['max_force']:.4f}")
        print(f"Mean ERG: {stats['mean_erg']:.6f}")
        print(f"Max ERG: {stats['max_erg']:.6f}")
        
        # Save data
        out_dir = "erg_outputs"
        print(f"\nSaving recordings...")
        npz_file = self.recorder.save_npz(out_dir)
        csv_file = self.recorder.save_csv(out_dir)
        print(f"✓ Saved: {npz_file}")
        print(f"✓ Saved: {csv_file}")


def main():
    """Run real-time plotter."""
    # Find model path intelligently
    # 智能查找模型路径
    
    # Try multiple possible paths
    possible_paths = [
        # 从当前目录开始
        "src/exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml",
        # 从脚本所在目录的上级开始
        "../../../exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml",
        # 从项目根目录开始
        "exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml",
        # 绝对路径（在工作目录中）
        os.path.expanduser("~/exoforge-feature-sensors-user-intent/src/exoforge_description/mjcf/anatomical/jaw/jaw_model_rigid.xml"),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✓ Found model at: {path}")
            break
    
    if model_path is None:
        print("✗ Could not find model file!")
        print("Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    muscle_name = "superficial_masseter_left"
    
    try:
        # Check if we have a display for plotting
        import matplotlib
        try:
            import matplotlib.pyplot as plt
            # Try to detect if we can show plots
            has_display = os.environ.get('DISPLAY') is not None
        except:
            has_display = False
        
        plotter = RealtimeSignalPlotter(model_path, muscle_name)
        
        if has_display:
            print("\n✓ Display detected, launching real-time plotter with MuJoCo Viewer...")
            # Run with real-time matplotlib plot AND MuJoCo Viewer
            plotter.run_with_realtime_plot(with_viewer=True)
        else:
            print("\n✗ No display detected, using MuJoCo Viewer mode...")
            print("(For real-time plotting, run with DISPLAY set or use SSH X11 forwarding)")
            # Alternative: run with just MuJoCo Viewer (no real-time plot)
            plotter.run_with_mujoco_viewer()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
