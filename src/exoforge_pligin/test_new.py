"""ERG Manual Testing with MuJoCo Viewer - Direct Configuration

ERG manual testing framework / ERG manual testing framework

Workflow:
1. Select muscle directly in code
2. Open MuJoCo Viewer
3. Manually control muscle activation (via sliders)
4. Real-time record activation, force, ERG signal
5. After closing window, auto save data and plot

Usage:
    1. Modify SELECTED_MUSCLES configuration below
    2. Run: python test_new.py
    3. In MuJoCo Viewer, manually control muscle activation (use sliders)
    4. Close Viewer window (ESC or click close button)
    5. Data auto saves to erg_test/ folder
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

# Support both package and direct execution
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .erg_plugin_core import ErgFilter, GaussianNoiseGenerator, ColoredNoiseGenerator
    from .erg_recorder import MultiMuscleRecorder, ErgRecorder
    from .erg_visualization import plot_erg_signal, plot_multi_erg
except ImportError:
    from erg_plugin_core import ErgFilter, GaussianNoiseGenerator, ColoredNoiseGenerator
    from erg_recorder import MultiMuscleRecorder, ErgRecorder
    from erg_visualization import plot_erg_signal, plot_multi_erg

import mujoco
from mujoco import viewer

# CONFIGURATION SECTION - Modify here!
# ============================================================================

# Select muscles to test
SELECTED_MUSCLES = [
    "superficial_masseter_left",
    "superficial_masseter_right",
    # "anterior_temporalis_left",
    # "anterior_temporalis_right",
]

# Available muscles
AVAILABLE_MUSCLES = [
    "superficial_masseter_left",
    "superficial_masseter_right",
    "anterior_temporalis_left",
    "anterior_temporalis_right",
    "posterior_temporalis_left",
    "posterior_temporalis_right",
]

# Select noise generator
# Options:
#   - GaussianNoiseGenerator(std_dev=0.05)  # Gaussian white noise
#   - ColoredNoiseGenerator(std_dev=0.05, alpha=1.0)  # Pink noise 1/f
#   - None  # No noise (clean signal)

NOISE_GENERATOR = GaussianNoiseGenerator(std_dev=0.05)  # Medium noise

# Filter parameters
ERG_FILTER_CONFIG = {
    "k": 3.0,           # tanh compression factor (1.0-5.0)
    "alpha": 0.3,       # force contribution weight (0.0-1.0)
    "noise_std": 0.05,  # backup noise (used if noise_generator=None)
    "a_hp": 0.995,      # high-pass coefficient (0.9-0.999)
    "a_lp": 0.75,       # low-pass coefficient (0.5-0.99)
    "fmax": 1.0,        # max force for normalization
    "noise_generator": NOISE_GENERATOR,  # pluggable noise generator
}

# ============================================================================


class ManualTestController:
    """Manual testing controller with MuJoCo Viewer"""
    
    def __init__(self, muscles: List[str], output_dir: str = "erg_test"):
        self.muscles = muscles
        self.output_dir = output_dir
        self.session_dir = self._create_session_dir()
        
        print(f"Session directory: {self.session_dir}")
        print(f"Selected muscles: {', '.join(muscles)}")
        
        # Load MuJoCo model
        self.model, self.data = self._load_model()
        print(f"MuJoCo model loaded")
        
        # Initialize ERG filters with custom configuration
        self.erg_filters = {
            muscle: ErgFilter(**ERG_FILTER_CONFIG) 
            for muscle in muscles
        }
        self.recorder = MultiMuscleRecorder(muscle_names=muscles)
        print(f"Initialized {len(muscles)} muscle(s)")
        print(f"  ERG Filter Config:")
        print(f"    - noise_generator: {type(NOISE_GENERATOR).__name__}")
        
        # Get muscle actuator IDs (for correct activation/force retrieval)
        self.muscle_act_ids = {}
        for muscle in muscles:
            try:
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle
                )
                self.muscle_act_ids[muscle] = act_id
                print(f"  - {muscle} (act_id={act_id})")
            except Exception as e:
                print(f"  ! {muscle} not found: {e}")
        
        self.dt = self.model.opt.timestep
        self.step_count = 0
        
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
    
    def _get_muscle_activation(self, muscle_name: str) -> float:
        """
        Get muscle activation from MuJoCo data by muscle name
        
        Get correct activation value from MuJoCo data based on muscle name
        """
        if muscle_name not in self.muscle_act_ids:
            return 0.0
        
        try:
            act_id = self.muscle_act_ids[muscle_name]
            # Get from act array (the actual muscle activation)
            activation = float(self.data.act[act_id])
            return np.clip(activation, 0.0, 1.0)
        except:
            pass
        
        return 0.0
    
    def _get_muscle_force(self, muscle_name: str) -> float:
        """
        Get muscle force from MuJoCo data by muscle name
        
        Get correct force value from MuJoCo data based on muscle name
        """
        if muscle_name not in self.muscle_act_ids:
            return 0.0
        
        try:
            act_id = self.muscle_act_ids[muscle_name]
            # Get from actuator_force array
            force = float(self.data.actuator_force[act_id])
            return force
        except:
            pass
        
        return 0.0
    
    def run_manual_session(self):
        """Run manual session with MuJoCo viewer"""
        print(f"\n" + "="*80)
        print("MANUAL TESTING SESSION")
        print("="*80)
        print(f"\nInstructions:")
        print(f"1. MuJoCo Viewer window is open")
        print(f"2. Use sliders to control muscle activation")
        print(f"3. Data will be recorded in real-time")
        print(f"4. Close window (ESC or click close button) to end test")
        print(f"\nStarting to record...")
        print()
        
        # Launch passive viewer
        with viewer.launch_passive(self.model, self.data) as v:
            while v.is_running():
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Get muscle data
                muscle_data = {}
                for muscle in self.muscles:
                    # Get activation and force from MuJoCo
                    activation = self._get_muscle_activation(muscle)
                    force = self._get_muscle_force(muscle)
                    
                    # Process through ERG filter
                    erg_signal = self.erg_filters[muscle].step(activation, force)
                    muscle_data[muscle] = (erg_signal, activation, force)
                
                # Record data
                self.recorder.record_step(t=self.data.time, muscle_data=muscle_data)
                
                self.step_count += 1
                if self.step_count % 250 == 0:
                    elapsed = self.data.time
                    print(f"  Recording: {elapsed:.1f}s ({self.step_count} steps)")
                
                # Sync viewer
                v.sync()
        
        print(f"\nManual session completed")
        print(f"  Total time: {self.data.time:.2f} seconds")
        print(f"  Total steps: {self.step_count}")
        print(f"  Data points: {len(self.recorder.recorders[self.muscles[0]].time_array)}")
    
    def save_data(self):
        """Save all data to NPZ and CSV"""
        print(f"\nSaving data...")
        files = self.recorder.save_all(out_dir=self.session_dir)
        
        for muscle, file_info in files.items():
            print(f"  - {muscle}:")
            print(f"    - NPZ: {file_info['npz']}")
            print(f"    - CSV: {file_info['csv']}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        print(f"\nGenerating plots...")
        
        plots_dir = os.path.join(self.session_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        for muscle in self.recorder.recorders.keys():
            muscle_recorder = self.recorder.recorders[muscle]
            print(f"\n  Plotting: {muscle}")
            
            # Plot 1: ERG signal with 3 panels
            try:
                plot_erg_signal(
                    muscle_recorder,
                    window=None,
                    save_path=os.path.join(plots_dir, f"{muscle}_erg_signal.png"),
                )
                print(f"    - ERG signal plot saved")
            except Exception as e:
                print(f"    ! ERG signal plot failed: {e}")
            
            # Plot 2: Time series
            try:
                data = muscle_recorder.to_numpy()
                t = data["time_array"]
                erg = data["erg_signal"]
                act = data["activation"]
                force = np.abs(data["force"])
                
                fig, axes = plt.subplots(3, 1, figsize=(14, 9))
                fig.suptitle(f"Manual Test - {muscle}", fontsize=14, fontweight="bold")
                
                # Activation
                axes[0].plot(t, act, color="orange", linewidth=1.5, label="Activation")
                axes[0].fill_between(t, act, alpha=0.3, color="orange")
                axes[0].set_ylabel("Activation", fontsize=11)
                axes[0].set_title("Muscle Activation (User Input)")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="upper right")
                axes[0].set_ylim([0, 1])
                
                # Force
                axes[1].plot(t, force, color="green", linewidth=1.5, label="Force")
                axes[1].fill_between(t, force, alpha=0.3, color="green")
                axes[1].set_ylabel("Force (N)", fontsize=11)
                axes[1].set_title("Contraction Force (Physical Output)")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc="upper right")
                
                # ERG Signal
                axes[2].plot(t, erg, color="steelblue", linewidth=1, label="ERG Signal", alpha=0.8)
                axes[2].fill_between(t, erg, alpha=0.3, color="steelblue")
                axes[2].set_ylabel("ERG (uV)", fontsize=11)
                axes[2].set_xlabel("Time (s)", fontsize=11)
                axes[2].set_title("Processed ERG Envelope")
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc="upper right")
                
                plt.tight_layout()
                save_path = os.path.join(plots_dir, f"{muscle}_time_series.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    - Time series plot saved")
            except Exception as e:
                print(f"    ! Time series plot failed: {e}")
        
        # Multi-muscle comparison if available
        if len(self.recorder.recorders) > 1:
            try:
                plot_multi_erg(
                    self.recorder.recorders,
                    save_path=os.path.join(plots_dir, "multi_muscle_comparison.png"),
                )
                print(f"\n  - Multi-muscle comparison plot saved")
            except Exception as e:
                print(f"\n  ! Comparison plot failed: {e}")
        
        print(f"\nAll plots saved to: {plots_dir}")


def main():
    """Main test function"""
    # Validate selected muscles
    invalid_muscles = [m for m in SELECTED_MUSCLES if m not in AVAILABLE_MUSCLES]
    if invalid_muscles:
        print(f"Invalid muscles: {invalid_muscles}")
        print(f"   Available: {', '.join(AVAILABLE_MUSCLES)}")
        return
    
    if not SELECTED_MUSCLES:
        print(f"No muscles selected!")
        return
    
    print("\n" + "="*80)
    print("ERG MANUAL TEST")
    print("="*80)
    
    try:
        # Create controller
        controller = ManualTestController(muscles=SELECTED_MUSCLES)
        
        # Run manual session (opens MuJoCo Viewer)
        print(f"\nOpening MuJoCo Viewer in 1 second...")
        import time
        time.sleep(1)
        
        controller.run_manual_session()
        
        # Save data
        controller.save_data()
        
        # Generate plots
        controller.generate_plots()
        
        # Summary
        print("\n" + "="*80)
        print("TEST COMPLETED!")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"   {controller.session_dir}")
        print(f"\nOutput files:")
        print(f"   - *.npz (compressed data)")
        print(f"   - *.csv (text data)")
        print(f"   - plots/*.png (visualizations)")
        print()
        
    except KeyboardInterrupt:
        print(f"\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
