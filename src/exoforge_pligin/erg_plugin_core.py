"""ERG signal processing core module (independent, no external dependencies).

Core ERG signal processing module - completely independent, no external dependencies.
Contains only ErgFilter class and basic utility functions.

Signal Processing Pipeline: User Action -> ERG Signal
======================================================================

Input Layer: User Actions from MuJoCo
User controls muscle via MuJoCo Viewer
activation in [0.0, 1.0]  <- User intention
force in R (from physics engine)  <- Physics feedback

5-Stage Processing Pipeline
======================================================================

STAGE 1: Signal Fusion
s(t) = activation + alpha * |force| / fmax

- activation: User intention (0.0-1.0)
- |force|/fmax: Force feedback (0.0-1.0)
- alpha: Weight coefficient (default 0.2, adjustable)

Output: s in [0.0, 2.0]
Result: Fused signal of intention + physical feedback

STAGE 2: Nonlinearity + Noise
snl(t) = tanh(k*s) + eta

- tanh(k*s): Hyperbolic tangent compression
  - k: Compression factor (default 3.0)
  - tanh maps any value to [-1, 1]
  - Simulates muscle saturation property

- eta: Noise (pluggable)
  - GaussianNoiseGenerator: Gaussian white noise
  - ColoredNoiseGenerator: Pink noise 1/f
  - NoNoiseGenerator: No noise
  - Simulates real biosignal characteristics

Output: snl in [-1.2, 1.2] (with noise)
Result: Nonlinear, noisy source signal (Realistic biosignal)

STAGE 3: High-Pass Filter (DC Removal)
y_hp(t) = a_hp * (y_hp(t-1) + snl(t) - snl(t-1))

- a_hp: High-pass coefficient (default 0.995, very strong)
- (snl(t) - snl(t-1)): Differential term, calculates signal change

Key properties:
- Removes DC component (baseline) -> only retains AC component
- If signal is constant -> high-pass output is 0
- Fast changes -> high-pass output increases

Output: y_hp in [-1, 1]
Result: Dynamic signal without DC bias (AC component only)

STAGE 4: Full-Wave Rectification
y_rect = |y_hp|

- Take absolute value -> positive and negative signals both become positive
- Only care about magnitude, not direction

Why: EMG signal oscillates, but we only care about "strength"

Output: y_rect >= 0
Result: Non-negative strength signal (magnitude only)

STAGE 5: Low-Pass Filter (Envelope Extraction)
y_erg = a_lp * y_erg(t-1) + (1 - a_lp) * y_rect(t)

- a_lp: Low-pass coefficient (default 0.90)
- Exponential Moving Average (EMA)

Coefficient breakdown:
- a_lp * y_erg(t-1): 90% from old value (smooth)
- (1-a_lp) * y_rect(t): 10% from new value (responsive)

Time constant: tau ~= 19ms (decay to 63%)
Frequency: Smooths >50Hz fast changes, preserves <10Hz slow changes

Output: y_erg in [0.0, 1.0] (normalized)
Result: Smooth muscle strength envelope

Output Layer: ERG Signal - Ready for Intent Recognition
======================================================================
Final ERG envelope signal

Range: [0.0, 1.0]
Characteristics: Smooth, low-noise, represents muscle strength

Usage:
- Muscle intent recognition
- Control signal generation
- Biofeedback
- Data logging

Key Parameters
======================================================================

Source Signal Fusion:
alpha (0.0-1.0, default 0.2)
  - Controls force impact on source signal
  - alpha=0.0 -> only activation, ignore force
  - alpha=0.5 -> activation and force each 50%
  - alpha=1.0 -> activation and force equally important

Nonlinear Compression:
k (1.0-5.0, default 3.0)
  - Controls tanh "steepness"
  - Small k -> more linear
  - Large k -> more nonlinear
  - Default 3.0 good for medium nonlinearity

High-Pass Filter:
a_hp (0.9-0.999, default 0.995)
  - Controls DC removal strength
  - Close to 1.0 -> stronger high-pass (cleaner)
  - Close to 0.9 -> weaker high-pass (preserves more low freq)
  - Default 0.995 very strong, nearly complete DC removal

Low-Pass Filter:
a_lp (0.5-0.99, default 0.90)
  - Controls envelope smoothness
  - a_lp=0.99 -> super smooth, slow response (~200ms lag)
  - a_lp=0.75 -> medium smooth, faster (~80ms lag)
  - a_lp=0.50 -> fast response, not smooth (~4ms lag)
  - Default 0.90 balances smoothness and response

Noise:
noise_generator (pluggable)
  - GaussianNoiseGenerator(0.02): Low noise, clean signal
  - GaussianNoiseGenerator(0.05): Medium noise, close to real
  - ColoredNoiseGenerator(0.05, 1.0): Pink noise, most realistic
  - NoNoiseGenerator(): No noise, for testing

Real-Time Data Flow Example
======================================================================

t=0ms: User relaxed
  activation = 0.0, force = 0.0
  -> s = 0.0
  -> snl ~= 0 + eta
  -> y_hp ~= 0 (high-pass filters DC)
  -> y_rect ~= 0
  -> y_erg ~= 0 (no muscle activity)

t=100ms: User starts contraction
  activation = 0.5, force = 2.0
  -> s = 0.5 + 0.2*2.0/1.0 = 0.9
  -> snl = tanh(3*0.9) + eta ~= 0.72 + eta
  -> y_hp increases (signal changes)
  -> y_rect increases (rectified)
  -> y_erg increases (smoothed) (muscle starts activity)

t=500ms: User sustained contraction
  activation = 0.8, force = 4.0
  -> s = 0.8 + 0.2*4.0/1.0 = 1.6
  -> snl = tanh(3*1.6) + eta ~= 0.99 + eta (approaches saturation)
  -> y_hp ~= 0 (little signal change)
  -> y_erg stable (sustained muscle contraction)

t=1000ms: User relaxes
  activation = 0.0, force = 0.1
  -> s = 0.0 + 0.2*0.1/1.0 = 0.02
  -> snl ~= 0.02 (signal drops)
  -> y_hp decreases (clear drop)
  -> y_rect decreases (drops)
  -> y_erg decays slowly (muscle relaxing)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Pluggable Noise Modules
# ======================================================================

class NoiseGenerator:
    """Base class for noise generation"""
    
    def generate(self) -> float:
        """Generate noise value"""
        return 0.0
    
    def reset(self) -> None:
        """Reset state if needed"""
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    """Gaussian (white noise) generator
    
    Generates Gaussian white noise to simulate real EMG random fluctuations
    """
    
    def __init__(self, std_dev: float = 0.02):
        """
        Parameters:
            std_dev: standard deviation
                - 0.00: no noise
                - 0.02: low noise (default, clean signal)
                - 0.05: medium noise
                - 0.10: high noise (realistic EMG)
        """
        self.std_dev = std_dev
    
    def generate(self) -> float:
        """Generate Gaussian noise"""
        if self.std_dev <= 0:
            return 0.0
        return float(np.random.normal(0.0, self.std_dev))


class ColoredNoiseGenerator(NoiseGenerator):
    """Colored (1/f) noise generator
    
    Generates 1/f noise (pink noise), closer to real biosignal frequency spectrum
    """
    
    def __init__(self, std_dev: float = 0.02, alpha: float = 1.0):
        """
        Parameters:
            std_dev: standard deviation
            alpha: spectral exponent
                - 0.0: white noise (flat spectrum)
                - 1.0: pink noise 1/f
                - 2.0: brown noise 1/f^2
        """
        self.std_dev = std_dev
        self.alpha = alpha
        self.prev_value = 0.0
    
    def generate(self) -> float:
        """Generate colored noise using simple low-pass filter"""
        if self.std_dev <= 0:
            return 0.0
        
        # Simple first-order filter for 1/f spectrum
        white = np.random.normal(0.0, self.std_dev)
        # a_color: controls noise "color", larger = smoother
        a_color = self.alpha / 3.0  # Normalize to [0, ~0.33]
        self.prev_value = a_color * self.prev_value + (1 - a_color) * white
        return self.prev_value
    
    def reset(self) -> None:
        """Reset internal state"""
        self.prev_value = 0.0


class NoNoiseGenerator(NoiseGenerator):
    """No noise (for clean signal testing)"""
    
    def generate(self) -> float:
        return 0.0


# Configuration
# ======================================================================

@dataclass
class ErgConfig:
    """ERG filter configuration
    
    Parameter description:
    - k: tanh compression factor (default: 3.0)
    - alpha: force contribution factor (default: 0.2)
    - noise_std: Gaussian noise std dev (default: 0.02)
    - a_hp: high-pass IIR coefficient (default: 0.995)
    - a_lp: low-pass IIR coefficient (default: 0.90)
    - fmax: max force estimate for normalization (default: 1.0)
    - noise_type: type of noise generator ('gaussian', 'colored', 'none') (default: 'gaussian')
    """
    k: float = 3.0
    alpha: float = 0.2
    noise_std: float = 0.02
    a_hp: float = 0.995
    a_lp: float = 0.90
    fmax: float = 1.0
    noise_type: str = 'gaussian'  # 'gaussian', 'colored', or 'none'


# Main Filter Class
# ======================================================================

class ErgFilter:
    """Single-channel ERG envelope with simple HP + rect + LP cascade.

    5-stage ERG envelope filter: high-pass -> rectify -> low-pass
    """

    def __init__(
        self,
        k: float = 3.0,
        alpha: float = 0.2,
        noise_std: float = 0.02,
        a_hp: float = 0.995,
        a_lp: float = 0.90,
        fmax: float = 1.0,
        noise_generator: Optional[NoiseGenerator] = None,
    ) -> None:
        """Initialize ERG filter with processing parameters and pluggable noise generator.
        
        Parameters:
        
        k (float): tanh compression factor
            - Range: 1.0-5.0
            - Effect: Controls nonlinearity degree
            - Higher k -> stronger nonlinear compression
            - Default: 3.0 (medium nonlinearity)
        
        alpha (float): force contribution weight to source signal
            - Range: 0.0-1.0
            - Effect: How much force feedback is mixed in
            - alpha=0.0: only activation, ignore force
            - alpha=1.0: activation and force equally important
            - Default: 0.2 (small force contribution)
        
        noise_std (float): noise standard deviation
            - Range: 0.0-0.1
            - Effect: How much "biological noise" to add
            - Higher -> noisier signal
            - Default: 0.02 (moderate noise)
        
        a_hp (float): high-pass IIR coefficient
            - Range: 0.9-0.999
            - Effect: Controls high-pass filter strength
            - Close to 1.0 -> stronger high-pass (cleaner)
            - Default: 0.995 (very strong)
        
        a_lp (float): low-pass IIR coefficient
            - Range: 0.5-0.99
            - Effect: Controls envelope smoothness
            - 0.99 -> very smooth but slow response
            - 0.50 -> fast response but not smooth
            - Default: 0.90 (balanced)
        
        fmax (float): max force for normalization
            - Range: >0
            - Effect: Normalizes force to [0, 1]
            - Default: 1.0
        
        noise_generator (Optional[NoiseGenerator]): pluggable noise generator
            - None (default): uses GaussianNoiseGenerator(noise_std)
            - GaussianNoiseGenerator(0.05): Gaussian noise
            - ColoredNoiseGenerator(0.05, alpha=1.0): Pink noise (1/f)
            - NoNoiseGenerator(): No noise (for testing)
            
            Example:
                # Use Gaussian noise with std 0.05
                filter1 = ErgFilter(noise_std=0.05)
                
                # Use pink noise
                filter2 = ErgFilter(
                    noise_generator=ColoredNoiseGenerator(std_dev=0.05, alpha=1.0)
                )
                
                # No noise
                filter3 = ErgFilter(noise_generator=NoNoiseGenerator())
        """
        self.k = k
        self.alpha = alpha
        self.noise_std = noise_std
        self.a_hp = a_hp
        self.a_lp = a_lp
        self.fmax = max(fmax, 1e-6)  # Prevent division by zero
        
        # Initialize noise generator
        if noise_generator is None:
            self.noise_generator = GaussianNoiseGenerator(std_dev=noise_std)
        else:
            self.noise_generator = noise_generator
        
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state to zero"""
        self.prev_snl = 0.0
        self.y_hp = 0.0
        self.y_erg = 0.0
        if hasattr(self.noise_generator, 'reset'):
            self.noise_generator.reset()

    def step(self, activation: float, force: float) -> float:
        """Process one timestep and return ERG envelope value
        
        Inputs:
            activation (float): Muscle activation value (0.0-1.0)
                From user's manual control in MuJoCo Viewer
                Represents user's intention strength
            
            force (float): Actual force produced by actuator
                From MuJoCo physics simulation feedback
                Positive = muscle contraction, negative = extension
        
        Output:
            float: ERG envelope value (0.0-1.0)
                Smooth signal representing muscle strength
                For intent recognition, control signals, etc.
        """
        # STAGE 1: Fuse source signal
        # s(t) = activation + alpha * |force| / fmax
        s = activation + self.alpha * abs(force) / self.fmax

        # STAGE 2: Nonlinearity + Noise
        # snl(t) = tanh(k*s) + eta
        snl = math.tanh(self.k * s) + self.noise_generator.generate()

        # STAGE 3: High-pass filter (DC removal)
        # y_hp(t) = a_hp * (y_hp(t-1) + snl(t) - snl(t-1))
        self.y_hp = self.a_hp * (self.y_hp + snl - self.prev_snl)

        # STAGE 4: Full-wave rectification
        # y_rect = |y_hp|
        y_rect = abs(self.y_hp)

        # STAGE 5: Low-pass filter for envelope extraction
        # y_erg = a_lp * y_erg(t-1) + (1 - a_lp) * y_rect(t)
        self.y_erg = self.a_lp * self.y_erg + (1.0 - self.a_lp) * y_rect

        # Update internal state for next step
        self.prev_snl = snl
        
        return self.y_erg

    def get_state(self) -> dict:
        """Get current filter state"""
        return {
            "prev_snl": self.prev_snl,
            "y_hp": self.y_hp,
            "y_erg": self.y_erg,
        }

    def reset(self) -> None:
        """Reset filter to initial state"""
        self._reset_state()


def build_erg_processor(
    act_id: int,
    alpha: float = 0.2,
    k: float = 3.0,
    noise_std: float = 0.02,
    a_hp: float = 0.995,
    a_lp: float = 0.90,
    fmax: Optional[float] = None,
) -> Tuple[int, ErgFilter]:
    """Create ERG filter for a given actuator ID.
    
    Creates ERG filter without dependence on MuJoCo model.
    """
    filt = ErgFilter(
        k=k,
        alpha=alpha,
        noise_std=noise_std,
        a_hp=a_hp,
        a_lp=a_lp,
        fmax=fmax or 1.0,
    )
    return act_id, filt
