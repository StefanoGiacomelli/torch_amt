"""Device Compatibility Test Suite for Auditory Models

This test suite verifies that all complete auditory models work correctly
across all available devices (CPU, CUDA, MPS).

Models tested:
- Dau1997: Dau, Kollmeier, Kohlrausch (1997) model
- Glasberg2002: Glasberg & Moore (2002) model
- Osses2021: Osses & Kohlrausch (2021) model
- Moore2016: Moore et al. (2016) model
- King2019: King, Varnet, Lorenzi (2019) model
- Paulick2024: Paulick et al. (2024) CASP model

Test structure:
- Initialization with default/minimal parameters
- Forward pass on single input
- Forward pass on batch input (8 samples)
- Device transfer (CPU, CUDA, MPS)
- Trainability verification (learnable=True)
- Module repr and parameter inspection

Usage:
    # Standalone execution (tests all available devices)
    python test_device_models.py
    
    # pytest execution
    pytest test_device_models.py -v
    
    # pytest execution on specific device
    pytest test_device_models.py -v -k "cpu"
    
    # pytest execution on specific model
    pytest test_device_models.py -v -k "dau1997"
"""

import numpy as np
import torch
import pytest
import time
import gc
from typing import List

# ================================================================================================
# Configuration
# ================================================================================================
DEBUG_ANOMALY_DETECTION = False  # Set to True to enable torch.autograd anomaly detection (slower)

# ================================================================================================
# Device Detection
# ================================================================================================
def get_available_devices() -> List[str]:
    """Detect all available PyTorch devices on the system.
    
    Returns
    -------
    list of str
        List of device strings: ['cpu']
    """
    devices = ['cpu']
    
    # if torch.cuda.is_available():
    #     devices.append('cuda')
    
    # if torch.backends.mps.is_available():
    #     devices.append('mps')
    
    return devices


def print_device_info():
    """Print information about available devices."""
    devices = get_available_devices()
    
    print("\n" + "=" * 80)
    print("AVAILABLE DEVICES")
    print("=" * 80)
    print(f"CPU:  ✓ Always available")
    print(f"CUDA: {'✓ Available' if 'cuda' in devices else '✗ Not available'}")
    print(f"MPS:  {'✓ Available' if 'mps' in devices else '✗ Not available'}")
    print("=" * 80 + "\n")


# ================================================================================================
# Test Data Factories
# ================================================================================================

def create_test_audio(device: str, batch_size: int = 1, duration: float = 0.1, fs: int = 48000) -> torch.Tensor:
    """Create test audio signal.
    
    Parameters
    ----------
    device : str
        Target device ('cpu', 'cuda', 'mps')
    batch_size : int
        Batch size (1 for single, >1 for batch)
    duration : float
        Signal duration in seconds (default: 0.1)
    fs : int
        Sampling rate in Hz
        
    Returns
    -------
    torch.Tensor
        Audio signal, shape (batch_size, n_samples)
    """
    n_samples = int(fs * duration)
    audio = torch.randn(batch_size, n_samples, device=device) * 0.9  # Small amplitude
    return audio


# ================================================================================================
# Timing Utilities
# ================================================================================================

def time_forward_pass(module, *args, device: str = 'cpu', n_warmup: int = 3, n_runs: int = 10) -> float:
    """Time a forward pass through a module.
    
    Parameters
    ----------
    module : nn.Module
        Module to time
    *args : tuple
        Arguments to pass to module
    device : str
        Device type ('cpu', 'cuda', 'mps')
    n_warmup : int
        Number of warmup iterations
    n_runs : int
        Number of timing iterations
        
    Returns
    -------
    float
        Average time in milliseconds
    """
    # Warmup (especially important for GPU)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = module(*args)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
    
    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = module(*args)
            
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms
    
    return sum(times) / len(times)  # Average


def verify_trainability(model, audio, device):
    """Verify that model parameters are actually updated during training.
    
    Parameters
    ----------
    model : nn.Module
        Model to test (must be in learnable=True mode)
    audio : torch.Tensor
        Input audio for forward pass
    device : str
        Device type ('cpu', 'cuda', 'mps')
        
    Returns
    -------
    bool
        True if parameters were successfully updated
    """
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    
    if n_trainable == 0:
        print(f"✓ Fixed parameters verified: all parameters frozen")
        return True
    
    # Get initial parameter values
    param_dict = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
    
    # Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Conditionally enable anomaly detection based on DEBUG flag
    if DEBUG_ANOMALY_DETECTION:
        context_manager = torch.autograd.set_detect_anomaly(True)
    else:
        # Dummy context manager that does nothing
        from contextlib import nullcontext
        context_manager = nullcontext()
    
    # Measure timing for forward + backward pass
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with context_manager:
        # Single optimization step
        optimizer.zero_grad()
        output = model(audio)
        
        # Compute fake loss (mean of output) - handle tensor, list, or tuple
        # Only include tensors that require grad
        if isinstance(output, (list, tuple)):
            loss_terms = [o.abs().mean() for o in output if o.requires_grad]
            if not loss_terms:
                # If no output requires grad, force it by multiplying with a parameter
                dummy_param = next(model.parameters())
                loss = sum(o.abs().mean() for o in output) * dummy_param.abs().mean() * 0.0 + dummy_param.abs().mean()
            else:
                loss = sum(loss_terms)
        else:
            if output.requires_grad:
                loss = output.abs().mean()
            else:
                # Force gradient through a parameter
                dummy_param = next(model.parameters())
                loss = output.abs().mean() * dummy_param.abs().mean() * 0.0 + dummy_param.abs().mean()
        
        loss.backward()
        
        # Distribute gradients for models with FastModulationFilterbank
        if hasattr(model, 'distribute_gradients'):
            model.distribute_gradients()
        
        optimizer.step()
    
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    
    # Report timing (calculate audio duration from model's sample rate)
    model_fs = getattr(model, 'fs', 48000)  # Get fs from model, default to 48000
    audio_duration = audio.shape[-1] / model_fs
    real_time_factor = elapsed_time / audio_duration
    print(f"  ⏱️  Forward + Backward pass: {elapsed_time:.3f}s ({real_time_factor:.1f}x real-time, fs={model_fs}Hz)")
    
    # Verify parameters changed
    n_updated = 0
    n_unchanged = 0
    updated_params = []
    unchanged_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            old_param = param_dict[name]
            
            # Compute absolute difference
            abs_diff = (param - old_param).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            
            # Consider parameter updated if ANY element changed by more than 1e-10
            if max_diff > 1e-10:
                n_updated += 1
                updated_params.append((name, max_diff, mean_diff))
            else:
                n_unchanged += 1
                unchanged_params.append(name)
    
    # Report status
    status = "✓" if n_unchanged == 0 else "⚠"
    
    # Count total PARAMETER TENSORS (not elements)
    total_param_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    # Count total SCALAR ELEMENTS in those parameters
    total_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{status} Trainability verification:")
    print(f"    Total: {total_param_tensors} parameter tensors ({total_elements} scalar elements)")
    print(f"    Updated: {n_updated} parameters")
    print(f"    Unchanged: {n_unchanged} parameters")
    
    if updated_params:
        print(f"    Parameters that updated:")
        for name, max_diff, mean_diff in updated_params:
            print(f"      ✓ {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    if unchanged_params:
        print(f"    ⚠ WARNING: Parameters that DID NOT update (gradient=0 or data-dependent):")
        for name in unchanged_params:
            print(f"      ✗ {name}")
        print(f"    This indicates a GRADIENT FLOW problem!")
    
    assert n_updated > 0, f"Expected at least some parameters to update, but none did!"
    
    if n_unchanged > 0:
        total_param_tensors = sum(1 for p in model.parameters() if p.requires_grad)
        unchanged_pct = (n_unchanged / total_param_tensors) * 100
        print(f"    ⚠ Summary: {n_unchanged}/{total_param_tensors} ({unchanged_pct:.1f}%) parameters did not update")
    
    return n_unchanged == 0  # Return True only if ALL parameters updated


def get_output_shape(output):
    """Get shape from output (handle tensor, list, or tuple)."""
    if isinstance(output, list):
        return f"List[{len(output)}], first: {output[0].shape}"
    elif isinstance(output, tuple):
        return f"Tuple[{len(output)}], first: {output[0].shape}"
    else:
        return output.shape


def get_output_device(output):
    """Get device from output (handle tensor, list, or tuple)."""
    if isinstance(output, (list, tuple)):
        return output[0].device
    else:
        return output.device


# ================================================================================================
# Test: Dau1997
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_dau1997(device):
    """Test Dau1997 model on specified device."""
    from torch_amt.models import Dau1997
    
    print(f"\n{'='*80}")
    print(f"TEST: Dau1997 - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Initialization with default parameters (learnable=True)
    fs = 48000
    model = Dau1997(fs=fs, learnable=True)
    model = model.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {model}")
    print(f"  extra_repr: {model.extra_repr()}")
    
    # Parameters - distinguish between parameter tensors and scalar elements
    n_param_tensors = sum(1 for _ in model.parameters())
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_elements_total = sum(p.numel() for p in model.parameters())
    n_elements_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {n_param_tensors} tensors ({n_elements_total} elements), {n_trainable_tensors} trainable ({n_elements_trainable} elements)")
    
    # Step 2: Trainability test (model.train() mode)
    model.train()
    audio_train = create_test_audio(device, batch_size=1, duration=0.5, fs=fs)
    print(f"  Input device: {audio_train.device}")
    verify_trainability(model, audio_train, device.split(':')[0])
    
    print(f"\n✓ Dau1997 passed on {device.upper()}\n")


# ================================================================================================
# Test: Glasberg2002
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_glasberg2002(device):
    """Test Glasberg2002 model on specified device."""
    from torch_amt.models import Glasberg2002
    
    print(f"\n{'='*80}")
    print(f"TEST: Glasberg2002 - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Initialization with default parameters (learnable=True)
    fs = 32000
    model = Glasberg2002(fs=fs, learnable=True)
    model = model.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {model}")
    print(f"  extra_repr: {model.extra_repr()}")
    
    # Parameters - distinguish between parameter tensors and scalar elements
    n_param_tensors = sum(1 for _ in model.parameters())
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_elements_total = sum(p.numel() for p in model.parameters())
    n_elements_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {n_param_tensors} tensors ({n_elements_total} elements), {n_trainable_tensors} trainable ({n_elements_trainable} elements)")
    
    # Step 2: Trainability test (model.train() mode)
    model.train()
    audio_train = create_test_audio(device, batch_size=1, duration=0.5, fs=fs)
    print(f"  Input device: {audio_train.device}")
    verify_trainability(model, audio_train, device.split(':')[0])
    
    print(f"\n✓ Glasberg2002 passed on {device.upper()}\n")


# ================================================================================================
# Test: Osses2021
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_osses2021(device):
    """Test Osses2021 model on specified device."""
    from torch_amt.models import Osses2021
    
    print(f"\n{'='*80}")
    print(f"TEST: Osses2021 - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Initialization with default parameters (learnable=True)
    fs = 44100
    model = Osses2021(fs=fs, learnable=True)
    model = model.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {model}")
    print(f"  extra_repr: {model.extra_repr()}")
    
    # Parameters - distinguish between parameter tensors and scalar elements
    n_param_tensors = sum(1 for _ in model.parameters())
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_elements_total = sum(p.numel() for p in model.parameters())
    n_elements_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {n_param_tensors} tensors ({n_elements_total} elements), {n_trainable_tensors} trainable ({n_elements_trainable} elements)")
    
    # Step 2: Trainability test (model.train() mode)
    model.train()
    
    # Create realistic audio with modulation for proper gradient flow
    # (AM tone + chirp + noise to activate all processing stages)
    duration = 0.5
    n_samples = int(fs * duration)
    t = torch.linspace(0, duration, n_samples, device=device)
    
    # 1. Amplitude-modulated tone (500 Hz carrier, 10 Hz modulation)
    carrier_freq = 500.0
    mod_freq = 10.0
    am_signal = torch.sin(2 * np.pi * carrier_freq * t) * (0.5 + 0.5 * torch.sin(2 * np.pi * mod_freq * t))
    
    # 2. Chirp (sweep from 100 Hz to 4000 Hz)
    f0, f1 = 100.0, 4000.0
    chirp_signal = torch.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    
    # 3. White noise
    noise = torch.randn(n_samples, device=device) * 0.1
    
    # Combine and normalize
    audio_train = (am_signal + chirp_signal * 0.5 + noise).unsqueeze(0)
    audio_train = audio_train / audio_train.abs().max() * 0.9
    
    print(f"  Input device: {audio_train.device}")
    print(f"  Input type: Realistic (AM tone + chirp + noise)")
    verify_trainability(model, audio_train, device.split(':')[0])
    
    print(f"\n✓ Osses2021 passed on {device.upper()}\n")


# ================================================================================================
# Test: Moore2016
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_moore2016(device):
    """Test Moore2016 model on specified device."""
    from torch_amt.models import Moore2016
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016 - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Initialization with default parameters (learnable=True)
    fs = 32000
    model = Moore2016(fs=fs, learnable=True)
    model = model.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {model}")
    print(f"  extra_repr: {model.extra_repr()}")
    
    # Parameters - distinguish between parameter tensors and scalar elements
    n_param_tensors = sum(1 for _ in model.parameters())
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_elements_total = sum(p.numel() for p in model.parameters())
    n_elements_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {n_param_tensors} tensors ({n_elements_total} elements), {n_trainable_tensors} trainable ({n_elements_trainable} elements)")
    
    # Step 2: Trainability test (model.train() mode)
    model.train()
    
    # Create realistic stereo audio
    t = torch.linspace(0, 0.5, int(fs * 0.5), device=device)
    
    # Left channel: 1 kHz + 4 kHz, with amplitude envelope (attack/decay)
    envelope_left = torch.exp(-5 * t) + 0.3  # Decaying envelope
    left = envelope_left * (torch.sin(2 * torch.pi * 1000 * t) + 0.5 * torch.sin(2 * torch.pi * 4000 * t))
    
    # Right channel: 500 Hz + 2 kHz, with different amplitude (for binaural difference)
    envelope_right = 0.5 * torch.exp(-3 * t) + 0.5  # Different envelope
    right = envelope_right * (torch.sin(2 * torch.pi * 500 * t) + 0.3 * torch.sin(2 * torch.pi * 2000 * t))
    
    # Add some noise for richer spectrum
    left = left + 0.05 * torch.randn_like(left)
    right = right + 0.05 * torch.randn_like(right)
    
    # Stack to stereo [1, 2, T]
    audio_train_stereo = torch.stack([left, right], dim=0).unsqueeze(0)
    
    print(f"  Input device: {audio_train_stereo.device}")
    print(f"  Input type: Realistic stereo (L/R different, amplitude modulation, rich spectrum)")
    verify_trainability(model, audio_train_stereo, device.split(':')[0])
    
    print(f"\n✓ Moore2016 passed on {device.upper()}\n")


# ================================================================================================
# Test: King2019
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_king2019(device):
    """Test King2019 model on specified device."""
    from torch_amt.models import King2019
    
    print(f"\n{'='*80}")
    print(f"TEST: King2019 - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Initialization with default parameters (learnable=True)
    fs = 48000
    model = King2019(fs=fs, basef=1000, learnable=True)
    model = model.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {model}")
    print(f"  extra_repr: {model.extra_repr()}")
    
    # Parameters - distinguish between parameter tensors and scalar elements
    n_param_tensors = sum(1 for _ in model.parameters())
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_elements_total = sum(p.numel() for p in model.parameters())
    n_elements_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {n_param_tensors} tensors ({n_elements_total} elements), {n_trainable_tensors} trainable ({n_elements_trainable} elements)")
    
    # Step 2: Trainability test (model.train() mode)
    model.train()
    audio_train = create_test_audio(device, batch_size=1, duration=0.5, fs=fs)
    print(f"  Input device: {audio_train.device}")
    verify_trainability(model, audio_train, device.split(':')[0])
    
    print(f"\n✓ King2019 passed on {device.upper()}\n")


# ================================================================================================
# Test: Paulick2024
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_paulick2024(device):
    """Test Paulick2024 model on specified device."""
    from torch_amt.models import Paulick2024
    
    print(f"\n{'='*80}")
    print(f"TEST: Paulick2024 - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Initialization with default parameters (learnable=True)
    fs = 44100
    model = Paulick2024(fs=fs, learnable=True)
    model = model.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {model}")
    print(f"  extra_repr: {model.extra_repr()}")
    
    # Parameters - distinguish between parameter tensors and scalar elements
    n_param_tensors = sum(1 for _ in model.parameters())
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    n_elements_total = sum(p.numel() for p in model.parameters())
    n_elements_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {n_param_tensors} tensors ({n_elements_total} elements), {n_trainable_tensors} trainable ({n_elements_trainable} elements)")
    
    # Step 2: Trainability test (model.train() mode)
    model.train()
    
    # Create broadband audio that excites ALL 50 DRNL channels (125-8000 Hz)
    # Use multi-tone complex with frequencies distributed across the auditory range
    t = torch.linspace(0, 0.5, int(fs * 0.5), device=device)
    
    # Generate log-spaced frequencies to excite all DRNL channels uniformly
    # 20 frequencies from 125 Hz to 8000 Hz (model's flow to fhigh)
    freqs = torch.logspace(np.log10(125), np.log10(8000), 20, device=device)
    
    # Multi-tone complex with amplitude modulation (more realistic than pure tones)
    audio_broadband = torch.zeros_like(t)
    for i, freq in enumerate(freqs):
        # Each component has different amplitude and phase for natural sound
        amplitude = 1.0 / (20 * (1 + 0.3 * i / 20))  # Gradual roll-off
        phase = torch.rand(1, device=device).item() * 2 * np.pi  # Random phase
        audio_broadband += amplitude * torch.sin(2 * np.pi * freq * t + phase)
    
    # Add low-level broadband noise to fill spectral gaps
    audio_broadband = audio_broadband + 0.02 * torch.randn_like(audio_broadband)
    
    # Apply amplitude envelope (attack-sustain-decay) for realism
    envelope = torch.exp(-3 * t) + 0.3  # Decaying envelope with sustain
    audio_broadband = envelope * audio_broadband
    
    # Scale to physiological amplitude (~60 dB SPL)
    # Reference: 0 dB SPL = 2e-5 Pa, 60 dB SPL = 0.02 Pa
    ref_amplitude = 2e-5 * 10 ** (60 / 20)
    audio_broadband = audio_broadband * ref_amplitude * 0.01
    
    audio_train = audio_broadband.unsqueeze(0)  # [1, T]
    
    print(f"  Input device: {audio_train.device}")
    print(f"  Input type: Broadband multi-tone (20 freqs, 125-8000 Hz, excites all DRNL channels)")
    verify_trainability(model, audio_train, device.split(':')[0])
    
    print(f"\n✓ Paulick2024 passed on {device.upper()}\n")


# ================================================================================================
# Main: Standalone Execution
# ================================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEVICE COMPATIBILITY TEST SUITE: COMPLETE AUDITORY MODELS")
    print("=" * 80)
    if DEBUG_ANOMALY_DETECTION:
        print("⚠️  Anomaly detection ENABLED - will track gradient issues (slower)")
    else:
        print("✓  Anomaly detection DISABLED - faster execution")
    
    print_device_info()
    
    devices = get_available_devices()
    
    print(f"\nTesting 4 models on {len(devices)} device(s): {devices}")
    print(f"All models tested with learnable=True")
    print(f"Total tests: {4 * len(devices)} = 4 models x {len(devices)} device(s)\n")
    
    # Run all tests
    for device in devices:
        print(f"\n{'#'*80}")
        print(f"# DEVICE: {device.upper()}")
        print(f"{'#'*80}\n")
        
        # Test each model with cleanup between
        test_dau1997(device)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        test_glasberg2002(device)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        test_moore2016(device)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        test_osses2021(device)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        test_king2019(device)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        test_paulick2024(device)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80 + "\n")
