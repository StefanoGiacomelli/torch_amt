"""Device Compatibility Test Suite for adaptation.py

This test suite verifies that adaptation loops in adaptation.py work correctly
across all available devices (CPU, CUDA, MPS).

Contents:
- 1 nn.Module class: AdaptLoop

Test structure:
- Initialization with default/preset parameters
- Application on single input
- Application on batch input
- Device transfer (CPU, CUDA, MPS)
- Module repr and parameter inspection
- Timing measurements
- Learnable parameter verification
- Preset configurations test
- Shape handling: (F, T) and (B, F, T)

Usage:
    # Standalone execution (tests all available devices)
    python test_device_adaptation.py
    
    # pytest execution
    pytest test_device_adaptation.py -v
    
    # pytest execution on specific device
    pytest test_device_adaptation.py -v -k "cpu"
"""

import torch
import pytest
import time
from typing import List


# ================================================================================================
# Device Detection
# ================================================================================================

def get_available_devices() -> List[str]:
    """Detect all available PyTorch devices on the system.
    
    Returns
    -------
    list of str
        List of device strings: ['cpu'], ['cpu', 'cuda'], or ['cpu', 'mps']
    """
    devices = ['cpu']
    
    if torch.cuda.is_available():
        devices.append('cuda')
    
    if torch.backends.mps.is_available():
        devices.append('mps')
    
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

def create_test_signal(device: str, batch_size: int = 1, n_channels: int = 31, 
                       duration: float = 0.1, fs: int = 16000) -> torch.Tensor:
    """Create test filterbank output signal.
    
    Parameters
    ----------
    device : str
        Target device ('cpu', 'cuda', 'mps')
    batch_size : int
        Batch size (1 for single, >1 for batch)
    n_channels : int
        Number of frequency channels
    duration : float
        Signal duration in seconds
    fs : int
        Sampling rate in Hz
        
    Returns
    -------
    torch.Tensor
        Filterbank output signal, shape (batch_size, n_channels, n_samples)
    """
    n_samples = int(fs * duration)
    # Simulate filterbank output (positive values with some dynamic range)
    signal = torch.randn(batch_size, n_channels, n_samples, device=device).abs() * 0.01 + 0.001
    return signal


def time_forward_pass(module, *args, device='cpu', n_warmup=2, n_runs=5):
    """Time a forward pass through a module.
    
    Parameters
    ----------
    module : nn.Module
        Module to benchmark
    *args : torch.Tensor
        Input tensors for forward pass
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


# ================================================================================================
# Test: AdaptLoop - Default Parameters
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_adaptloop_default(device, learnable):
    """Test AdaptLoop with default parameters on specified device."""
    from torch_amt.common.adaptation import AdaptLoop
    
    print(f"\n{'='*80}")
    print(f"TEST: AdaptLoop (default) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters
    fs = 16000
    adapt = AdaptLoop(fs=fs, learnable=learnable)
    adapt = adapt.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {adapt}")
    print(f"  extra_repr: {adapt.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in adapt.parameters())
    param_names = [name for name, _ in adapt.named_parameters()]
    print(f"  Parameters: {n_params} total, names={param_names}")
    
    # Forward pass - single (31 channels, 0.1 sec)
    signal_single = create_test_signal(device, batch_size=1, n_channels=31, duration=0.1, fs=fs)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(adapt, signal_single, device=device.split(':')[0])
    output_single = adapt(signal_single)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward single: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    signal_batch = create_test_signal(device, batch_size=4, n_channels=31, duration=0.1, fs=fs)
    avg_time_batch = time_forward_pass(adapt, signal_batch, device=device.split(':')[0])
    output_batch = adapt(signal_batch)
    assert output_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  {signal_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    # Test shape handling: (F, T) input
    signal_2d = create_test_signal(device, batch_size=1, n_channels=31, duration=0.05, fs=fs).squeeze(0)
    output_2d = adapt(signal_2d)
    assert output_2d.shape == signal_2d.shape
    print(f"✓ Forward 2D:     {signal_2d.shape} -> {output_2d.shape}")
    
    print(f"\n✓ AdaptLoop (default) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: AdaptLoop - Preset dau1997
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_adaptloop_dau1997(device, learnable):
    """Test AdaptLoop with dau1997 preset on specified device."""
    from torch_amt.common.adaptation import AdaptLoop
    
    print(f"\n{'='*80}")
    print(f"TEST: AdaptLoop (dau1997) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with dau1997 preset
    fs = 16000
    adapt = AdaptLoop(fs=fs, preset='dau1997', learnable=learnable)
    adapt = adapt.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {adapt}")
    print(f"  Preset: dau1997 (limit=10.0)")
    
    # Verify preset parameters
    assert adapt.num_loops == 5
    limit_val = adapt.limit.item() if isinstance(adapt.limit, torch.Tensor) else adapt.limit
    assert limit_val == 10.0
    print(f"  Verified: num_loops=5, limit=10.0")
    
    # Forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=31, duration=0.1, fs=fs)
    avg_time = time_forward_pass(adapt, signal, device=device.split(':')[0])
    output = adapt(signal)
    assert output.device.type == device.split(':')[0]
    print(f"✓ Forward: {signal.shape} -> {output.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ AdaptLoop (dau1997) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: AdaptLoop - Preset osses2021
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_adaptloop_osses2021(device, learnable):
    """Test AdaptLoop with osses2021 preset on specified device."""
    from torch_amt.common.adaptation import AdaptLoop
    
    print(f"\n{'='*80}")
    print(f"TEST: AdaptLoop (osses2021) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with osses2021 preset
    fs = 16000
    adapt = AdaptLoop(fs=fs, preset='osses2021', learnable=learnable)
    adapt = adapt.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {adapt}")
    print(f"  Preset: osses2021 (limit=5.0)")
    
    # Verify preset parameters
    assert adapt.num_loops == 5
    limit_val = adapt.limit.item() if isinstance(adapt.limit, torch.Tensor) else adapt.limit
    assert limit_val == 5.0
    print(f"  Verified: num_loops=5, limit=5.0")
    
    # Forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=31, duration=0.1, fs=fs)
    avg_time = time_forward_pass(adapt, signal, device=device.split(':')[0])
    output = adapt(signal)
    assert output.device.type == device.split(':')[0]
    print(f"✓ Forward: {signal.shape} -> {output.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ AdaptLoop (osses2021) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: AdaptLoop - Preset paulick2024
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_adaptloop_paulick2024(device, learnable):
    """Test AdaptLoop with paulick2024 preset on specified device."""
    from torch_amt.common.adaptation import AdaptLoop
    
    print(f"\n{'='*80}")
    print(f"TEST: AdaptLoop (paulick2024) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with paulick2024 preset (requires 50 channels)
    fs = 16000
    adapt = AdaptLoop(fs=fs, preset='paulick2024', learnable=learnable)
    adapt = adapt.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {adapt}")
    print(f"  Preset: paulick2024 (limit=10.0, freq-specific minlvl)")
    
    # Verify preset parameters
    assert adapt.num_loops == 5
    assert adapt.use_freq_specific_minlvl == True
    assert adapt.minlvl_per_channel.shape[0] == 50
    print(f"  Verified: num_loops=5, freq_specific_minlvl=True (50 channels)")
    
    # Forward pass (MUST use 50 channels for paulick2024)
    signal = create_test_signal(device, batch_size=2, n_channels=50, duration=0.1, fs=fs)
    avg_time = time_forward_pass(adapt, signal, device=device.split(':')[0])
    output = adapt(signal)
    assert output.device.type == device.split(':')[0]
    print(f"✓ Forward: {signal.shape} -> {output.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ AdaptLoop (paulick2024) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: AdaptLoop - Custom Parameters
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_adaptloop_custom(device, learnable):
    """Test AdaptLoop with custom parameters on specified device."""
    from torch_amt.common.adaptation import AdaptLoop
    
    print(f"\n{'='*80}")
    print(f"TEST: AdaptLoop (custom) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Custom configuration: 3 loops with custom time constants
    fs = 16000
    tau_custom = torch.tensor([0.01, 0.1, 0.5])
    limit_custom = 8.0
    minspl_custom = -10.0
    
    adapt = AdaptLoop(fs=fs, tau=tau_custom, limit=limit_custom, 
                     minspl=minspl_custom, learnable=learnable)
    adapt = adapt.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {adapt}")
    print(f"  Custom: num_loops=3, tau=[0.01, 0.1, 0.5], limit=8.0, minspl=-10 dB")
    
    # Verify custom parameters
    assert adapt.num_loops == 3
    limit_val = adapt.limit.item() if isinstance(adapt.limit, torch.Tensor) else adapt.limit
    assert abs(limit_val - 8.0) < 1e-6
    print(f"  Verified: num_loops=3, limit=8.0")
    
    # Forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=20, duration=0.1, fs=fs)
    avg_time = time_forward_pass(adapt, signal, device=device.split(':')[0])
    output = adapt(signal)
    assert output.device.type == device.split(':')[0]
    print(f"✓ Forward: {signal.shape} -> {output.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ AdaptLoop (custom) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: AdaptLoop - Per-Channel minspl
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_adaptloop_perchannel_minspl(device):
    """Test AdaptLoop with per-channel minspl on specified device."""
    from torch_amt.common.adaptation import AdaptLoop
    
    print(f"\n{'='*80}")
    print(f"TEST: AdaptLoop (per-channel minspl) - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Per-channel minspl (frequency-dependent threshold)
    fs = 16000
    n_channels = 20
    minspl_perchan = torch.linspace(-20, 10, n_channels)  # Varying thresholds
    
    adapt = AdaptLoop(fs=fs, minspl=minspl_perchan, learnable=False)
    adapt = adapt.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {adapt}")
    print(f"  Per-channel minspl: {n_channels} channels, range=[{minspl_perchan.min():.1f}, {minspl_perchan.max():.1f}] dB")
    
    # Verify per-channel setup
    assert adapt.use_freq_specific_minlvl == True
    assert adapt.minlvl_per_channel.shape[0] == n_channels
    print(f"  Verified: freq_specific_minlvl=True, {n_channels} channels")
    
    # Forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=n_channels, duration=0.1, fs=fs)
    avg_time = time_forward_pass(adapt, signal, device=device.split(':')[0])
    output = adapt(signal)
    assert output.device.type == device.split(':')[0]
    print(f"✓ Forward: {signal.shape} -> {output.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ AdaptLoop (per-channel minspl) passed on {device.upper()}\n")


# ================================================================================================
# Main: Standalone Execution
# ================================================================================================

def main():
    """Run all tests on all available devices (standalone execution)."""
    print("\n" + "="*80)
    print("ADAPTATION DEVICE COMPATIBILITY TEST SUITE")
    print("="*80)
    
    # Print device info
    print_device_info()
    
    devices = get_available_devices()
    
    print(f"Running tests on {len(devices)} device(s): {', '.join(devices)}\n")
    
    # Test functions
    test_functions = [
        ("AdaptLoop (default)", test_adaptloop_default),
        ("AdaptLoop (dau1997)", test_adaptloop_dau1997),
        ("AdaptLoop (osses2021)", test_adaptloop_osses2021),
        ("AdaptLoop (paulick2024)", test_adaptloop_paulick2024),
        ("AdaptLoop (custom)", test_adaptloop_custom),
        ("AdaptLoop (per-channel minspl)", test_adaptloop_perchannel_minspl),
    ]
    
    # Run all tests on all devices with both learnable values
    results = {}
    learnable_values = [False, True]
    
    for device in devices:
        print(f"\n{'#'*80}")
        print(f"# TESTING ON DEVICE: {device.upper()}")
        print(f"{'#'*80}\n")
        
        results[device] = {}
        
        for test_name, test_func in test_functions:
            results[device][test_name] = []
            
            # Check if test requires learnable parameter
            if test_func == test_adaptloop_perchannel_minspl:
                # Only run once (no learnable parameter)
                try:
                    test_func(device)
                    results[device][test_name].append("✓ PASSED")
                except Exception as e:
                    print(f"\n✗ FAILED: {test_name} on {device.upper()}")
                    print(f"  Error: {str(e)}\n")
                    results[device][test_name].append(f"✗ FAILED: {str(e)}")
            else:
                # Run with learnable=False and learnable=True
                for learnable in learnable_values:
                    try:
                        test_func(device, learnable)
                        results[device][test_name].append(f"✓ PASSED (learnable={learnable})")
                    except Exception as e:
                        print(f"\n✗ FAILED: {test_name} on {device.upper()} (learnable={learnable})")
                        print(f"  Error: {str(e)}\n")
                        results[device][test_name].append(f"✗ FAILED (learnable={learnable}): {str(e)}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for device in devices:
        print(f"{device.upper()}:")
        for test_name, test_results in results[device].items():
            status = "✓ PASSED" if all("✓" in r for r in test_results) else "✗ FAILED"
            print(f"  {test_name:40s}: {status}")
        print()
    
    # Check if all tests passed
    all_passed = all(
        all("✓" in r for r in test_results)
        for device_results in results.values()
        for test_results in device_results.values()
    )
    
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED ON ALL DEVICES")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
