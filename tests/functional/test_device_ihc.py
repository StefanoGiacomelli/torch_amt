"""Device Compatibility Test Suite for ihc.py

This test suite verifies that inner hair cell (IHC) models in ihc.py work correctly
across all available devices (CPU, CUDA, MPS).

Contents:
- 2 nn.Module classes: IHCEnvelope, IHCPaulick2024

Test structure:
- Initialization with default/preset parameters
- Application on single input
- Application on batch input
- Device transfer (CPU, CUDA, MPS)
- Module repr and parameter inspection
- Timing measurements
- Learnable parameter verification
- Preset configurations test (dau1996, breebaart2001, king2019, lindemann)
- Shape handling: (F, T) and (B, F, T)

Usage:
    # Standalone execution (tests all available devices)
    python test_device_ihc.py
    
    # pytest execution
    pytest test_device_ihc.py -v
    
    # pytest execution on specific device
    pytest test_device_ihc.py -v -k "cpu"
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
                       duration: float = 0.1, fs: int = 16000, dtype: torch.dtype = torch.float32) -> torch.Tensor:
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
    dtype : torch.dtype
        Data type for the signal
        
    Returns
    -------
    torch.Tensor
        Filterbank output signal, shape (batch_size, n_channels, n_samples)
    """
    n_samples = int(fs * duration)
    # Simulate filterbank output (positive values with some dynamic range)
    signal = torch.randn(batch_size, n_channels, n_samples, device=device, dtype=dtype).abs() * 0.01 + 0.001
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
# Test: IHCEnvelope - dau1996 preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_ihcenvelope_dau1996(device, learnable):
    """Test IHCEnvelope with dau1996 preset."""
    from torch_amt.common.ihc import IHCEnvelope

    from torch_amt.common.ihc import IHCEnvelope
    
    print("\n" + "=" * 80)
    print(f"TEST: IHCEnvelope (dau1996) - Device: {device.upper()}, Learnable: {learnable}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCEnvelope(fs=fs, method='dau1996', learnable=learnable).to(device)
    
    # Check initialization
    print("✓ Initialization successful")
    print(f"  Module: {ihc}")
    print(f"  extra_repr: {ihc.extra_repr()}")
    
    # Check parameters
    if learnable:
        params = list(ihc.parameters())
        param_names = [name for name, _ in ihc.named_parameters()]
        print(f"  Parameters: {len(params)} total, names={param_names}")
    else:
        print(f"  Parameters: 0 (buffers only)")
    
    # Verify preset configuration
    assert ihc.method == 'dau1996'
    assert ihc.cutoff == 1000.0
    assert ihc.order == 1
    assert ihc.iterations == 1
    print(f"  Verified: method=dau1996, cutoff=1000 Hz, order=1, iterations=1")
    
    # Test forward pass
    x = create_test_signal(device, batch_size=2, n_channels=31, duration=0.1, fs=fs)
    print(f"  Input device: {x.device}")
    
    y = ihc(x)
    print(f"  Output device: {y.device}")
    assert y.shape == x.shape
    
    # Timing
    avg_time = time_forward_pass(ihc, x, device=device)
    print(f"✓ Forward: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCEnvelope (dau1996) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: IHCEnvelope - breebaart2001 preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_ihcenvelope_breebaart2001(device, learnable):
    """Test IHCEnvelope with breebaart2001 preset."""
    from torch_amt.common.ihc import IHCEnvelope

    print("\n" + "=" * 80)
    print(f"TEST: IHCEnvelope (breebaart2001) - Device: {device.upper()}, Learnable: {learnable}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCEnvelope(fs=fs, method='breebaart2001', learnable=learnable).to(device)
    
    print("✓ Initialization successful")
    print(f"  Module: {ihc}")
    
    # Verify preset configuration
    assert ihc.method == 'breebaart2001'
    assert ihc.cutoff == 2000.0
    assert ihc.order == 1
    assert ihc.iterations == 5
    print(f"  Verified: method=breebaart2001, cutoff=2000 Hz, iterations=5 (effective ~770 Hz)")
    
    # Test forward pass
    x = create_test_signal(device, batch_size=2, n_channels=31, duration=0.1, fs=fs)
    y = ihc(x)
    
    # Timing
    avg_time = time_forward_pass(ihc, x, device=device)
    print(f"✓ Forward: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCEnvelope (breebaart2001) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: IHCEnvelope - king2019 preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_ihcenvelope_king2019(device, learnable):
    """Test IHCEnvelope with king2019 preset."""
    from torch_amt.common.ihc import IHCEnvelope

    from torch_amt.common.ihc import IHCEnvelope
    print("\n" + "=" * 80)
    print(f"TEST: IHCEnvelope (king2019) - Device: {device.upper()}, Learnable: {learnable}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCEnvelope(fs=fs, method='king2019', learnable=learnable).to(device)
    
    print("✓ Initialization successful")
    print(f"  Module: {ihc}")
    
    # Verify preset configuration
    assert ihc.method == 'king2019'
    assert ihc.cutoff == 1500.0
    assert ihc.order == 1
    assert ihc.iterations == 1
    print(f"  Verified: method=king2019, cutoff=1500 Hz (intentional deviation from MATLAB)")
    
    # Test forward pass
    x = create_test_signal(device, batch_size=2, n_channels=31, duration=0.1, fs=fs)
    y = ihc(x)
    
    # Timing
    avg_time = time_forward_pass(ihc, x, device=device)
    print(f"✓ Forward: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCEnvelope (king2019) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: IHCEnvelope - lindemann preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_ihcenvelope_lindemann(device, learnable):
    """Test IHCEnvelope with lindemann preset."""
    from torch_amt.common.ihc import IHCEnvelope

    from torch_amt.common.ihc import IHCEnvelope
    print("\n" + "=" * 80)
    print(f"TEST: IHCEnvelope (lindemann) - Device: {device.upper()}, Learnable: {learnable}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCEnvelope(fs=fs, method='lindemann', learnable=learnable).to(device)
    
    print("✓ Initialization successful")
    print(f"  Module: {ihc}")
    
    # Verify preset configuration
    assert ihc.method == 'lindemann'
    assert ihc.cutoff == 800.0
    assert ihc.order == 1
    assert ihc.iterations == 1
    print(f"  Verified: method=lindemann, cutoff=800 Hz")
    
    # Test forward pass
    x = create_test_signal(device, batch_size=2, n_channels=31, duration=0.1, fs=fs)
    y = ihc(x)
    
    # Timing
    avg_time = time_forward_pass(ihc, x, device=device)
    print(f"✓ Forward: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCEnvelope (lindemann) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: IHCEnvelope - Custom Parameters
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_ihcenvelope_custom(device, learnable):
    """Test IHCEnvelope with custom cutoff override."""
    from torch_amt.common.ihc import IHCEnvelope

    from torch_amt.common.ihc import IHCEnvelope
    print("\n" + "=" * 80)
    print(f"TEST: IHCEnvelope (custom) - Device: {device.upper()}, Learnable: {learnable}")
    print("=" * 80 + "\n")
    
    fs = 16000
    # Custom: dau1996 method but with cutoff override
    ihc = IHCEnvelope(fs=fs, method='dau1996', cutoff=1200.0, learnable=learnable).to(device)
    
    print("✓ Initialization successful")
    print(f"  Module: {ihc}")
    print(f"  Custom: cutoff=1200 Hz (overriding dau1996 default)")
    
    # Verify configuration
    assert ihc.method == 'dau1996'
    assert ihc.cutoff == 1200.0
    print(f"  Verified: cutoff override successful")
    
    # Test forward pass
    x = create_test_signal(device, batch_size=2, n_channels=20, duration=0.1, fs=fs)
    y = ihc(x)
    
    # Timing
    avg_time = time_forward_pass(ihc, x, device=device)
    print(f"✓ Forward: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCEnvelope (custom) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: IHCEnvelope - 2D Input
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_ihcenvelope_2d_input(device):
    """Test IHCEnvelope with 2D input (no batch dimension)."""
    from torch_amt.common.ihc import IHCEnvelope

    from torch_amt.common.ihc import IHCEnvelope
    
    print("\n" + "=" * 80)
    print(f"TEST: IHCEnvelope (2D input) - Device: {device.upper()}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCEnvelope(fs=fs, method='dau1996').to(device)
    
    # 2D input (F, T)
    x_2d = torch.randn(31, 800, device=device)
    y_2d = ihc(x_2d)
    
    assert y_2d.shape == x_2d.shape
    assert y_2d.ndim == 2
    print(f"✓ Forward 2D: {x_2d.shape} -> {y_2d.shape}")
    
    print(f"\n✓ IHCEnvelope (2D input) passed on {device.upper()}\n")


# ================================================================================================
# Test: IHCPaulick2024 - Default Parameters
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_ihcpaulick2024_default(device, learnable):
    """Test IHCPaulick2024 with default parameters."""
    from torch_amt.common.ihc import IHCPaulick2024
    print("\n" + "=" * 80)
    print(f"TEST: IHCPaulick2024 (default) - Device: {device.upper()}, Learnable: {learnable}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCPaulick2024(fs=fs, learnable=learnable).to(device)
    
    # Check initialization
    print("✓ Initialization successful")
    print(f"  Module: {ihc}")
    print(f"  extra_repr: {ihc.extra_repr()}")
    
    # Check parameters
    if learnable:
        params = list(ihc.parameters())
        param_names = [name for name, _ in ihc.named_parameters()]
        print(f"  Parameters: {len(params)} total (13 physiological), names={param_names[:3]}...")
    else:
        print(f"  Parameters: 0 (buffers only)")
    
    # Verify configuration
    assert ihc.fs == fs
    assert ihc.dtype == torch.float32  # Changed to float32 for MPS compatibility
    assert ihc.precharge_duration == 0.05
    print(f"  Verified: fs={fs} Hz, dtype=float32, precharge=50 ms")
    
    # Test forward pass with float32 input (changed from float64 for MPS compatibility)
    x = create_test_signal(device, batch_size=2, n_channels=50, duration=0.1, fs=fs, dtype=torch.float32)
    print(f"  Input device: {x.device}, dtype: {x.dtype}")
    
    y = ihc(x)
    print(f"  Output device: {y.device}, dtype: {y.dtype}")
    assert y.shape == x.shape
    assert y.dtype == torch.float32
    
    # Timing
    avg_time = time_forward_pass(ihc, x, device=device)
    print(f"✓ Forward: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCPaulick2024 (default) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: IHCPaulick2024 - Batch Sizes
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_ihcpaulick2024_batch_sizes(device):
    """Test IHCPaulick2024 with different batch sizes."""
    from torch_amt.common.ihc import IHCPaulick2024
    print("\n" + "=" * 80)
    print(f"TEST: IHCPaulick2024 (batch sizes) - Device: {device.upper()}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCPaulick2024(fs=fs).to(device)
    
    print("✓ Initialization successful")
    
    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        x = create_test_signal(device, batch_size=batch_size, n_channels=50, duration=0.05, fs=fs, dtype=torch.float32)
        y = ihc(x)
        
        assert y.shape == x.shape
        avg_time = time_forward_pass(ihc, x, device=device, n_runs=2)
        print(f"✓ Batch {batch_size}: {x.shape} -> {y.shape} ({avg_time:.3f} ms avg)")
    
    print(f"\n✓ IHCPaulick2024 (batch sizes) passed on {device.upper()}\n")


# ================================================================================================
# Test: IHCPaulick2024 - 2D Input
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_ihcpaulick2024_2d_input(device):
    """Test IHCPaulick2024 with 2D input (no batch dimension)."""
    from torch_amt.common.ihc import IHCPaulick2024
    print("\n" + "=" * 80)
    print(f"TEST: IHCPaulick2024 (2D input) - Device: {device.upper()}")
    print("=" * 80 + "\n")
    
    fs = 16000
    ihc = IHCPaulick2024(fs=fs).to(device)
    
    # 2D input (F, T)
    x_2d = torch.randn(50, 800, dtype=torch.float32, device=device)
    y_2d = ihc(x_2d)
    
    assert y_2d.shape == x_2d.shape
    assert y_2d.ndim == 2
    print(f"✓ Forward 2D: {x_2d.shape} -> {y_2d.shape}")
    
    print(f"\n✓ IHCPaulick2024 (2D input) passed on {device.upper()}\n")


# =============================================================================
# Main Runner
# =============================================================================

def main():
    """Run all tests manually."""
    print("\n" + "=" * 80)
    print("IHC DEVICE COMPATIBILITY TEST SUITE")
    print("=" * 80)
    
    print_device_info()
    
    devices = get_available_devices()
    print(f"Running tests on {len(devices)} device(s): {', '.join(devices)}\n")
    
    # Test counters
    test_results = {device: {} for device in devices}
    
    # Run tests for each device
    for device in devices:
        print("\n" + "#" * 80)
        print(f"# TESTING ON DEVICE: {device.upper()}")
        print("#" * 80 + "\n")
        
        # IHCEnvelope tests
        for learnable in [False, True]:
            try:
                test_ihcenvelope_dau1996(device, learnable)
                test_results[device][f'IHCEnvelope (dau1996, learnable={learnable})'] = 'PASSED'
            except Exception as e:
                test_results[device][f'IHCEnvelope (dau1996, learnable={learnable})'] = f'FAILED: {e}'
            
            try:
                test_ihcenvelope_breebaart2001(device, learnable)
                test_results[device][f'IHCEnvelope (breebaart2001, learnable={learnable})'] = 'PASSED'
            except Exception as e:
                test_results[device][f'IHCEnvelope (breebaart2001, learnable={learnable})'] = f'FAILED: {e}'
            
            try:
                test_ihcenvelope_king2019(device, learnable)
                test_results[device][f'IHCEnvelope (king2019, learnable={learnable})'] = 'PASSED'
            except Exception as e:
                test_results[device][f'IHCEnvelope (king2019, learnable={learnable})'] = f'FAILED: {e}'
            
            try:
                test_ihcenvelope_lindemann(device, learnable)
                test_results[device][f'IHCEnvelope (lindemann, learnable={learnable})'] = 'PASSED'
            except Exception as e:
                test_results[device][f'IHCEnvelope (lindemann, learnable={learnable})'] = f'FAILED: {e}'
            
            try:
                test_ihcenvelope_custom(device, learnable)
                test_results[device][f'IHCEnvelope (custom, learnable={learnable})'] = 'PASSED'
            except Exception as e:
                test_results[device][f'IHCEnvelope (custom, learnable={learnable})'] = f'FAILED: {e}'
        
        # IHCEnvelope 2D input (no learnable parametrization)
        try:
            test_ihcenvelope_2d_input(device)
            test_results[device]['IHCEnvelope (2D input)'] = 'PASSED'
        except Exception as e:
            test_results[device]['IHCEnvelope (2D input)'] = f'FAILED: {e}'
        
        # IHCPaulick2024 tests
        for learnable in [False, True]:
            try:
                test_ihcpaulick2024_default(device, learnable)
                test_results[device][f'IHCPaulick2024 (default, learnable={learnable})'] = 'PASSED'
            except Exception as e:
                test_results[device][f'IHCPaulick2024 (default, learnable={learnable})'] = f'FAILED: {e}'
        
        try:
            test_ihcpaulick2024_batch_sizes(device)
            test_results[device]['IHCPaulick2024 (batch sizes)'] = 'PASSED'
        except Exception as e:
            test_results[device]['IHCPaulick2024 (batch sizes)'] = f'FAILED: {e}'
        
        try:
            test_ihcpaulick2024_2d_input(device)
            test_results[device]['IHCPaulick2024 (2D input)'] = 'PASSED'
        except Exception as e:
            test_results[device]['IHCPaulick2024 (2D input)'] = f'FAILED: {e}'
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80 + "\n")
    
    for device in devices:
        print(f"{device.upper()}:")
        for test_name, result in test_results[device].items():
            status = "✓ PASSED" if result == 'PASSED' else f"✗ {result}"
            print(f"  {test_name:45s}: {status}")
        print()
    
    # Check if all passed
    all_passed = all(result == 'PASSED' for device_results in test_results.values() 
                     for result in device_results.values())
    
    if all_passed:
        print("=" * 80)
        print("✓ ALL TESTS PASSED ON ALL DEVICES")
        print("=" * 80 + "\n")
    else:
        print("=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
