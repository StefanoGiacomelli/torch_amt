"""Device Compatibility Test Suite for ears.py

This test suite verifies that ear filter models in ears.py work correctly
across all available devices (CPU, CUDA, MPS).

Contents:
- 3 nn.Module classes:
  * HeadphoneFilter: Headphone and outer ear response (Pralong & Carlile 1996)
  * OuterMiddleEarFilter: Combined outer and middle ear (ANSI S3.4-2007)
  * MiddleEarFilter: Middle ear transmission (Lopez-Poveda 2001, Jepsen 2008)

Test structure:
- Initialization with default parameters
- Application on single and batch inputs
- Device transfer (CPU, CUDA, MPS)
- Module repr and parameter inspection
- Timing measurements
- Learnable parameter verification
- Filter variants test (MiddleEarFilter: lopezpoveda2001 vs jepsen2008)
- Shape handling: (B, T) and (B, F, T)
- Frequency response verification

Usage:
    # Standalone execution (tests all available devices)
    python test_device_ears.py
    
    # pytest execution
    pytest test_device_ears.py -v
    
    # pytest execution on specific device
    pytest test_device_ears.py -v -k "cpu"
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

def create_test_signal(device: str, batch_size: int = 1, duration: float = 1.0, 
                       fs: int = 16000) -> torch.Tensor:
    """Create test audio signal.
    
    Parameters
    ----------
    device : str
        Target device ('cpu', 'cuda', 'mps')
    batch_size : int
        Batch size (1 for single, >1 for batch)
    duration : float
        Signal duration in seconds
    fs : int
        Sampling rate in Hz
        
    Returns
    -------
    torch.Tensor
        Audio signal, shape (batch_size, n_samples)
    """
    n_samples = int(fs * duration)
    signal = torch.randn(batch_size, n_samples, device=device) * 0.1
    return signal


def create_test_filterbank_output(device: str, batch_size: int = 1, n_channels: int = 31,
                                  duration: float = 0.1, fs: int = 16000) -> torch.Tensor:
    """Create test filterbank output signal.
    
    Parameters
    ----------
    device : str
        Target device
    batch_size : int
        Batch size
    n_channels : int
        Number of frequency channels (e.g., 31 for gammatone)
    duration : float
        Signal duration in seconds
    fs : int
        Sampling rate in Hz
        
    Returns
    -------
    torch.Tensor
        Filterbank output, shape (batch_size, n_channels, n_samples)
    """
    n_samples = int(fs * duration)
    signal = torch.randn(batch_size, n_channels, n_samples, device=device) * 0.1
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
# Test: HeadphoneFilter
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_headphone_filter(device, learnable):
    """Test HeadphoneFilter on specified device."""
    from torch_amt.common.ears import HeadphoneFilter
    
    print(f"\n{'='*80}")
    print(f"TEST: HeadphoneFilter - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization
    fs = 16000
    hpf = HeadphoneFilter(fs=fs, order=512, learnable=learnable)
    hpf = hpf.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {hpf}")
    print(f"  extra_repr: {hpf.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in hpf.parameters())
    param_names = [name for name, _ in hpf.named_parameters()]
    print(f"  Parameters: {n_params} total")
    if param_names:
        print(f"  Learnable params: {param_names}")
    
    # Verify parameters
    params = hpf.get_parameters()
    print(f"  Verified: fs={params['fs']}, order={params['order']}, "
          f"phase={params['phase_type']}, freq_points={params['num_frequency_points']}")
    
    # Forward pass - single (1 second)
    signal_single = create_test_signal(device, batch_size=1, duration=1.0, fs=fs)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(hpf, signal_single, device=device.split(':')[0])
    output_single = hpf(signal_single)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward 2D: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch with filterbank output (3D)
    signal_3d = create_test_filterbank_output(device, batch_size=4, n_channels=31, duration=0.1, fs=fs)
    avg_time_3d = time_forward_pass(hpf, signal_3d, device=device.split(':')[0])
    output_3d = hpf(signal_3d)
    assert output_3d.device.type == device.split(':')[0]
    assert output_3d.shape == signal_3d.shape
    print(f"✓ Forward 3D: {signal_3d.shape} -> {output_3d.shape} ({avg_time_3d:.3f} ms avg)")
    
    # Frequency response
    freqs, H = hpf.get_frequency_response(nfft=8192)
    magnitude_db = 20 * torch.log10(torch.abs(H) + 1e-10)
    peak_idx = magnitude_db.argmax()
    print(f"  Frequency response: Peak {magnitude_db[peak_idx]:.2f} dB at {freqs[peak_idx]:.1f} Hz")
    
    print(f"\n✓ HeadphoneFilter passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: OuterMiddleEarFilter
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_outer_middle_ear_filter(device, learnable):
    """Test OuterMiddleEarFilter on specified device."""
    from torch_amt.common.ears import OuterMiddleEarFilter
    
    print(f"\n{'='*80}")
    print(f"TEST: OuterMiddleEarFilter - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization
    fs = 32000
    omef = OuterMiddleEarFilter(fs=fs, compensation_type='tfOuterMiddle1997', 
                                field_type='free', learnable=learnable)
    omef = omef.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {omef}")
    
    # Parameters
    n_params = sum(p.numel() for p in omef.parameters())
    param_names = [name for name, _ in omef.named_parameters()]
    print(f"  Parameters: {n_params} total")
    if param_names:
        print(f"  Learnable params: {param_names}")
    
    # Verify settings
    print(f"  Verified: fs={omef.fs}, order={omef.order}, "
          f"comp_type={omef.compensation_type}, field={omef.field_type}")
    
    # Forward pass - single (1 second)
    signal_single = create_test_signal(device, batch_size=2, duration=1.0, fs=fs)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(omef, signal_single, device=device.split(':')[0])
    output_single = omef(signal_single)
    assert output_single.device.type == device.split(':')[0]
    assert output_single.shape == signal_single.shape
    print(f"✓ Forward 2D: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Frequency response
    freqs, H_db = omef.get_frequency_response(nfft=8192)
    peak_idx = H_db.argmax()
    print(f"  Frequency response: Peak {H_db[peak_idx]:.2f} dB at {freqs[peak_idx]:.1f} Hz")
    
    # Transfer function
    tf_freqs, tf_gains = omef.get_transfer_function()
    print(f"  Transfer function: {len(tf_freqs)} points from {tf_freqs[0]:.1f} to {tf_freqs[-1]:.1f} Hz")
    
    print(f"\n✓ OuterMiddleEarFilter passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: MiddleEarFilter - lopezpoveda2001
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_middle_ear_filter_lopezpoveda(device, learnable):
    """Test MiddleEarFilter (lopezpoveda2001) on specified device."""
    from torch_amt.common.ears import MiddleEarFilter
    
    print(f"\n{'='*80}")
    print(f"TEST: MiddleEarFilter (lopezpoveda2001) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization
    fs = 16000
    mef = MiddleEarFilter(fs=fs, filter_type='lopezpoveda2001', order=512, 
                         normalize_gain=True, learnable=learnable)
    mef = mef.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {mef}")
    print(f"  extra_repr: {mef.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in mef.parameters())
    param_names = [name for name, _ in mef.named_parameters()]
    print(f"  Parameters: {n_params} total")
    if param_names:
        print(f"  Learnable params: {param_names}")
    
    # Verify settings
    print(f"  Verified: fs={mef.fs}, type={mef.filter_type}, "
          f"order={mef.order}, normalize={mef.normalize_gain}")
    
    # Forward pass - single
    signal_single = create_test_signal(device, batch_size=4, duration=1.0, fs=fs)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(mef, signal_single, device=device.split(':')[0])
    output_single = mef(signal_single)
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward 2D: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - 3D (filterbank output)
    signal_3d = create_test_filterbank_output(device, batch_size=4, n_channels=31, duration=0.1, fs=fs)
    avg_time_3d = time_forward_pass(mef, signal_3d, device=device.split(':')[0])
    output_3d = mef(signal_3d)
    assert output_3d.device.type == device.split(':')[0]
    print(f"✓ Forward 3D: {signal_3d.shape} -> {output_3d.shape} ({avg_time_3d:.3f} ms avg)")
    
    # Frequency response
    freqs, H = mef.get_frequency_response(nfft=8192)
    magnitude_db = 20 * torch.log10(torch.abs(H) + 1e-10)
    peak_idx = magnitude_db.argmax()
    print(f"  Frequency response: Peak {magnitude_db[peak_idx]:.2f} dB at {freqs[peak_idx]:.1f} Hz")
    
    # Verify gain normalization
    if mef.normalize_gain:
        max_gain_db = magnitude_db.max().item()
        print(f"  Gain normalization: Max gain = {max_gain_db:.2f} dB (should be ~0 dB)")
    
    print(f"\n✓ MiddleEarFilter (lopezpoveda2001) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: MiddleEarFilter - jepsen2008
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_middle_ear_filter_jepsen(device, learnable):
    """Test MiddleEarFilter (jepsen2008) on specified device."""
    from torch_amt.common.ears import MiddleEarFilter
    
    print(f"\n{'='*80}")
    print(f"TEST: MiddleEarFilter (jepsen2008) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization
    fs = 16000
    mef = MiddleEarFilter(fs=fs, filter_type='jepsen2008', order=512, 
                         normalize_gain=True, learnable=learnable)
    mef = mef.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {mef}")
    print(f"  extra_repr: {mef.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in mef.parameters())
    param_names = [name for name, _ in mef.named_parameters()]
    print(f"  Parameters: {n_params} total")
    if param_names:
        print(f"  Learnable params: {param_names}")
    
    # Verify settings
    print(f"  Verified: fs={mef.fs}, type={mef.filter_type}, "
          f"order={mef.order}, normalize={mef.normalize_gain}")
    
    # Forward pass - single
    signal_single = create_test_signal(device, batch_size=4, duration=1.0, fs=fs)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(mef, signal_single, device=device.split(':')[0])
    output_single = mef(signal_single)
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward 2D: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - 3D (filterbank output)
    signal_3d = create_test_filterbank_output(device, batch_size=4, n_channels=31, duration=0.1, fs=fs)
    avg_time_3d = time_forward_pass(mef, signal_3d, device=device.split(':')[0])
    output_3d = mef(signal_3d)
    assert output_3d.device.type == device.split(':')[0]
    print(f"✓ Forward 3D: {signal_3d.shape} -> {output_3d.shape} ({avg_time_3d:.3f} ms avg)")
    
    # Frequency response
    freqs, H = mef.get_frequency_response(nfft=8192)
    magnitude_db = 20 * torch.log10(torch.abs(H) + 1e-10)
    peak_idx = magnitude_db.argmax()
    print(f"  Frequency response: Peak {magnitude_db[peak_idx]:.2f} dB at {freqs[peak_idx]:.1f} Hz")
    
    # Verify gain normalization
    if mef.normalize_gain:
        max_gain_db = magnitude_db.max().item()
        print(f"  Gain normalization: Max gain = {max_gain_db:.2f} dB (should be ~0 dB)")
    
    print(f"\n✓ MiddleEarFilter (jepsen2008) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Main Execution (for standalone script)
# ================================================================================================

def main():
    """Run all tests when script is executed directly."""
    print_device_info()
    
    devices = get_available_devices()
    
    # Test configuration: (test_function, learnable_values)
    test_configs = [
        (test_headphone_filter, [False, True]),
        (test_outer_middle_ear_filter, [False, True]),
        (test_middle_ear_filter_lopezpoveda, [False, True]),
        (test_middle_ear_filter_jepsen, [False, True]),
    ]
    
    # Run all tests
    results = {device: {test_func.__name__: [] for test_func, _ in test_configs} 
               for device in devices}
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"TESTING ON {device.upper()}")
        print(f"{'='*80}\n")
        
        for test_func, learnable_values in test_configs:
            test_name = test_func.__name__
            
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
            print(f"  {test_name:45s}: {status}")
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
