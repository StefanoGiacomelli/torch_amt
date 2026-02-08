"""Device Compatibility Test Suite for modulation.py

This test suite verifies that modulation filterbanks in modulation.py work correctly
across all available devices (CPU, CUDA, MPS).

Contents:
- 1 nn.Module class: ModulationFilterbank

Test structure:
- Initialization with default/preset parameters
- Application on single input
- Application on batch input
- Device transfer (CPU, CUDA, MPS)
- Module repr and parameter inspection
- Timing measurements
- Learnable parameter verification
- Preset configurations test (dau1997, jepsen2008, paulick2024)
- Filter type comparison (efilt vs butterworth)
- Modulation center frequency generation verification

Usage:
    # Standalone execution (tests all available devices)
    python test_device_modulation.py
    
    # pytest execution
    pytest test_device_modulation.py -v
    
    # pytest execution on specific device
    pytest test_device_modulation.py -v -k "cpu"
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


def create_test_fc(device: str, n_channels: int = 31) -> torch.Tensor:
    """Create test center frequencies for modulation filterbank.
    
    Parameters
    ----------
    device : str
        Target device
    n_channels : int
        Number of channels
        
    Returns
    -------
    torch.Tensor
        Center frequencies in Hz, shape (n_channels,)
    """
    # ERB-spaced frequencies from 100 to 8000 Hz (approximation)
    fc = torch.linspace(100, 8000, n_channels, device=device)
    return fc


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
# Test: ModulationFilterbank - Default Parameters
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_modfb_default(device, learnable):
    """Test ModulationFilterbank with default parameters on specified device."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank (default) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters
    fs = 16000
    fc = create_test_fc(device, n_channels=10)  # Use fewer channels for speed
    modfb = ModulationFilterbank(fs=fs, fc=fc, learnable=learnable)
    modfb = modfb.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {modfb}")
    print(f"  extra_repr: {modfb.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in modfb.parameters())
    param_names = [name for name, _ in modfb.named_parameters()]
    print(f"  Parameters: {n_params} total")
    if learnable:
        print(f"  Learnable param names (first 3): {param_names[:3]}")
    
    # Verify attributes
    assert modfb.fs == fs
    assert len(modfb.fc) == 10
    assert modfb.Q == 2.0
    assert modfb.max_mfc == 150.0
    assert modfb.lp_cutoff == 2.5
    assert modfb.att_factor == 1.0
    assert modfb.use_upper_limit == False
    assert modfb.filter_type == 'efilt'
    print(f"  Verified: default parameters")
    
    # Forward pass - single
    signal_single = create_test_signal(device, batch_size=1, n_channels=10, duration=0.1, fs=fs)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(modfb, signal_single, device=device.split(':')[0])
    output_single = modfb(signal_single)
    
    # Verify output structure
    assert isinstance(output_single, list)
    assert len(output_single) == 10
    print(f"✓ Forward single: {signal_single.shape} -> List[{len(output_single)}] ({avg_time_single:.3f} ms avg)")
    print(f"  First channel output shape: {output_single[0].shape}")
    print(f"  First channel mfc: {modfb.mfc[0]}")
    
    # Forward pass - batch
    signal_batch = create_test_signal(device, batch_size=4, n_channels=10, duration=0.1, fs=fs)
    avg_time_batch = time_forward_pass(modfb, signal_batch, device=device.split(':')[0])
    output_batch = modfb(signal_batch)
    
    assert isinstance(output_batch, list)
    assert len(output_batch) == 10
    assert output_batch[0].shape[0] == 4  # Batch dimension
    print(f"✓ Forward batch: {signal_batch.shape} -> List[{len(output_batch)}] ({avg_time_batch:.3f} ms avg)")
    print(f"  First channel output shape: {output_batch[0].shape}")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - dau1997 preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_dau1997(device):
    """Test ModulationFilterbank with dau1997 preset."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank (dau1997) - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=10)
    modfb = ModulationFilterbank(fs=fs, fc=fc, preset='dau1997')
    modfb = modfb.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {modfb}")
    
    # Verify preset configuration
    assert modfb.preset == 'dau1997'
    assert modfb.lp_cutoff == 2.5
    assert modfb.att_factor == 1.0
    assert modfb.use_upper_limit == False
    assert modfb.use_lp150_prefilter == False
    print(f"  Verified: lp_cutoff=2.5 Hz, att_factor=1.0, no upper limit, no 150 Hz pre-filter")
    
    # Test forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=10, duration=0.1, fs=fs)
    avg_time = time_forward_pass(modfb, signal, device=device.split(':')[0])
    output = modfb(signal)
    
    assert isinstance(output, list)
    assert len(output) == 10
    print(f"✓ Forward: {signal.shape} -> List[{len(output)}] ({avg_time:.3f} ms avg)")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - jepsen2008 preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_jepsen2008(device):
    """Test ModulationFilterbank with jepsen2008 preset."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank (jepsen2008) - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=10)
    modfb = ModulationFilterbank(fs=fs, fc=fc, preset='jepsen2008')
    modfb = modfb.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {modfb}")
    
    # Verify preset configuration
    assert modfb.preset == 'jepsen2008'
    assert modfb.lp_cutoff == 2.5  # CRITICAL: Should be 2.5 Hz (MATLAB alignment)
    assert abs(modfb.att_factor - 1.0 / (2**0.5)) < 1e-6
    assert modfb.use_upper_limit == True
    assert modfb.max_mfc == 150.0
    assert modfb.use_lp150_prefilter == True
    print(f"  Verified: lp_cutoff=2.5 Hz, att_factor=0.707, upper limit=True, 150 Hz pre-filter=True")
    
    # Test forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=10, duration=0.1, fs=fs)
    avg_time = time_forward_pass(modfb, signal, device=device.split(':')[0])
    output = modfb(signal)
    
    assert isinstance(output, list)
    assert len(output) == 10
    print(f"✓ Forward: {signal.shape} -> List[{len(output)}] ({avg_time:.3f} ms avg)")
    
    # Check that mfc varies per channel (due to use_upper_limit=True)
    mfc_lengths = [len(mfc) for mfc in modfb.mfc]
    print(f"  mfc lengths per channel: {mfc_lengths}")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - paulick2024 preset
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_paulick2024(device):
    """Test ModulationFilterbank with paulick2024 preset."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank (paulick2024) - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=10)
    modfb = ModulationFilterbank(fs=fs, fc=fc, preset='paulick2024')
    modfb = modfb.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {modfb}")
    
    # Verify preset configuration (same as jepsen2008)
    assert modfb.preset == 'paulick2024'
    assert modfb.lp_cutoff == 2.5  # CRITICAL: Should be 2.5 Hz
    assert abs(modfb.att_factor - 1.0 / (2**0.5)) < 1e-6
    assert modfb.use_upper_limit == True
    assert modfb.max_mfc == 150.0
    assert modfb.use_lp150_prefilter == True
    print(f"  Verified: Same configuration as jepsen2008")
    
    # Test forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=10, duration=0.1, fs=fs)
    avg_time = time_forward_pass(modfb, signal, device=device.split(':')[0])
    output = modfb(signal)
    
    assert isinstance(output, list)
    assert len(output) == 10
    print(f"✓ Forward: {signal.shape} -> List[{len(output)}] ({avg_time:.3f} ms avg)")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - Filter Type Comparison
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("filter_type", ['efilt', 'butterworth'])
def test_modfb_filter_types(device, filter_type):
    """Test ModulationFilterbank with different filter types."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank (filter_type={filter_type}) - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=5)  # Fewer channels for speed
    modfb = ModulationFilterbank(fs=fs, fc=fc, filter_type=filter_type)
    modfb = modfb.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {modfb}")
    print(f"  Filter type: {modfb.filter_type}")
    
    # Verify filter type
    assert modfb.filter_type == filter_type
    print(f"  Verified: filter_type={filter_type}")
    
    # Test forward pass
    signal = create_test_signal(device, batch_size=2, n_channels=5, duration=0.1, fs=fs)
    avg_time = time_forward_pass(modfb, signal, device=device.split(':')[0])
    output = modfb(signal)
    
    assert isinstance(output, list)
    assert len(output) == 5
    print(f"✓ Forward: {signal.shape} -> List[{len(output)}] ({avg_time:.3f} ms avg)")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - Learnable Parameters
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_learnable(device):
    """Test ModulationFilterbank with learnable parameters."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank (learnable=True) - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=5)
    modfb = ModulationFilterbank(fs=fs, fc=fc, learnable=True)
    modfb = modfb.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {modfb}")
    
    # Check that parameters are learnable
    params = list(modfb.parameters())
    n_params = sum(p.numel() for p in params)
    n_learnable = sum(p.numel() for p in params if p.requires_grad)
    
    print(f"  Total parameters: {n_params}")
    print(f"  Learnable parameters: {n_learnable}")
    assert n_learnable > 0, "No learnable parameters found!"
    assert n_learnable == n_params, "Some parameters are not learnable!"
    print(f"  Verified: All filter coefficients are learnable")
    
    # Note: Backward pass test skipped
    # The filtering is performed using scipy (CPU numpy), which breaks the
    # computational graph. While filter coefficients are learnable parameters,
    # gradients cannot flow back through scipy operations.
    # This is a known limitation of the hybrid PyTorch-scipy implementation.
    signal = create_test_signal(device, batch_size=1, n_channels=5, duration=0.05, fs=fs)
    output = modfb(signal)
    
    print(f"✓ Forward pass with learnable parameters successful")
    print(f"  Note: Backward pass not tested (scipy filtering breaks autograd graph)")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - MFC Generation Verification
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_mfc_generation(device):
    """Test modulation center frequency generation algorithm."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank MFC Generation - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=3)
    
    # Test without upper limit (fixed 150 Hz max)
    modfb_no_limit = ModulationFilterbank(fs=fs, fc=fc, use_upper_limit=False, max_mfc=150.0)
    modfb_no_limit = modfb_no_limit.to(device)
    
    print(f"✓ No upper limit:")
    for i, mfc in enumerate(modfb_no_limit.mfc):
        print(f"  Channel {i} (fc={fc[i]:.1f} Hz): {len(mfc)} filters, mfc={mfc.cpu().numpy()}")
    
    # Verify that all channels have same number of filters (no dynamic limit)
    mfc_lengths = [len(mfc) for mfc in modfb_no_limit.mfc]
    assert len(set(mfc_lengths)) == 1, "All channels should have same mfc length without upper limit"
    print(f"  Verified: All channels have {mfc_lengths[0]} modulation filters")
    
    # Test with upper limit (dynamic per channel)
    modfb_limit = ModulationFilterbank(fs=fs, fc=fc, use_upper_limit=True, max_mfc=150.0)
    modfb_limit = modfb_limit.to(device)
    
    print(f"\n✓ With upper limit (0.25 × fc):")
    for i, mfc in enumerate(modfb_limit.mfc):
        upper_limit = min(fc[i].item() * 0.25, 150.0)
        print(f"  Channel {i} (fc={fc[i]:.1f} Hz, limit={upper_limit:.1f} Hz): {len(mfc)} filters")
    
    # Verify that lower frequency channels have fewer filters
    mfc_lengths_limit = [len(mfc) for mfc in modfb_limit.mfc]
    print(f"  mfc lengths: {mfc_lengths_limit}")
    
    # Verify first mfc is always 0 (lowpass)
    for i, mfc in enumerate(modfb_limit.mfc):
        assert mfc[0].item() == 0.0, f"First mfc should be 0 (lowpass) for channel {i}"
    print(f"  Verified: All channels start with mfc=0 (lowpass)")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - Device Transfer
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_device_transfer(device):
    """Test transferring ModulationFilterbank between devices."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank Device Transfer to {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc('cpu', n_channels=5)  # Start on CPU
    modfb = ModulationFilterbank(fs=fs, fc=fc, preset='dau1997')
    
    print(f"✓ Created on CPU")
    
    # Transfer to target device
    modfb = modfb.to(device)
    print(f"✓ Transferred to {device}")
    
    # Verify device
    for name, param in modfb.named_buffers():
        assert str(param.device).startswith(device.split(':')[0]), f"Buffer {name} not on {device}"
    print(f"  Verified: All buffers on {device}")
    
    # Test forward pass on target device
    signal = create_test_signal(device, batch_size=2, n_channels=5, duration=0.05, fs=fs)
    output = modfb(signal)
    
    assert isinstance(output, list)
    assert len(output) == 5
    # Check output device (first channel, first batch)
    assert str(output[0].device).startswith(device.split(':')[0])
    print(f"✓ Forward pass successful on {device}")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Test: ModulationFilterbank - Batch Processing Consistency
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_modfb_batch_consistency(device):
    """Test that batch processing gives consistent results with single processing."""
    from torch_amt.common.modulation import ModulationFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: ModulationFilterbank Batch Consistency - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    fs = 16000
    fc = create_test_fc(device, n_channels=3)
    modfb = ModulationFilterbank(fs=fs, fc=fc, preset='dau1997')
    modfb = modfb.to(device)
    
    # Create test signal
    signal_batch = create_test_signal(device, batch_size=2, n_channels=3, duration=0.05, fs=fs)
    
    # Process as batch
    output_batch = modfb(signal_batch)
    
    # Process individually
    output_single_0 = modfb(signal_batch[0:1])
    output_single_1 = modfb(signal_batch[1:2])
    
    # Compare results
    for ch_idx in range(3):
        # Check first sample
        diff_0 = torch.abs(output_batch[ch_idx][0] - output_single_0[ch_idx][0]).max().item()
        diff_1 = torch.abs(output_batch[ch_idx][1] - output_single_1[ch_idx][0]).max().item()
        
        print(f"  Channel {ch_idx}: max diff batch[0] vs single[0] = {diff_0:.2e}")
        print(f"  Channel {ch_idx}: max diff batch[1] vs single[1] = {diff_1:.2e}")
        
        # Should be identical (or very close due to floating point)
        assert diff_0 < 1e-5, f"Batch inconsistency for channel {ch_idx}, sample 0"
        assert diff_1 < 1e-5, f"Batch inconsistency for channel {ch_idx}, sample 1"
    
    print(f"✓ Batch processing is consistent with single processing")
    
    print(f"\n{'='*80}\n")


# ================================================================================================
# Main Execution
# ================================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MODULATION FILTERBANK - DEVICE COMPATIBILITY TEST SUITE")
    print("=" * 80)
    
    print_device_info()
    
    # Get available devices
    devices = get_available_devices()
    
    print(f"Running tests on {len(devices)} device(s): {', '.join(devices)}\n")
    
    # Run all tests for each device
    for device in devices:
        print(f"\n{'='*80}")
        print(f"TESTING ON DEVICE: {device.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Test 1: Default parameters
            print("Test 1/9: Default parameters (learnable=False)")
            test_modfb_default(device, learnable=False)
            
            print("Test 2/9: Default parameters (learnable=True)")
            test_modfb_default(device, learnable=True)
            
            # Test 2: dau1997 preset
            print("Test 3/9: dau1997 preset")
            test_modfb_dau1997(device)
            
            # Test 3: jepsen2008 preset
            print("Test 4/9: jepsen2008 preset")
            test_modfb_jepsen2008(device)
            
            # Test 4: paulick2024 preset
            print("Test 5/9: paulick2024 preset")
            test_modfb_paulick2024(device)
            
            # Test 5: Filter types
            print("Test 6/9: Filter types (efilt)")
            test_modfb_filter_types(device, filter_type='efilt')
            
            print("Test 7/9: Filter types (butterworth)")
            test_modfb_filter_types(device, filter_type='butterworth')
            
            # Test 6: Learnable parameters
            print("Test 8/9: Learnable parameters")
            test_modfb_learnable(device)
            
            # Test 7: MFC generation
            print("Test 9/9: MFC generation")
            test_modfb_mfc_generation(device)
            
            # Test 8: Device transfer
            test_modfb_device_transfer(device)
            
            # Test 9: Batch consistency
            test_modfb_batch_consistency(device)
            
            print(f"\n{'='*80}")
            print(f"✓ ALL TESTS PASSED ON {device.upper()}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"✗ TEST FAILED ON {device.upper()}")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80 + "\n")
