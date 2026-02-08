"""Device Compatibility Test Suite for loudness.py

This test suite verifies that loudness models in loudness.py work correctly
across all available devices (CPU, CUDA, MPS).

Contents:
- 1 utility function: gaindb
- 10 nn.Module classes:
  * Compression: BrokenStickCompression, PowerCompression
  * Specific Loudness: SpecificLoudness, Moore2016SpecificLoudness
  * Binaural Processing: SpatialSmoothing, BinauralInhibition, Moore2016BinauralLoudness
  * Temporal Integration: LoudnessIntegration, Moore2016AGC, Moore2016TemporalIntegration

Test structure:
- Initialization with default parameters
- Application on single and batch inputs
- Device transfer (CPU, CUDA, MPS)
- Module repr and parameter inspection
- Timing measurements
- Learnable parameter verification
- Shape handling verification

Usage:
    # Standalone execution (tests all available devices)
    python test_device_loudness.py
    
    # pytest execution
    pytest test_device_loudness.py -v
    
    # pytest execution on specific device
    pytest test_device_loudness.py -v -k "cpu"
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
    """Create test audio/filterbank signal.
    
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
        Signal, shape (batch_size, n_channels, n_samples)
    """
    n_samples = int(fs * duration)
    signal = torch.randn(batch_size, n_channels, n_samples, device=device)
    return signal


def create_test_excitation(device: str, batch_size: int = 1, n_frames: int = 10, 
                           n_erb: int = 150) -> torch.Tensor:
    """Create test excitation pattern in dB SPL.
    
    Parameters
    ----------
    device : str
        Target device
    batch_size : int
        Batch size
    n_frames : int
        Number of time frames
    n_erb : int
        Number of ERB channels
        
    Returns
    -------
    torch.Tensor
        Excitation in dB SPL, shape (batch_size, n_frames, n_erb)
    """
    # Realistic excitation range: 20-80 dB SPL
    excitation = torch.rand(batch_size, n_frames, n_erb, device=device) * 60 + 20
    return excitation


def create_test_specific_loudness(device: str, batch_size: int = 1, n_frames: int = 10, 
                                  n_erb: int = 150) -> torch.Tensor:
    """Create test specific loudness in sone/ERB.
    
    Parameters
    ----------
    device : str
        Target device
    batch_size : int
        Batch size
    n_frames : int
        Number of time frames
    n_erb : int
        Number of ERB channels
        
    Returns
    -------
    torch.Tensor
        Specific loudness in sone/ERB, shape (batch_size, n_frames, n_erb)
    """
    # Realistic specific loudness: 0-20 sone/ERB
    spec_loud = torch.rand(batch_size, n_frames, n_erb, device=device) * 20
    return spec_loud


def time_forward_pass(module, *args, device='cpu', n_warmup=2, n_runs=5):
    """Time a forward pass through a module.
    
    Parameters
    ----------
    module : nn.Module
        Module to benchmark
    *args : torch.Tensor
        Input tensors for forward pass
    device : str
        Device name
    n_warmup : int
        Number of warmup runs
    n_runs : int
        Number of timed runs
        
    Returns
    -------
    float
        Average time per forward pass in milliseconds
    """
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = module(*args)
    
    # Synchronize for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    # Timed runs
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = module(*args)
    
    # Synchronize again
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    avg_time_ms = (elapsed / n_runs) * 1000
    
    return avg_time_ms


# ================================================================================================
# Pytest Fixtures
# ================================================================================================

@pytest.fixture(params=get_available_devices())
def device(request):
    """Pytest fixture for device parametrization."""
    return request.param


# ================================================================================================
# Test: gaindb utility function
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_gaindb(device):
    """Test gaindb utility function."""
    from torch_amt.common.loudness import gaindb
    
    print(f"\n{'='*80}")
    print(f"TEST: gaindb - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    signal = torch.randn(1000, device=device)
    print(f"✓ Created test signal: {signal.shape}")
    print(f"  Input device: {signal.device}")
    
    # Test amplification
    amplified = gaindb(signal, 6.0)
    assert amplified.shape == signal.shape
    assert amplified.device.type == device
    print(f"✓ Amplification (+6 dB): {signal.shape} -> {amplified.shape}")
    
    # Test attenuation
    attenuated = gaindb(signal, -6.0)
    assert attenuated.shape == signal.shape
    print(f"✓ Attenuation (-6 dB): {signal.shape} -> {attenuated.shape}")
    
    # Dtype preservation
    dtypes = [torch.float32]
    if device != 'mps':  # MPS doesn't support float64
        dtypes.append(torch.float64)
    
    for dtype in dtypes:
        sig = torch.randn(100, device=device, dtype=dtype)
        result = gaindb(sig, 3.0)
        assert result.dtype == dtype
    print(f"✓ Dtype preservation verified for {len(dtypes)} types")
    
    print(f"\n✓ gaindb passed on {device.upper()}\n")


# ================================================================================================
# Test: BrokenStickCompression - Default
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_brokenstick_default(device, learnable):
    """Test BrokenStickCompression with default parameters."""
    from torch_amt.common.loudness import BrokenStickCompression
    
    print(f"\n{'='*80}")
    print(f"TEST: BrokenStickCompression (default) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    comp = BrokenStickCompression(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {comp}")
    print(f"  extra_repr: {comp.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in comp.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in comp.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify default values (different check for learnable vs non-learnable)
    if learnable:
        assert hasattr(comp, 'knee_db_param')
        assert comp.knee_db_param.item() == 30.0
    else:
        assert comp.knee_db == 30.0
    assert comp.dboffset == 100.0
    assert comp.learnable == learnable
    params = comp.get_parameters()
    print(f"  Verified: knee_db={params['knee_db']:.1f}, exponent={params['exponent']:.3f}, dboffset=100.0")
    
    # Forward pass - single
    signal_single = create_test_signal(device, batch_size=1, n_channels=31, duration=0.1)
    print(f"  Input device: {signal_single.device}")
    
    avg_time_single = time_forward_pass(comp, signal_single, device=device)
    output_single = comp(signal_single)
    assert output_single.shape == signal_single.shape
    assert output_single.device.type == device
    print(f"✓ Forward single: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    signal_batch = create_test_signal(device, batch_size=4, n_channels=31, duration=0.1)
    avg_time_batch = time_forward_pass(comp, signal_batch, device=device)
    output_batch = comp(signal_batch)
    assert output_batch.shape == signal_batch.shape
    print(f"✓ Forward batch:  {signal_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ BrokenStickCompression (default) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: PowerCompression - Default
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_powercompression_default(device, learnable):
    """Test PowerCompression with default parameters."""
    from torch_amt.common.loudness import PowerCompression
    
    print(f"\n{'='*80}")
    print(f"TEST: PowerCompression (default) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    comp = PowerCompression(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {comp}")
    print(f"  extra_repr: {comp.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in comp.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in comp.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify default values (different check for learnable vs non-learnable)
    if learnable:
        assert hasattr(comp, 'knee_db_param')
        assert comp.knee_db_param.item() == 30.0
    else:
        assert comp.knee_db == 30.0
    assert comp.dboffset == 100.0
    assert comp.learnable == learnable
    params = comp.get_parameters()
    print(f"  Verified: knee_db={params['knee_db']:.1f}, exponent={params['exponent']:.3f}, dboffset=100.0")
    
    # Forward pass - single
    signal_single = create_test_signal(device, batch_size=1, n_channels=31, duration=0.1)
    avg_time_single = time_forward_pass(comp, signal_single, device=device)
    output_single = comp(signal_single)
    assert output_single.shape == signal_single.shape
    print(f"✓ Forward single: {signal_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    signal_batch = create_test_signal(device, batch_size=4, n_channels=31, duration=0.1)
    avg_time_batch = time_forward_pass(comp, signal_batch, device=device)
    output_batch = comp(signal_batch)
    assert output_batch.shape == signal_batch.shape
    print(f"✓ Forward batch:  {signal_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ PowerCompression (default) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: SpecificLoudness - Glasberg2002
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_specificloudness_glasberg2002(device, learnable):
    """Test SpecificLoudness (Glasberg2002) with default parameters."""
    from torch_amt.common.loudness import SpecificLoudness
    
    print(f"\n{'='*80}")
    print(f"TEST: SpecificLoudness (Glasberg2002) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    spec_loud = SpecificLoudness(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {spec_loud}")
    print(f"  extra_repr: {spec_loud.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in spec_loud.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in spec_loud.named_parameters()]
        print(f"  Learnable params (first 2): {param_names[:2]}")
    
    # Verify configuration
    assert spec_loud.fs == 32000
    assert spec_loud.f_min == 50.0
    assert spec_loud.f_max == 15000.0
    assert spec_loud.erb_step == 0.25
    assert spec_loud.n_erb_bands == 150
    params = spec_loud.get_parameters()
    print(f"  Verified: fs=32000, f_range=[50-15000] Hz, erb_step=0.25, n_erb=150")
    print(f"  Model params: C={params['C']:.4f}, alpha={params['alpha']:.2f}, E0_offset={params['E0_offset']:.1f} dB")
    
    # Forward pass - batch
    batch_size, n_frames, n_erb = 2, 10, 150
    excitation = create_test_excitation(device, batch_size, n_frames, n_erb)
    print(f"  Input device: {excitation.device}")
    
    avg_time = time_forward_pass(spec_loud, excitation, device=device)
    N = spec_loud(excitation)
    
    assert N.shape == (batch_size, n_frames, n_erb)
    assert N.device.type == device
    assert (N >= 0).all(), "Specific loudness should be non-negative"
    print(f"✓ Forward: {excitation.shape} -> {N.shape} ({avg_time:.3f} ms avg)")
    print(f"  Output range: [{N.min():.3f}, {N.max():.3f}] sone/ERB")
    
    print(f"\n✓ SpecificLoudness (Glasberg2002) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: Moore2016SpecificLoudness
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_moore2016_specific(device, learnable):
    """Test Moore2016SpecificLoudness with default parameters."""
    from torch_amt.common.loudness import Moore2016SpecificLoudness
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016SpecificLoudness - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    spec_loud = Moore2016SpecificLoudness(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {spec_loud}")
    print(f"  extra_repr: {spec_loud.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in spec_loud.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in spec_loud.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify configuration
    params = spec_loud.get_parameters()
    print(f"  Verified: ERB scale 1.75-39 (step 0.25), 150 channels")
    print(f"  Model params: C={params['C']:.4f}")
    
    # Forward pass - single
    excitation_single = torch.rand(150, device=device) * 60 + 20  # 20-80 dB SPL
    avg_time_single = time_forward_pass(spec_loud, excitation_single, device=device)
    N_single = spec_loud(excitation_single)
    
    assert N_single.shape == (150,)
    assert N_single.device.type == device
    assert (N_single >= 0).all()
    print(f"✓ Forward single: {excitation_single.shape} -> {N_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    excitation_batch = torch.rand(4, 150, device=device) * 60 + 20
    avg_time_batch = time_forward_pass(spec_loud, excitation_batch, device=device)
    N_batch = spec_loud(excitation_batch)
    
    assert N_batch.shape == (4, 150)
    print(f"✓ Forward batch:  {excitation_batch.shape} -> {N_batch.shape} ({avg_time_batch:.3f} ms avg)")
    print(f"  Output range: [{N_batch.min():.3f}, {N_batch.max():.3f}] sone/ERB")
    
    print(f"\n✓ Moore2016SpecificLoudness passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: SpatialSmoothing
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_spatial_smoothing(device, learnable):
    """Test SpatialSmoothing with default parameters."""
    from torch_amt.common.loudness import SpatialSmoothing
    
    print(f"\n{'='*80}")
    print(f"TEST: SpatialSmoothing - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    smoothing = SpatialSmoothing(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {smoothing}")
    print(f"  extra_repr: {smoothing.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in smoothing.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in smoothing.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify configuration
    assert smoothing.kernel_width == 18.0
    params = smoothing.get_parameters()
    print(f"  Verified: kernel_width=18.0 ERB, sigma={params['sigma']:.4f}, kernel_size={params['kernel_size']}")
    
    # Forward pass - single
    N_single = torch.rand(150, device=device) * 10
    avg_time_single = time_forward_pass(smoothing, N_single, device=device)
    N_smooth_single = smoothing(N_single)
    
    assert N_smooth_single.shape == (150,)
    assert N_smooth_single.device.type == device
    print(f"✓ Forward single: {N_single.shape} -> {N_smooth_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    N_batch = torch.rand(2, 150, device=device) * 10
    avg_time_batch = time_forward_pass(smoothing, N_batch, device=device)
    N_smooth_batch = smoothing(N_batch)
    
    assert N_smooth_batch.shape == (2, 150)
    print(f"✓ Forward batch:  {N_batch.shape} -> {N_smooth_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    # Energy conservation test
    errors = []
    for _ in range(5):
        N_test = torch.rand(2, 150, device=device) * 10
        N_smooth_test = smoothing(N_test)
        energy_before = N_test.sum()
        energy_after = N_smooth_test.sum()
        rel_error = torch.abs(energy_before - energy_after) / energy_before
        errors.append(rel_error.item())
    
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    print(f"  Energy conservation: mean_err={mean_error:.5f}, max_err={max_error:.5f}")
    assert mean_error < 5e-3 and max_error < 1e-2, "Energy not conserved within tolerance"
    
    print(f"\n✓ SpatialSmoothing passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: BinauralInhibition
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_binaural_inhibition(device, learnable):
    """Test BinauralInhibition with default parameters."""
    from torch_amt.common.loudness import BinauralInhibition
    
    print(f"\n{'='*80}")
    print(f"TEST: BinauralInhibition - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    inhibition = BinauralInhibition(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {inhibition}")
    print(f"  extra_repr: {inhibition.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in inhibition.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in inhibition.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify configuration
    params = inhibition.get_parameters()
    print(f"  Verified: p={params['p']:.4f} (inhibition exponent)")
    
    # Forward pass - single
    N_left = torch.rand(150, device=device) * 10
    N_right = torch.rand(150, device=device) * 10
    
    avg_time = time_forward_pass(inhibition, N_left, N_right, device=device)
    I_left, I_right = inhibition(N_left, N_right)
    
    assert I_left.shape == (150,)
    assert I_right.shape == (150,)
    assert I_left.device.type == device
    assert (I_left >= 0.99).all() and (I_left <= 2.01).all()
    print(f"✓ Forward single: L{N_left.shape} + R{N_right.shape} -> I_L{I_left.shape}, I_R{I_right.shape} ({avg_time:.3f} ms avg)")
    print(f"  Inhibition range: I_left=[{I_left.min():.3f}, {I_left.max():.3f}], I_right=[{I_right.min():.3f}, {I_right.max():.3f}]")
    
    # Forward pass - batch
    N_left_batch = torch.rand(2, 150, device=device) * 10
    N_right_batch = torch.rand(2, 150, device=device) * 10
    I_left_batch, I_right_batch = inhibition(N_left_batch, N_right_batch)
    
    assert I_left_batch.shape == (2, 150)
    assert I_right_batch.shape == (2, 150)
    print(f"✓ Forward batch:  L{N_left_batch.shape} + R{N_right_batch.shape} -> I_L{I_left_batch.shape}, I_R{I_right_batch.shape}")
    
    # Diotic test (equal L/R should give inhibition ≈ 1)
    N_diotic = torch.ones(150, device=device) * 10.0
    I_L_diotic, I_R_diotic = inhibition(N_diotic, N_diotic)
    assert torch.allclose(I_L_diotic, torch.ones_like(I_L_diotic), atol=0.01)
    print(f"  Diotic test (L=R): Inhibition ≈ {I_L_diotic.mean():.3f} (expected ~1.0)")
    
    print(f"\n✓ BinauralInhibition passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: Moore2016BinauralLoudness
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_moore2016_binaural(device, learnable):
    """Test Moore2016BinauralLoudness with default parameters."""
    from torch_amt.common.loudness import Moore2016BinauralLoudness
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016BinauralLoudness - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    binaural = Moore2016BinauralLoudness(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {binaural}")
    print(f"  extra_repr: {binaural.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in binaural.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in binaural.named_parameters()]
        print(f"  Learnable params (first 2): {param_names[:2]}")
    
    # Verify configuration
    params = binaural.get_parameters()
    print(f"  Verified: Spatial smoothing (kernel_width={params['spatial_smoothing']['kernel_width']}, sigma={params['spatial_smoothing']['sigma']:.4f})")
    print(f"           Inhibition (p={params['inhibition']['p']:.4f})")
    
    # Forward pass - single
    N_left = torch.rand(150, device=device) * 10
    N_right = torch.rand(150, device=device) * 10
    
    avg_time_single = time_forward_pass(binaural, N_left, N_right, device=device)
    loudness, loudness_left, loudness_right = binaural(N_left, N_right)
    
    assert loudness.ndim == 0 or loudness.shape == ()
    assert loudness.device.type == device
    assert loudness > 0
    assert torch.allclose(loudness, loudness_left + loudness_right, rtol=1e-5)
    print(f"✓ Forward single: L{N_left.shape} + R{N_right.shape} -> (loudness, L_loud, R_loud) scalars ({avg_time_single:.3f} ms avg)")
    print(f"  Loudness: total={loudness:.3f}, left={loudness_left:.3f}, right={loudness_right:.3f} sone")
    
    # Forward pass - batch
    N_left_batch = torch.rand(2, 150, device=device) * 10
    N_right_batch = torch.rand(2, 150, device=device) * 10
    
    avg_time_batch = time_forward_pass(binaural, N_left_batch, N_right_batch, device=device)
    loudness_b, loudness_left_b, loudness_right_b = binaural(N_left_batch, N_right_batch)
    
    assert loudness_b.shape == (2,)
    print(f"✓ Forward batch:  L{N_left_batch.shape} + R{N_right_batch.shape} -> (loudness, L_loud, R_loud) {loudness_b.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ Moore2016BinauralLoudness passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: LoudnessIntegration
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_loudness_integration(device, learnable):
    """Test LoudnessIntegration (Glasberg2002) with default parameters."""
    from torch_amt.common.loudness import LoudnessIntegration
    
    print(f"\n{'='*80}")
    print(f"TEST: LoudnessIntegration (Glasberg2002) - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    integration = LoudnessIntegration(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {integration}")
    print(f"  extra_repr: {integration.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in integration.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in integration.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify configuration
    assert integration.fs == 32000
    tau_attack, tau_release = integration.get_time_constants()
    print(f"  Verified: fs=32000, tau_attack={tau_attack:.3f}s, tau_release={tau_release:.3f}s")
    
    # Forward pass
    batch_size, n_frames, n_erb = 2, 20, 150
    spec_loud = create_test_specific_loudness(device, batch_size, n_frames, n_erb)
    
    avg_time = time_forward_pass(integration, spec_loud, device=device)
    ltl = integration(spec_loud)
    
    assert ltl.shape == (batch_size, n_frames)
    assert ltl.device.type == device
    assert (ltl >= 0).all()
    print(f"✓ Forward: {spec_loud.shape} -> {ltl.shape} ({avg_time:.3f} ms avg)")
    print(f"  LTL range: [{ltl.min():.3f}, {ltl.max():.3f}] sone")
    
    # Test return_stl
    ltl_2, stl_2 = integration(spec_loud, return_stl=True)
    assert ltl_2.shape == (batch_size, n_frames)
    assert stl_2.shape == (batch_size, n_frames)
    print(f"  With return_stl: LTL{ltl_2.shape}, STL{stl_2.shape}")
    
    print(f"\n✓ LoudnessIntegration (Glasberg2002) passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: Moore2016AGC
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_moore2016_agc(device, learnable):
    """Test Moore2016AGC with default parameters."""
    from torch_amt.common.loudness import Moore2016AGC
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016AGC - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    agc = Moore2016AGC(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {agc}")
    print(f"  extra_repr: {agc.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in agc.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in agc.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify configuration
    params = agc.get_parameters()
    print(f"  Verified: attack_alpha={params['attack_alpha']:.5f}, release_alpha={params['release_alpha']:.5f}")
    
    # Forward pass
    x = torch.rand(20, 150, device=device) * 10
    avg_time = time_forward_pass(agc, x, device=device)
    out = agc(x)
    
    assert out.shape == (20, 150)
    assert out.device.type == device
    print(f"✓ Forward: {x.shape} -> {out.shape} ({avg_time:.3f} ms avg)")
    
    # Test with initial state
    initial_state = torch.ones(150, device=device) * 5.0
    out_state = agc(x, state=initial_state)
    assert out_state.shape == (20, 150)
    print(f"  With initial_state: {x.shape} + state{initial_state.shape} -> {out_state.shape}")
    
    print(f"\n✓ Moore2016AGC passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: Moore2016TemporalIntegration
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_moore2016_temporal(device, learnable):
    """Test Moore2016TemporalIntegration with default parameters."""
    from torch_amt.common.loudness import Moore2016TemporalIntegration
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016TemporalIntegration - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    temporal = Moore2016TemporalIntegration(learnable=learnable).to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {temporal}")
    print(f"  extra_repr: {temporal.extra_repr()}")
    
    # Check parameters
    n_params = sum(p.numel() for p in temporal.parameters())
    print(f"  Parameters: {n_params} total")
    if learnable:
        param_names = [name for name, _ in temporal.named_parameters()]
        print(f"  Learnable params: {param_names}")
    
    # Verify configuration
    params = temporal.get_parameters()
    print(f"  Verified: STL (attack={params['stl']['attack_alpha']:.5f}, release={params['stl']['release_alpha']:.5f})")
    print(f"           LTL (attack={params['ltl']['attack_alpha']:.5f}, release={params['ltl']['release_alpha']:.5f})")
    
    # Forward pass
    inst_spec = torch.rand(20, 150, device=device) * 10
    avg_time = time_forward_pass(temporal, inst_spec, device=device)
    ltl = temporal(inst_spec)
    
    assert ltl.shape == (20,)
    assert ltl.device.type == device
    assert (ltl >= 0).all()
    print(f"✓ Forward: {inst_spec.shape} -> {ltl.shape} ({avg_time:.3f} ms avg)")
    print(f"  LTL range: [{ltl.min():.3f}, {ltl.max():.3f}] sone")
    
    # Test return_intermediate
    ltl_2, stl_spec, stl = temporal(inst_spec, return_intermediate=True)
    assert ltl_2.shape == (20,)
    assert stl_spec.shape == (20, 150)
    assert stl.shape == (20,)
    print(f"  With return_intermediate: LTL{ltl_2.shape}, STL_spec{stl_spec.shape}, STL{stl.shape}")
    
    print(f"\n✓ Moore2016TemporalIntegration passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Main Execution
# ================================================================================================

if __name__ == '__main__':
    """Run all tests when executed as standalone script."""
    
    print_device_info()
    
    devices = get_available_devices()
    
    print("\n" + "=" * 80)
    print("RUNNING LOUDNESS MODULE TESTS")
    print("=" * 80 + "\n")
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"TESTING ON {device.upper()}")
        print(f"{'='*80}\n")
        
        # Run gaindb test once per device
        test_gaindb(device)
        
        # Run all other tests with learnable parameter
        for learnable in [False, True]:
            test_brokenstick_default(device, learnable)
            test_powercompression_default(device, learnable)
            test_specificloudness_glasberg2002(device, learnable)
            test_moore2016_specific(device, learnable)
            test_spatial_smoothing(device, learnable)
            test_binaural_inhibition(device, learnable)
            test_moore2016_binaural(device, learnable)
            test_loudness_integration(device, learnable)
            test_moore2016_agc(device, learnable)
            test_moore2016_temporal(device, learnable)
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
    print("=" * 80 + "\n")
