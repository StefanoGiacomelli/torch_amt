"""Device Compatibility Test Suite for filterbanks.py

This test suite verifies that all functions and classes in filterbanks.py work correctly
across all available devices (CPU, CUDA, MPS).

Contents:
- 7 standalone functions: audfiltbw, erb2fc, fc2erb, f2erb, f2erbrate, erbrate2f, erbspacebw
- 7 nn.Module classes: GammatoneFilterbank, ERBIntegration, DRNLFilterbank, 
  MultiResolutionFFT, Moore2016Spectrum, ExcitationPattern, Moore2016ExcitationPattern

Test structure:
- Initialization with default/minimal parameters
- Application on single input
- Application on batch input
- Device transfer (CPU, CUDA, MPS)
- Module repr and parameter inspection (for classes)

Usage:
    # Standalone execution (tests all available devices)
    python test_device_filterbanks.py
    
    # pytest execution
    pytest test_device_filterbanks.py -v
    
    # pytest execution on specific device
    pytest test_device_filterbanks.py -v -k "cpu"
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

def create_test_audio(device: str, batch_size: int = 1, duration: float = 0.1, fs: int = 16000) -> torch.Tensor:
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
    audio = torch.randn(batch_size, n_samples, device=device)
    return audio


def create_test_frequencies(device: str, n_freqs: int = 3) -> torch.Tensor:
    """Create test frequency values.
    
    Parameters
    ----------
    device : str
        Target device
    n_freqs : int
        Number of frequency values
        
    Returns
    -------
    torch.Tensor
        Frequencies in Hz, shape (n_freqs,)
    """
    freqs = torch.linspace(100, 10000, n_freqs, device=device)
    return freqs


def create_test_fc(device: str, n_channels: int = 3) -> torch.Tensor:
    """Create test center frequencies for filterbanks.
    
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
    # MPS doesn't support logspace, compute on CPU then transfer
    # Using torch.pow(10, torch.linspace(...)) as alternative
    log_low = torch.log10(torch.tensor(500.0))
    log_high = torch.log10(torch.tensor(4000.0))
    fc = torch.pow(10, torch.linspace(log_low, log_high, n_channels, device=device))
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
# Test: Standalone Functions
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
def test_standalone_functions(device):
    """Test all standalone utility functions on specified device.
    
    Functions tested:
    - audfiltbw: Auditory filter bandwidth
    - erb2fc: ERB-rate to frequency
    - fc2erb: Frequency to ERB-rate
    - f2erb: Frequency to ERB bandwidth
    - f2erbrate: Frequency to ERB-rate
    - erbrate2f: ERB-rate to frequency
    - erbspacebw: ERB-spaced frequency grid
    """
    from torch_amt.common.filterbanks import audfiltbw, erb2fc, fc2erb, f2erb, f2erbrate, erbrate2f, erbspacebw
    
    print(f"\n{'='*80}")
    print(f"STANDALONE FUNCTIONS - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Test each function with single and batch inputs
    functions = [('audfiltbw', audfiltbw, create_test_frequencies(device, 1), create_test_frequencies(device, 4)),
                 ('erb2fc', erb2fc, torch.tensor([10.0], device=device), torch.tensor([5.0, 10.0, 20.0, 30.0], device=device)),
                 ('fc2erb', fc2erb, torch.tensor([1000.0], device=device), torch.tensor([500.0, 1000.0, 2000.0, 4000.0], device=device)),
                 ('f2erb', f2erb, torch.tensor([1000.0], device=device), torch.tensor([100.0, 500.0, 1000.0, 5000.0], device=device)),
                 ('f2erbrate', f2erbrate, torch.tensor([1000.0], device=device), torch.tensor([100.0, 500.0, 1000.0, 5000.0], device=device)),
                 ('erbrate2f', erbrate2f, torch.tensor([20.0], device=device), torch.tensor([5.0, 10.0, 20.0, 30.0], device=device)),
                 ]
    
    for func_name, func, input_single, input_batch in functions:
        print(f"Testing: {func_name}")
        
        # Single input
        start = time.time()
        output_single = func(input_single)
        time_single = (time.time() - start) * 1000
        assert output_single.device.type == device.split(':')[0], f"{func_name} output device mismatch"
        print(f"  ✓ Single input: {input_single.shape} -> {output_single.shape} ({time_single:.3f} ms)")
        
        # Batch input
        start = time.time()
        output_batch = func(input_batch)
        time_batch = (time.time() - start) * 1000
        assert output_batch.device.type == device.split(':')[0], f"{func_name} batch output device mismatch"
        print(f"  ✓ Batch input:  {input_batch.shape} -> {output_batch.shape} ({time_batch:.3f} ms)")
    
    # Test erbspacebw (special case: takes scalars, returns tensor)
    print(f"Testing: erbspacebw")
    start = time.time()
    fc_grid = erbspacebw(100.0, 8000.0, 150, device=device)
    time_grid = (time.time() - start) * 1000
    assert fc_grid.device.type == device.split(':')[0], "erbspacebw output device mismatch"
    print(f"  ✓ Output: {fc_grid.shape}, range=[{fc_grid[0]:.1f}, {fc_grid[-1]:.1f}] Hz ({time_grid:.3f} ms)")
    
    print(f"\n✓ All standalone functions passed on {device.upper()}\n")


# ================================================================================================
# Test: GammatoneFilterbank
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_gammatone_filterbank(device, learnable):
    """Test GammatoneFilterbank on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import GammatoneFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: GammatoneFilterbank - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    fs = 16000
    fc = create_test_fc(device, n_channels=5)
    filterbank = GammatoneFilterbank(fc=fc, fs=fs, n=4, learnable=learnable)
    filterbank = filterbank.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {filterbank}")
    print(f"  extra_repr: {filterbank.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in filterbank.parameters())
    param_names = [name for name, _ in filterbank.named_parameters()]
    print(f"  Parameters: {n_params} total, names={param_names}")
    
    # Forward pass - single
    audio_single = create_test_audio(device, batch_size=1, fs=fs)
    print(f"  Input device: {audio_single.device}")
    avg_time_single = time_forward_pass(filterbank, audio_single, device=device.split(':')[0])
    output_single = filterbank(audio_single)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward single: {audio_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    audio_batch = create_test_audio(device, batch_size=4, fs=fs)
    avg_time_batch = time_forward_pass(filterbank, audio_batch, device=device.split(':')[0])
    output_batch = filterbank(audio_batch)
    assert output_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  {audio_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ GammatoneFilterbank passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: ERBIntegration
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_erb_integration(device, learnable):
    """Test ERBIntegration on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import ERBIntegration
    
    print(f"\n{'='*80}")
    print(f"TEST: ERBIntegration - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    # Default: fs=32000, f_min=50.0, f_max=15000.0, erb_step=0.25
    fs = 32000
    integration = ERBIntegration(fs=fs, f_min=50.0, f_max=15000.0, erb_step=0.25, learnable=learnable)
    integration = integration.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {integration}")
    print(f"  extra_repr: {integration.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in integration.parameters())
    param_names = [name for name, _ in integration.named_parameters()]
    print(f"  Parameters: {n_params} total")
    if param_names:
        print(f"  Param names: {param_names[:3]}..." if len(param_names) > 3 else f"  Param names: {param_names}")
    
    # Forward pass - single (needs PSD and freqs)
    n_frames = 10
    n_freqs = 513  # Typical FFT size // 2 + 1
    psd_single = torch.randn(1, n_frames, n_freqs, device=device).abs()
    freqs = torch.linspace(0, fs/2, n_freqs, device=device)
    print(f"  Input devices: psd={psd_single.device}, freqs={freqs.device}")
    
    avg_time_single = time_forward_pass(integration, psd_single, freqs, device=device.split(':')[0])
    output_single = integration(psd_single, freqs)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward single: psd={psd_single.shape}, freqs={freqs.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    psd_batch = torch.randn(4, n_frames, n_freqs, device=device).abs()
    avg_time_batch = time_forward_pass(integration, psd_batch, freqs, device=device.split(':')[0])
    output_batch = integration(psd_batch, freqs)
    assert output_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  psd={psd_batch.shape}, freqs={freqs.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ ERBIntegration passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: DRNLFilterbank
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_drnl_filterbank(device, learnable):
    """Test DRNLFilterbank on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import DRNLFilterbank
    
    print(f"\n{'='*80}")
    print(f"TEST: DRNLFilterbank - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    # MPS doesn't support float64, use float32
    fs = 16000
    dtype = torch.float32 if device == 'mps' else torch.float64
    fc = create_test_fc(device, n_channels=5)
    filterbank = DRNLFilterbank(fc=fc, fs=fs, learnable=learnable, dtype=dtype)
    filterbank = filterbank.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {filterbank}")
    print(f"  extra_repr: {filterbank.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in filterbank.parameters())
    param_names = [name for name, _ in filterbank.named_parameters()]
    print(f"  Parameters: {n_params} total")
    
    # Forward pass - single
    audio_single = create_test_audio(device, batch_size=1, fs=fs)
    print(f"  Input device: {audio_single.device}")
    avg_time_single = time_forward_pass(filterbank, audio_single, device=device.split(':')[0])
    output_single = filterbank(audio_single)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward single: {audio_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    audio_batch = create_test_audio(device, batch_size=4, fs=fs)
    avg_time_batch = time_forward_pass(filterbank, audio_batch, device=device.split(':')[0])
    output_batch = filterbank(audio_batch)
    assert output_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  {audio_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ DRNLFilterbank passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: MultiResolutionFFT
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_multi_resolution_fft(device, learnable):
    """Test MultiResolutionFFT on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import MultiResolutionFFT
    
    print(f"\n{'='*80}")
    print(f"TEST: MultiResolutionFFT - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    # Default: fs=32000, window_lengths=[2048, 1024, 512, 256, 128, 64], hop_fraction=0.5
    fs = 32000
    mrf = MultiResolutionFFT(fs=fs, learnable=learnable)
    mrf = mrf.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {mrf}")
    print(f"  extra_repr: {mrf.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in mrf.parameters())
    param_names = [name for name, _ in mrf.named_parameters()]
    print(f"  Parameters: {n_params} total")
    
    # Forward pass - single
    audio_single = create_test_audio(device, batch_size=1, duration=0.5, fs=fs)
    print(f"  Input device: {audio_single.device}")
    avg_time_single = time_forward_pass(mrf, audio_single, device=device.split(':')[0])
    psd_single, freqs = mrf(audio_single)
    print(f"  Output devices: psd={psd_single.device}, freqs={freqs.device}")
    assert psd_single.device.type == device.split(':')[0]
    assert freqs.device.type == device.split(':')[0]
    print(f"✓ Forward single: {audio_single.shape} -> psd={psd_single.shape}, freqs={freqs.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    audio_batch = create_test_audio(device, batch_size=4, duration=0.5, fs=fs)
    avg_time_batch = time_forward_pass(mrf, audio_batch, device=device.split(':')[0])
    psd_batch, freqs_batch = mrf(audio_batch)
    assert psd_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  {audio_batch.shape} -> psd={psd_batch.shape}, freqs={freqs_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ MultiResolutionFFT passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: Moore2016Spectrum
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_moore2016_spectrum(device, learnable):
    """Test Moore2016Spectrum on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import Moore2016Spectrum
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016Spectrum - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    # REQUIRED: fs=32000 (strictly enforced), default: segment_duration=1, db_max=93.98
    fs = 32000
    spectrum = Moore2016Spectrum(fs=fs, segment_duration=1, db_max=93.98, learnable=learnable)
    spectrum = spectrum.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {spectrum}")
    print(f"  extra_repr: {spectrum.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in spectrum.parameters())
    param_names = [name for name, _ in spectrum.named_parameters()]
    print(f"  Parameters: {n_params} total")
    
    # Forward pass - single (requires BINAURAL audio: (batch, 2, samples))
    audio_single = create_test_audio(device, batch_size=1, duration=0.5, fs=fs)
    audio_single_binaural = audio_single.unsqueeze(1).repeat(1, 2, 1)  # (1, 2, samples)
    print(f"  Input device: {audio_single_binaural.device}")
    
    # Moore2016Spectrum returns 4 values: freqs_left, levels_left, freqs_right, levels_right
    avg_time_single = time_forward_pass(spectrum, audio_single_binaural, device=device.split(':')[0])
    freqs_left, levels_left, freqs_right, levels_right = spectrum(audio_single_binaural)
    print(f"  Output devices: freqs_left={freqs_left.device}, levels_left={levels_left.device}")
    assert freqs_left.device.type == device.split(':')[0]
    assert levels_left.device.type == device.split(':')[0]
    print(f"✓ Forward single: {audio_single_binaural.shape} -> freqs_L={freqs_left.shape}, levels_L={levels_left.shape}, freqs_R={freqs_right.shape}, levels_R={levels_right.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    audio_batch = create_test_audio(device, batch_size=4, duration=0.5, fs=fs)
    audio_batch_binaural = audio_batch.unsqueeze(1).repeat(1, 2, 1)  # (4, 2, samples)
    
    avg_time_batch = time_forward_pass(spectrum, audio_batch_binaural, device=device.split(':')[0])
    freqs_left_b, levels_left_b, freqs_right_b, levels_right_b = spectrum(audio_batch_binaural)
    assert freqs_left_b.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  {audio_batch_binaural.shape} -> freqs_L={freqs_left_b.shape}, levels_L={levels_left_b.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ Moore2016Spectrum passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: ExcitationPattern
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_excitation_pattern(device, learnable):
    """Test ExcitationPattern on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import ExcitationPattern
    
    print(f"\n{'='*80}")
    print(f"TEST: ExcitationPattern - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    # Default: fs=32000, f_min=50.0, f_max=15000.0, erb_step=0.25
    fs = 32000
    pattern = ExcitationPattern(fs=fs, f_min=50.0, f_max=15000.0, erb_step=0.25, learnable=learnable)
    pattern = pattern.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {pattern}")
    print(f"  extra_repr: {pattern.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in pattern.parameters())
    param_names = [name for name, _ in pattern.named_parameters()]
    print(f"  Parameters: {n_params} total, names={param_names}")
    
    # Forward pass - single
    # ExcitationPattern.forward() takes only 1 argument: excitation (batch, n_frames, n_erb_bands)
    n_erb = pattern.n_erb_bands
    n_frames = 32
    excitation_single = torch.randn(1, n_frames, n_erb, device=device) * 10 + 60  # ~60 dB SPL
    print(f"  Input device: {excitation_single.device}")
    
    avg_time_single = time_forward_pass(pattern, excitation_single, device=device.split(':')[0])
    output_single = pattern(excitation_single)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward single: excitation={excitation_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    excitation_batch = torch.randn(4, n_frames, n_erb, device=device) * 10 + 60
    avg_time_batch = time_forward_pass(pattern, excitation_batch, device=device.split(':')[0])
    output_batch = pattern(excitation_batch)
    assert output_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  excitation={excitation_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ ExcitationPattern passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Test: Moore2016ExcitationPattern
# ================================================================================================

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("learnable", [False, True])
def test_moore2016_excitation_pattern(device, learnable):
    """Test Moore2016ExcitationPattern on specified device with learnable parameters."""
    from torch_amt.common.filterbanks import Moore2016ExcitationPattern
    
    print(f"\n{'='*80}")
    print(f"TEST: Moore2016ExcitationPattern - Device: {device.upper()}, Learnable: {learnable}")
    print(f"{'='*80}\n")
    
    # Initialization with default parameters from filterbanks.py
    # Default: erb_lower=1.75, erb_upper=39.0, erb_step=0.25, spreading_limit_octaves=4.0
    pattern = Moore2016ExcitationPattern(erb_lower=1.75, erb_upper=39.0, erb_step=0.25, 
                                         spreading_limit_octaves=4.0, learnable=learnable)
    pattern = pattern.to(device)
    
    print(f"✓ Initialization successful")
    print(f"  Module: {pattern}")
    print(f"  extra_repr: {pattern.extra_repr()}")
    
    # Parameters
    n_params = sum(p.numel() for p in pattern.parameters())
    param_names = [name for name, _ in pattern.named_parameters()]
    print(f"  Parameters: {n_params} total, names={param_names}")
    
    # Forward pass - single (requires sparse spectrum: freqs, levels)
    freqs_single = torch.tensor([[500.0, 1000.0, 2000.0]], device=device)
    levels_single = torch.tensor([[60.0, 65.0, 55.0]], device=device)
    print(f"  Input devices: freqs={freqs_single.device}, levels={levels_single.device}")
    
    avg_time_single = time_forward_pass(pattern, freqs_single, levels_single, device=device.split(':')[0])
    output_single = pattern(freqs_single, levels_single)
    print(f"  Output device: {output_single.device}")
    assert output_single.device.type == device.split(':')[0]
    print(f"✓ Forward single: freqs={freqs_single.shape}, levels={levels_single.shape} -> {output_single.shape} ({avg_time_single:.3f} ms avg)")
    
    # Forward pass - batch
    freqs_batch = torch.tensor([
        [500.0, 1000.0, 2000.0],
        [750.0, 1500.0, 3000.0],
        [600.0, 1200.0, 2400.0],
        [800.0, 1600.0, 3200.0]
    ], device=device)
    levels_batch = torch.tensor([
        [60.0, 65.0, 55.0],
        [58.0, 62.0, 54.0],
        [61.0, 66.0, 56.0],
        [59.0, 64.0, 53.0]
    ], device=device)
    
    avg_time_batch = time_forward_pass(pattern, freqs_batch, levels_batch, device=device.split(':')[0])
    output_batch = pattern(freqs_batch, levels_batch)
    assert output_batch.device.type == device.split(':')[0]
    print(f"✓ Forward batch:  freqs={freqs_batch.shape}, levels={levels_batch.shape} -> {output_batch.shape} ({avg_time_batch:.3f} ms avg)")
    
    print(f"\n✓ Moore2016ExcitationPattern passed on {device.upper()} (learnable={learnable})\n")


# ================================================================================================
# Main: Standalone Execution
# ================================================================================================

def main():
    """Run all tests on all available devices (standalone execution)."""
    print("\n" + "="*80)
    print("FILTERBANKS DEVICE COMPATIBILITY TEST SUITE")
    print("="*80)
    
    # Print device info
    print_device_info()
    
    devices = get_available_devices()
    
    print(f"Running tests on {len(devices)} device(s): {', '.join(devices)}\n")
    
    # Test functions
    test_functions = [
        ("Standalone Functions", test_standalone_functions),
        ("GammatoneFilterbank", test_gammatone_filterbank),
        ("ERBIntegration", test_erb_integration),
        ("DRNLFilterbank", test_drnl_filterbank),
        ("MultiResolutionFFT", test_multi_resolution_fft),
        ("Moore2016Spectrum", test_moore2016_spectrum),
        ("ExcitationPattern", test_excitation_pattern),
        ("Moore2016ExcitationPattern", test_moore2016_excitation_pattern),
    ]
    
    # Run all tests on all devices with both learnable values
    results = {}
    learnable_values = [False, True]
    
    for device in devices:
        print(f"\n{'#'*80}")
        print(f"# TESTING ON DEVICE: {device.upper()}")
        print(f"{'#'*80}\n")
        
        device_results = {}
        for test_name, test_func in test_functions:
            # Skip standalone functions (they don't have learnable parameter)
            if test_name == "Standalone Functions":
                try:
                    test_func(device)
                    device_results[test_name] = "✓ PASSED"
                except Exception as e:
                    device_results[test_name] = f"✗ FAILED: {str(e)[:60]}"
                    print(f"\n✗ {test_name} FAILED on {device}: {e}\n")
            else:
                # Test both learnable=False and learnable=True
                passed = True
                error_msg = ""
                for learnable in learnable_values:
                    try:
                        test_func(device, learnable)
                    except Exception as e:
                        passed = False
                        error_msg = str(e)
                        print(f"\n✗ {test_name} FAILED on {device} (learnable={learnable}): {e}\n")
                        break
                
                if passed:
                    device_results[test_name] = "✓ PASSED"
                else:
                    device_results[test_name] = f"✗ FAILED: {error_msg[:60]}"
        
        results[device] = device_results
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for device in devices:
        print(f"{device.upper()}:")
        for test_name, result in results[device].items():
            print(f"  {test_name:35s}: {result}")
        print()
    
    # Overall status
    all_passed = all(
        "PASSED" in result 
        for device_results in results.values() 
        for result in device_results.values()
    )
    
    if all_passed:
        print("="*80)
        print("✓ ALL TESTS PASSED ON ALL DEVICES")
        print("="*80 + "\n")
        return 0
    else:
        print("="*80)
        print("✗ SOME TESTS FAILED")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    import sys
    
    sys.exit(main())
