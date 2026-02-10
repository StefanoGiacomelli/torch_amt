"""
Profile torch_amt common submodules for performance analysis.

This script profiles all submodules in torch_amt.common/ across different devices
(CPU, CUDA, MPS) to measure forward pass timing, memory usage, and training overhead.

Author: Stefano Giacomelli
"""

import torch
import torch.nn as nn
import torch_amt
import pandas as pd
import numpy as np
from scipy import stats
import time
import gc
from pathlib import Path
from typing import Dict, Tuple, List, Any, Union

# ============================================================================
# CONFIGURATION
# ============================================================================

# Devices to profile (comment out unavailable devices)
DEVICES = [
    'cpu',
    # 'cuda:0',  # Uncomment if CUDA available
    # 'mps:0',   # Uncomment if Apple Silicon MPS available
]

# Profiling parameters
N_WARMUP_RUNS = 2  # Warmup runs for GPU (discarded)
N_PROFILE_RUNS = 10  # Number of profiling runs per configuration
SIGNAL_DURATION = 1.0  # Signal duration in seconds

# Output directory
OUTPUT_DIR = Path(__file__).parent  # Same folder as script
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODULE CONFIGURATIONS
# ============================================================================
# Default initialization parameters for each module
# Comment out any module you don't want to profile

MODULE_CONFIGS = {
    # Filterbanks
    'GammatoneFilterbank': {
        'fs': 48000,
        'fc': (80.0, 8000.0),
        'n': 4,
        'betamul': None,
        'learnable': True,
    },
    'ERBIntegration': {
        'fs': 32000,
        'f_min': 50.0,
        'f_max': 15000.0,
        'erb_step': 0.25,
        'learnable': True,
    },
    # 'DRNLFilterbank': {  # Commented by user - too slow
    #     'fc': (250.0, 8000.0),
    #     'fs': 48000,
    #     'n_channels': 50,
    #     'learnable': True,
    # },
    'FastDRNLFilterbank': {
        'fc': (250.0, 8000.0),
        'fs': 48000,
        'n_channels': 50,
        'ir_length': 4096,
        'learnable': True,
    },
    'MultiResolutionFFT': {
        'fs': 32000,
        'learnable': True,
    },
    'Moore2016Spectrum': {
        'fs': 32000,
        'learnable': True,
    },
    'ExcitationPattern': {
        'learnable': True,
    },
    'Moore2016ExcitationPattern': {
        'learnable': True,
    },
    
    # IHC Models
    'IHCEnvelope': {
        'fs': 48000,
        'method': 'dau1996',
        'learnable': True,
    },
    'IHCPaulick2024': {
        'fs': 48000,
        'learnable': True,
    },
    
    # Adaptation
    'AdaptLoop': {
        'fs': 48000,
        'limit': 5,
        'learnable': True,
    },
    
    # Modulation Filterbanks
    # 'ModulationFilterbank': {
    #     'fs': 48000,
    #     'preset': 'dau1997',
    #     'learnable': True,
    # },
    'FastModulationFilterbank': {
        'fs': 48000,
        'preset': 'dau1997',
        'learnable': True,
    },
    # 'King2019ModulationFilterbank': {
    #     'fs': 48000,
    #     'mflow': 2.0,
    #     'mfhigh': 150.0,
    #     'qfactor': 1.0,
    #     'learnable': True,
    # },
    'FastKing2019ModulationFilterbank': {
        'fs': 48000,
        'mflow': 2.0,
        'mfhigh': 150.0,
        'qfactor': 1.0,
        'learnable': True,
    },
    
    # Ears
    'HeadphoneFilter': {
        'fs': 48000,
        'learnable': True,
    },
    'MiddleEarFilter': {
        'fs': 48000,
        'learnable': True,
    },
    'OuterMiddleEarFilter': {
        'fs': 48000,
        'learnable': True,
    },
    
    # Loudness Processing
    'BrokenStickCompression': {
        'knee_db': 30.0,
        'exponent': 0.3,
        'dboffset': 100.0,
        'learnable': True,
    },
    'PowerCompression': {
        'exponent': 0.3,
        'learnable': True,
    },
    'SpecificLoudness': {
        'fs': 32000,
        'f_min': 50.0,
        'f_max': 15000.0,
        'erb_step': 0.25,
        'learnable': True,
    },
    'Moore2016SpecificLoudness': {
        'learnable': True,
    },
    'SpatialSmoothing': {
        'kernel_width': 18.0,
        'sigma': 0.08,
        'learnable': True,
    },
    'BinauralInhibition': {
        'p': 1.5978,
        'learnable': True,
    },
    'Moore2016BinauralLoudness': {
        'kernel_width': 18.0,
        'sigma': 0.08,
        'p': 1.5978,
        'learnable': True,
    },
    'LoudnessIntegration': {
        'learnable': True,
    },
    'Moore2016AGC': {
        'learnable': True,
    },
    'Moore2016TemporalIntegration': {
        'learnable': True,
    },
}

# ============================================================================
# INPUT SHAPE SPECIFICATIONS
# ============================================================================

def get_input_shape(module_name: str, config: Dict[str, Any], device: str) -> Tuple[Union[torch.Tensor, Tuple], int]:
    """
    Generate appropriate input tensor for a given module.
    
    Returns:
        tuple: (input_tensor_or_tuple, num_samples)
    """
    fs = config.get('fs', 48000)
    n_samples = int(fs * SIGNAL_DURATION)
    
    # Filterbanks: expect (B, T) mono input
    if module_name in ['GammatoneFilterbank', 
                       'FastDRNLFilterbank', 
                       'HeadphoneFilter', 
                       'MiddleEarFilter', 
                       'OuterMiddleEarFilter']:
        input_tensor = torch.randn(1, n_samples, device=device) * 0.01
        return input_tensor, n_samples
    
    # MultiResolutionFFT: (B, T) mono → returns (psd, freqs)
    elif module_name == 'MultiResolutionFFT':
        input_tensor = torch.randn(1, n_samples, device=device) * 0.01
        return input_tensor, n_samples
    
    # ERBIntegration: (psd, freqs) - psd: (B, n_frames, n_freq_bins), freqs: (n_freq_bins,)
    elif module_name == 'ERBIntegration':
        # Simulate MultiResolutionFFT output
        n_frames = 100
        n_freq_bins = 2048
        psd = torch.randn(1, n_frames, n_freq_bins, device=device).abs() * 0.01
        freqs = torch.linspace(0, 16000, n_freq_bins, device=device)
        return (psd, freqs), n_samples
    
    # Moore2016Spectrum: (B, 2, T) BINAURAL → returns (freqs_l, levels_l, freqs_r, levels_r)
    elif module_name == 'Moore2016Spectrum':
        input_tensor = torch.randn(1, 2, n_samples, device=device) * 0.01
        return input_tensor, n_samples
    
    # ExcitationPattern: (B, n_frames, n_erb_bands) - expects ERBIntegration-like output
    elif module_name == 'ExcitationPattern':
        # Simulate ERBIntegration output: (batch, n_frames, 150 ERB bands)
        # For 1 second at 32kHz with 32 sample hop: ~936 frames, but use 100 for simplicity
        n_frames = 100
        n_erb_bands = 150
        input_tensor = torch.randn(1, n_frames, n_erb_bands, device=device).abs() * 80.0 + 40.0  # 40-120 dB SPL range
        return input_tensor, n_samples
    
    # Moore2016ExcitationPattern: (freqs, levels) - sparse spectrum
    elif module_name == 'Moore2016ExcitationPattern':
        # Simulate sparse spectrum with 100 frequency components
        n_components = 100
        freqs = torch.linspace(500, 8000, n_components, device=device).unsqueeze(0)  # (1, 100)
        levels = torch.randn(1, n_components, device=device).abs() * 20 + 50.0  # 50-70 dB SPL
        return (freqs, levels), n_samples
    
    # Modules expecting filterbank output (B, F, T)
    elif module_name in ['IHCEnvelope', 'IHCPaulick2024', 'AdaptLoop',
                          'ModulationFilterbank', 'FastModulationFilterbank',
                          'King2019ModulationFilterbank', 
                          'FastKing2019ModulationFilterbank',
                          'BrokenStickCompression', 'PowerCompression',
                          'LoudnessIntegration']:
        F = 31  # Number of frequency channels
        input_tensor = torch.randn(1, F, n_samples, device=device) * 0.01
        return input_tensor, n_samples
    
    # Moore2016AGC: (n_frames, n_channels) - NO batch dimension
    elif module_name == 'Moore2016AGC':
        n_frames = 100
        n_channels = 150
        input_tensor = torch.randn(n_frames, n_channels, device=device).abs() * 5.0
        return input_tensor, n_samples
    
    # Moore2016TemporalIntegration: (n_frames, 150) - NO batch dimension
    elif module_name == 'Moore2016TemporalIntegration':
        n_frames = 100
        input_tensor = torch.randn(n_frames, 150, device=device).abs() * 5.0
        return input_tensor, n_samples
    
    # SpatialSmoothing: (B, 150) ERB channels only
    elif module_name == 'SpatialSmoothing':
        input_tensor = torch.randn(1, 150, device=device).abs() * 5.0
        return input_tensor, n_samples
    
    # SpecificLoudness: (B, n_frames, n_erb_bands) - expects excitation pattern
    elif module_name == 'SpecificLoudness':
        n_frames = 100
        n_erb_bands = 150
        input_tensor = torch.randn(1, n_frames, n_erb_bands, device=device).abs() * 40.0 + 40.0  # 40-80 dB SPL
        return input_tensor, n_samples
    
    # Moore2016SpecificLoudness: (B, 150) - single frame excitation in dB
    elif module_name == 'Moore2016SpecificLoudness':
        input_tensor = torch.randn(1, 150, device=device).abs() * 40.0 + 40.0  # 40-80 dB SPL
        return input_tensor, n_samples
    
    # BinauralInhibition: (left, right) both (B, 150) → returns (inhib_left, inhib_right)
    elif module_name == 'BinauralInhibition':
        left = torch.randn(1, 150, device=device).abs() * 5.0
        right = torch.randn(1, 150, device=device).abs() * 5.0
        return (left, right), n_samples
    
    # Moore2016BinauralLoudness: (specific_loud_left, specific_loud_right) both (B, 150)
    elif module_name == 'Moore2016BinauralLoudness':
        left = torch.randn(1, 150, device=device).abs() * 5.0
        right = torch.randn(1, 150, device=device).abs() * 5.0
        return (left, right), n_samples
    
    else:
        raise ValueError(f"Unknown module: {module_name}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_memory_mb(model: nn.Module) -> float:
    """Calculate model memory from parameters in MB."""
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params / (1024 ** 2)

def get_num_parameters(model: nn.Module) -> int:
    """Get total number of parameters."""
    return sum(p.numel() for p in model.parameters())

def cleanup_memory(device: str):
    """Clean up memory after profiling."""
    gc.collect()
    if 'cuda' in device:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif 'mps' in device:
        torch.mps.empty_cache()
        torch.mps.synchronize()

def profile_forward_pass(model: nn.Module,
                         input_tensor: Union[torch.Tensor, Tuple],
                         device: str,
                         module_name: str,
                         n_runs: int = N_PROFILE_RUNS,
                         warmup: bool = False) -> Tuple[List[float], float]:
    """
    Profile forward pass timing and memory.
    
    Args:
        model: Model to profile
        input_tensor: Input tensor or tuple of tensors
        device: Device string
        module_name: Module name for special handling
        n_runs: Number of runs
        warmup: If True, this is warmup (don't collect stats)
    
    Returns:
        tuple: (times_ms, peak_memory_mb)
    """
    model.eval()
    times = []
    peak_memory = 0.0
    
    # Reset peak memory stats for GPU
    if 'cuda' in device:
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        for _ in range(n_runs):
            # Synchronize before timing
            if 'cuda' in device:
                torch.cuda.synchronize()
            elif 'mps' in device:
                torch.mps.synchronize()
            
            # Time forward pass
            start_time = time.perf_counter()
            
            # Handle tuple input (BinauralInhibition)
            if isinstance(input_tensor, tuple):
                _ = model(*input_tensor)
            else:
                _ = model(input_tensor)
            
            # Synchronize after forward
            if 'cuda' in device:
                torch.cuda.synchronize()
            elif 'mps' in device:
                torch.mps.synchronize()
            
            end_time = time.perf_counter()
            
            if not warmup:
                times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Get peak memory for GPU
    if 'cuda' in device and not warmup:
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    
    return times, peak_memory

def profile_train_step(model: nn.Module,
                       input_tensor: Union[torch.Tensor, Tuple],
                       device: str,
                       module_name: str) -> float:
    """
    Profile complete training step (forward + backward + optimizer).
    
    Returns:
        float: Time in milliseconds, or -1 if no learnable parameters
    """
    model.train()
    
    # Check if model has any learnable parameters
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    if len(learnable_params) == 0:
        return -1.0  # No learnable parameters
    
    # Setup optimizer
    optimizer = torch.optim.SGD(learnable_params, lr=1e-1)
    
    # Synchronize before timing
    if 'cuda' in device:
        torch.cuda.synchronize()
    elif 'mps' in device:
        torch.mps.synchronize()
    
    start_time = time.perf_counter()
    
    try:
        # Training step
        optimizer.zero_grad()
        
        # Handle tuple input (BinauralInhibition)
        if isinstance(input_tensor, tuple):
            output = model(*input_tensor)
        else:
            output = model(input_tensor)
        
        # Handle different output types for loss computation
        if isinstance(output, list):
            # ModulationFilterbank returns list of tensors
            # Concatenate and sum
            loss = torch.cat([o.flatten() for o in output], dim=0).sum()
        elif isinstance(output, tuple):
            # Multi-output: use first element
            # Examples: ExcitationPattern, MultiResolutionFFT, Moore2016Spectrum, etc.
            loss = output[0].sum()
        else:
            # Single tensor output
            loss = output.sum()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Synchronize after step
        if 'cuda' in device:
            torch.cuda.synchronize()
        elif 'mps' in device:
            torch.mps.synchronize()
        
        end_time = time.perf_counter()
        
        return (end_time - start_time) * 1000  # Convert to ms
    
    except RuntimeError as e:
        # Some modules may not support backward (e.g., FastKing2019ModulationFilterbank)
        if "does not require grad" in str(e) or "grad_fn" in str(e):
            return -1.0
        else:
            raise

def compute_statistics(times_ms: List[float]) -> Dict[str, float]:
    """
    Compute timing statistics.
    
    Returns:
        dict: Statistics (mean, std, median, stderr, min, max, p95, skewness, kurtosis)
    """
    times_array = np.array(times_ms)
    
    return {
        'mean': np.mean(times_array),
        'std': np.std(times_array, ddof=1),
        'median': np.median(times_array),
        'stderr': stats.sem(times_array),
        'min': np.min(times_array),
        'max': np.max(times_array),
        'p95': np.percentile(times_array, 95),
        'skewness': stats.skew(times_array),
        'kurtosis': stats.kurtosis(times_array),
    }

def load_existing_results(csv_path: Path) -> List[str]:
    """Load already profiled modules from CSV."""
    if not csv_path.exists():
        return []
    
    try:
        df = pd.read_csv(csv_path)
        return df['module_name'].tolist()
    except Exception as e:
        print(f"Warning: Could not load existing results from {csv_path}: {e}")
        return []

def append_to_csv(results: Dict[str, Any], csv_path: Path):
    """Append results to CSV file."""
    df_new = pd.DataFrame([results])
    
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(csv_path, index=False)
    print(f"  → Results saved to {csv_path}")

# ============================================================================
# MAIN PROFILING FUNCTIONS
# ============================================================================

def profile_module(module_name: str,
                   module_class: type,
                   config: Dict[str, Any],
                   device: str) -> Dict[str, Any]:
    """
    Profile a single module on a single device.
    
    Returns:
        dict: Profiling results
    """
    print(f"\n  Profiling {module_name} on {device}...")
    
    try:
        # Special handling for ModulationFilterbank: need fc parameter
        if module_name in ['ModulationFilterbank', 'FastModulationFilterbank']:
            config = config.copy()
            config['fc'] = torch.linspace(80, 8000, 31)  # 31 ERB-spaced frequencies
        
        # Initialize model
        model = module_class(**config).to(device)
        model.eval()
        
        # Generate input
        input_tensor, n_samples = get_input_shape(module_name, config, device)
        
        # Get model info
        model_memory_mb = get_model_memory_mb(model)
        tot_parameters = get_num_parameters(model)
        
        print(f"    Model: {tot_parameters:,} parameters ({model_memory_mb:.2f} MB)")
        
        # Warmup (only for GPU)
        if device != 'cpu':
            print(f"    Warming up... ({N_WARMUP_RUNS} runs)")
            profile_forward_pass(model, input_tensor, device, module_name, n_runs=N_WARMUP_RUNS, warmup=True)
        
        # Profile forward pass
        print(f"    Profiling forward pass... ({N_PROFILE_RUNS} runs)")
        times_ms, peak_memory_mb = profile_forward_pass(model, input_tensor, device, module_name, n_runs=N_PROFILE_RUNS)
        
        # Compute statistics
        timing_stats = compute_statistics(times_ms)
        
        # Compute throughput
        fs = config.get('fs', 48000)
        throughput = (fs * SIGNAL_DURATION) / (timing_stats['mean'] / 1000)  # samples/sec
        
        # Profile training step
        print(f"    Profiling training step...")
        train_step_ms = profile_train_step(model, input_tensor, device, module_name)
        
        # Compile results
        results = {
            'module_name': module_name,
            'sample_rate': fs,
            'forward_mean_ms': timing_stats['mean'],
            'forward_std_ms': timing_stats['std'],
            'forward_median_ms': timing_stats['median'],
            'forward_stderr_ms': timing_stats['stderr'],
            'forward_min_ms': timing_stats['min'],
            'forward_max_ms': timing_stats['max'],
            'forward_p95_ms': timing_stats['p95'],
            'forward_skewness': timing_stats['skewness'],
            'forward_kurtosis': timing_stats['kurtosis'],
            'throughput_samples_per_sec': throughput,
            'model_memory_mb': model_memory_mb,
            'tot_parameters': tot_parameters,
            'total_train_step_ms': train_step_ms,
        }
        
        # Add peak memory only for CUDA
        if 'cuda' in device:
            results['forward_peak_memory_mb'] = peak_memory_mb
        
        print(f"    ✓ Forward: {timing_stats['mean']:.2f} ± {timing_stats['std']:.2f} ms")
        if train_step_ms >= 0:
            print(f"    ✓ Train step: {train_step_ms:.2f} ms")
        else:
            print(f"    ✓ Train step: N/A (no learnable parameters or not differentiable)")
        print(f"    ✓ Throughput: {throughput:,.0f} samples/sec")
        if 'cuda' in device and peak_memory_mb > 0:
            print(f"    ✓ Peak memory: {peak_memory_mb:.2f} MB")
        
        return results
    
    except Exception as e:
        print(f"    ✗ Error: {e}")
        raise

def main():
    """Main profiling loop."""
    print("=" * 80)
    print("torch_amt Submodules Profiling (Fixed Version)")
    print("=" * 80)
    
    # Verify torch_amt can be imported
    print(f"\ntorch_amt version: {torch_amt.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Display devices to profile
    print(f"\nDevices to profile: {DEVICES}")
    print(f"Signal duration: {SIGNAL_DURATION}s")
    print(f"Profile runs: {N_PROFILE_RUNS}")
    print(f"Warmup runs (GPU): {N_WARMUP_RUNS}")
    
    # Get module classes from torch_amt
    module_classes = {
        'GammatoneFilterbank': torch_amt.GammatoneFilterbank,
        'ERBIntegration': torch_amt.ERBIntegration,
        'FastDRNLFilterbank': torch_amt.FastDRNLFilterbank,
        'MultiResolutionFFT': torch_amt.MultiResolutionFFT,
        'Moore2016Spectrum': torch_amt.Moore2016Spectrum,
        'ExcitationPattern': torch_amt.ExcitationPattern,
        'Moore2016ExcitationPattern': torch_amt.Moore2016ExcitationPattern,
        'IHCEnvelope': torch_amt.IHCEnvelope,
        'IHCPaulick2024': torch_amt.IHCPaulick2024,
        'AdaptLoop': torch_amt.AdaptLoop,
        'ModulationFilterbank': torch_amt.ModulationFilterbank,
        'FastModulationFilterbank': torch_amt.FastModulationFilterbank,
        'King2019ModulationFilterbank': torch_amt.King2019ModulationFilterbank,
        'FastKing2019ModulationFilterbank': torch_amt.FastKing2019ModulationFilterbank,
        'HeadphoneFilter': torch_amt.HeadphoneFilter,
        'MiddleEarFilter': torch_amt.MiddleEarFilter,
        'OuterMiddleEarFilter': torch_amt.OuterMiddleEarFilter,
        'BrokenStickCompression': torch_amt.BrokenStickCompression,
        'PowerCompression': torch_amt.PowerCompression,
        'SpecificLoudness': torch_amt.SpecificLoudness,
        'Moore2016SpecificLoudness': torch_amt.Moore2016SpecificLoudness,
        'SpatialSmoothing': torch_amt.SpatialSmoothing,
        'BinauralInhibition': torch_amt.BinauralInhibition,
        'Moore2016BinauralLoudness': torch_amt.Moore2016BinauralLoudness,
        'LoudnessIntegration': torch_amt.LoudnessIntegration,
        'Moore2016AGC': torch_amt.Moore2016AGC,
        'Moore2016TemporalIntegration': torch_amt.Moore2016TemporalIntegration,
    }
    
    # Profile each device
    for device_str in DEVICES:
        print(f"\n{'=' * 80}")
        print(f"DEVICE: {device_str}")
        print(f"{'=' * 80}")
        
        # Setup CSV path
        csv_filename = device_str.replace(':', '_') + '_profile.csv'
        csv_path = OUTPUT_DIR / csv_filename
        
        # Load existing results
        existing_modules = load_existing_results(csv_path)
        print(f"\nAlready profiled: {len(existing_modules)} modules")
        
        # Profile each module
        for module_name in MODULE_CONFIGS.keys():
            # Skip if not in module_classes (commented out)
            if module_name not in module_classes:
                print(f"\n  ⊘ {module_name} skipped (not in module_classes)")
                continue
            
            # Skip if already profiled
            if module_name in existing_modules:
                print(f"\n  ✓ {module_name} already profiled, skipping")
                continue
            
            try:
                # Profile module
                config = MODULE_CONFIGS[module_name]
                module_class = module_classes[module_name]
                results = profile_module(module_name, module_class, config, device_str)
                
                # Save results
                append_to_csv(results, csv_path)
                
            except Exception as e:
                print(f"  ✗ Failed to profile {module_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            finally:
                # Clean up memory after each module
                cleanup_memory(device_str)
        
        print(f"\n{'-' * 80}")
        print(f"Device {device_str} profiling complete!")
        print(f"Results saved to: {csv_path}")
    
    print(f"\n{'=' * 80}")
    print("All profiling complete!")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    main()
