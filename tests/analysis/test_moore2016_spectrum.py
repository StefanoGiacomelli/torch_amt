"""
Moore2016Spectrum - Test Suite

Tests the 6-window multi-resolution FFT analysis for Moore2016 binaural loudness.

Test structure:
1. test_window_creation: Verifies 6 Hann windows are correctly centered
2. test_frequency_bands: Tests frequency band limiting per window
3. test_relevant_filtering: Tests relevant component filtering thresholds
4. test_stereo_processing: Tests left/right channel separation
5. test_tone_analysis: Tests with pure tones at known frequencies

Figures generated:
- moore2016_spectrum_windows.png: 6 Hann windows visualization
- moore2016_spectrum_bands.png: Frequency band assignments
- moore2016_spectrum_example.png: Multi-resolution spectrogram example
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import Moore2016Spectrum


TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'


def test_window_creation():
    """Test that 6 Hann windows are correctly created and centered."""
    print("\n" + "="*80)
    print("TEST 1: HANN WINDOW CREATION")
    print("="*80)
    
    spectrum = Moore2016Spectrum(fs=32000)
    
    print(f"\nWindow lengths: {spectrum.window_lengths}")
    print(f"Hop length: {spectrum.hop_length} samples ({spectrum.hop_length / 32:.1f} ms @ 32kHz)")
    
    # Verify windows exist
    for i, wlen in enumerate(spectrum.window_lengths):
        window = getattr(spectrum, f'window_{i}')
        print(f"\nWindow {i}: {wlen} samples ({wlen / 32:.1f} ms)")
        print(f"  Shape: {window.shape}")
        print(f"  Non-zero samples: {(window > 0).sum().item()}")
        print(f"  Max value: {window.max().item():.4f}")
        
        # Verify Hann window properties
        assert window.shape[0] == 2048, f"Window {i} should be padded to 2048 samples"
        # Hann window has near-zero values at edges, so check approximately
        assert (window > 1e-6).sum() >= wlen - 2, f"Window {i} should have ~{wlen} non-zero samples"
        # Short windows have larger quantization errors, especially 64 and 128 sample windows
        assert abs(window.max().item() - 1.0) < 5e-3, f"Window {i} max should be ~1.0"
    
    print("\n✓ All 6 windows created correctly")


def test_frequency_bands():
    """Test frequency band limiting for each window."""
    print("\n" + "="*80)
    print("TEST 2: FREQUENCY BAND LIMITING")
    print("="*80)
    
    spectrum = Moore2016Spectrum(fs=32000)
    
    print("\nFrequency band assignments:")
    for i, (wlen, limits) in enumerate(zip(spectrum.window_lengths, spectrum.freq_limits)):
        f_low, f_high = limits
        print(f"  Window {i} ({wlen} samples, {wlen/32:.1f} ms): {f_low:.0f} - {f_high:.0f} Hz")
    
    # Expected bands from MATLAB code
    expected = [
        (20, 80),
        (80, 500),
        (500, 1250),
        (1250, 2540),
        (2540, 4050),
        (4050, 15000)
    ]
    
    for i, (exp_low, exp_high) in enumerate(expected):
        act_low = spectrum.freq_limits[i, 0].item()
        act_high = spectrum.freq_limits[i, 1].item()
        assert abs(act_low - exp_low) < 1e-3, f"Window {i} low freq mismatch"
        assert abs(act_high - exp_high) < 1e-3, f"Window {i} high freq mismatch"
    
    print("\n✓ All frequency bands correct")


def test_relevant_filtering():
    """Test relevant component filtering with known thresholds."""
    print("\n" + "="*80)
    print("TEST 3: RELEVANT COMPONENT FILTERING")
    print("="*80)
    
    spectrum = Moore2016Spectrum(fs=32000)
    
    print(f"\nRelevant component thresholds:")
    print(f"  Max - threshold: {spectrum.threshold_max_minus.item()} dB")
    print(f"  Absolute threshold: {spectrum.threshold_absolute.item()} dB SPL")
    
    # Create test signal with multiple tones at known levels
    fs = 32000
    duration = 0.1  # 100 ms
    t = torch.linspace(0, duration, int(fs * duration))
    
    # Two tones: 1000 Hz @ 0 dB, 3000 Hz @ -70 dB
    # The -70 dB tone should be filtered out by the -60 dB threshold
    signal = (torch.sin(2 * np.pi * 1000 * t) + 
              0.0003162 * torch.sin(2 * np.pi * 3000 * t))  # -70 dB
    
    # Make stereo (same signal both channels)
    signal_stereo = signal.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)  # (1, 2, samples)
    
    # Process
    freqs_l, levels_l, freqs_r, levels_r = spectrum(signal_stereo)
    
    print(f"\nOutput shape:")
    print(f"  freqs_left: {freqs_l.shape}")
    print(f"  levels_left: {levels_l.shape}")
    
    # Check that we get components around 1000, 2000 Hz but not 3000 Hz
    # (3000 Hz @ -70 dB should be filtered out by -60 dB threshold)
    freqs_flat = freqs_l[0].flatten()
    levels_flat = levels_l[0].flatten()
    
    # Remove zeros (padding)
    mask = freqs_flat > 0
    freqs_detected = freqs_flat[mask]
    levels_detected = levels_flat[mask]
    
    print(f"\nDetected {len(freqs_detected)} relevant components")
    print(f"  Frequency range: {freqs_detected.min():.1f} - {freqs_detected.max():.1f} Hz")
    print(f"  Level range: {levels_detected.min():.1f} - {levels_detected.max():.1f} dB")
    
    # Should detect 1000 Hz peak but not 3000 Hz
    has_1000 = ((freqs_detected >= 900) & (freqs_detected <= 1100)).any()
    has_3000 = ((freqs_detected >= 2900) & (freqs_detected <= 3100)).any()
    
    print(f"\n  1000 Hz peak detected: {has_1000}")
    print(f"  3000 Hz peak detected: {has_3000} (should be False)")
    
    assert has_1000, "Should detect 1000 Hz peak"
    assert not has_3000, "Should NOT detect 3000 Hz peak (below threshold)"
    
    print("\n✓ Relevant component filtering works correctly")


def test_stereo_processing():
    """Test that left and right channels are processed independently."""
    print("\n" + "="*80)
    print("TEST 4: STEREO PROCESSING")
    print("="*80)
    
    spectrum = Moore2016Spectrum(fs=32000)
    
    fs = 32000
    duration = 0.1
    t = torch.linspace(0, duration, int(fs * duration))
    
    # Different tones in each ear
    left_signal = torch.sin(2 * np.pi * 500 * t)  # 500 Hz in left
    right_signal = torch.sin(2 * np.pi * 2000 * t)  # 2000 Hz in right
    
    # Combine to stereo
    signal_stereo = torch.stack([left_signal, right_signal], dim=0).unsqueeze(0)  # (1, 2, samples)
    
    # Process
    freqs_l, levels_l, freqs_r, levels_r = spectrum(signal_stereo)
    
    # Extract detected frequencies (remove padding)
    freqs_l_detected = freqs_l[0][freqs_l[0] > 0]
    freqs_r_detected = freqs_r[0][freqs_r[0] > 0]
    
    print(f"\nLeft channel:")
    print(f"  Detected frequencies: {freqs_l_detected[:10].tolist()[:10]}")
    
    print(f"\nRight channel:")
    print(f"  Detected frequencies: {freqs_r_detected[:10].tolist()[:10]}")
    
    # Left should have peak around 500 Hz
    has_500_left = ((freqs_l_detected >= 450) & (freqs_l_detected <= 550)).any()
    has_2000_left = ((freqs_l_detected >= 1900) & (freqs_l_detected <= 2100)).any()
    
    # Right should have peak around 2000 Hz
    has_500_right = ((freqs_r_detected >= 450) & (freqs_r_detected <= 550)).any()
    has_2000_right = ((freqs_r_detected >= 1900) & (freqs_r_detected <= 2100)).any()
    
    print(f"\n  Left: 500 Hz={has_500_left}, 2000 Hz={has_2000_left} (should be True, False)")
    print(f"  Right: 500 Hz={has_500_right}, 2000 Hz={has_2000_right} (should be False, True)")
    
    assert has_500_left and not has_2000_left, "Left channel should only have 500 Hz"
    assert not has_500_right and has_2000_right, "Right channel should only have 2000 Hz"
    
    print("\n✓ Left and right channels processed independently")


def test_visualization():
    """Generate visualization of multi-resolution spectrum."""
    print("\n" + "="*80)
    print("TEST 5: VISUALIZATION")
    print("="*80)
    
    spectrum = Moore2016Spectrum(fs=32000)
    
    # Create test signal: logarithmic chirp from 100 to 8000 Hz
    # Logarithmic chirp ensures equal energy per octave
    fs = 32000
    duration = 0.5  # 500 ms
    t = torch.linspace(0, duration, int(fs * duration))
    
    f0 = 100
    f1 = 8000
    
    # Logarithmic chirp: f(t) = f0 * (f1/f0)^(t/T)
    # Ensures equal time spent per logarithmic frequency bin
    instantaneous_freq = f0 * torch.pow(torch.tensor(f1 / f0), t / duration)
    
    # Phase for logarithmic chirp: φ(t) = 2π * f0 * T / ln(f1/f0) * [(f1/f0)^(t/T) - 1]
    log_ratio = np.log(f1 / f0)
    phase = 2 * np.pi * f0 * duration / log_ratio * (torch.pow(torch.tensor(f1 / f0), t / duration) - 1)
    
    # High amplitude to ensure all frequencies pass -30 dB SPL threshold
    chirp = 5.0 * torch.sin(phase)
    
    # Make stereo
    signal_stereo = chirp.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    
    # Process
    freqs_l, levels_l, freqs_r, levels_r = spectrum(signal_stereo)
    
    print(f"\nProcessed signal:")
    print(f"  Duration: {duration * 1000:.0f} ms")
    print(f"  Time segments: {freqs_l.shape[1]}")
    print(f"  Max components per segment: {freqs_l.shape[2]}")
    
    # Check detected frequency range
    all_freqs_l = freqs_l[0][freqs_l[0] > 0].cpu().numpy()
    print(f"\n  Detected frequency range (Left): {all_freqs_l.min():.1f} - {all_freqs_l.max():.1f} Hz")
    print(f"  Expected range: {f0} - {f1} Hz")
    print(f"  Coverage: {all_freqs_l.min():.1f}/{f0} to {all_freqs_l.max():.1f}/{f1}")
    
    # Create figure with 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # Plot 1: 6 Hann windows
    ax1 = axes[0, 0]
    for i, wlen in enumerate(spectrum.window_lengths):
        window = getattr(spectrum, f'window_{i}').cpu().numpy()
        time_ms = np.arange(len(window)) / 32  # @ 32kHz
        ax1.plot(time_ms, window, label=f'{wlen}s ({wlen/32:.0f}ms)', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time (ms)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('6 Hann Windows', fontsize=11)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency band assignments
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i, limits in enumerate(spectrum.freq_limits):
        f_low, f_high = limits
        ax2.barh(i, f_high - f_low, left=f_low, height=0.8, 
                color=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.text((f_low + f_high) / 2, i, f'{int(f_low)}-{int(f_high)} Hz',
                ha='center', va='center', fontsize=9)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels([f'W{i}' for i in range(6)], fontsize=9)
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_title('Frequency Band Assignments', fontsize=11)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Spectrogram (left channel)
    ax3 = axes[1, 0]
    
    # Convert to dense spectrogram for visualization
    time_axis = np.arange(freqs_l.shape[1])
    
    # For each time segment, plot detected frequencies
    for t_idx in range(freqs_l.shape[1]):
        freqs_t = freqs_l[0, t_idx].cpu().numpy()
        levels_t = levels_l[0, t_idx].cpu().numpy()
        
        # Remove padding
        mask = freqs_t > 0
        freqs_t = freqs_t[mask]
        levels_t = levels_t[mask]
        
        if len(freqs_t) > 0:
            sc = ax3.scatter([t_idx] * len(freqs_t), freqs_t, c=levels_t, 
                       cmap='viridis', s=8, vmin=-50, vmax=10, alpha=0.7)
    
    # Add theoretical chirp overlay
    chirp_freq = f0 * np.power(f1 / f0, time_axis / freqs_l.shape[1])
    ax3.plot(time_axis, chirp_freq, 'r-', linewidth=2, alpha=0.8, label='Expected chirp')
    
    ax3.set_xlabel('Time (ms)', fontsize=10)
    ax3.set_ylabel('Frequency (Hz)', fontsize=10)
    ax3.set_title('Multi-Resolution Spectrogram - LEFT', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_ylim(50, 10000)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Level (dB SPL)', fontsize=9)
    
    # Plot 4: Spectrogram (right channel)
    ax4 = axes[1, 1]
    
    # For each time segment, plot detected frequencies
    for t_idx in range(freqs_r.shape[1]):
        freqs_t = freqs_r[0, t_idx].cpu().numpy()
        levels_t = levels_r[0, t_idx].cpu().numpy()
        
        # Remove padding
        mask = freqs_t > 0
        freqs_t = freqs_t[mask]
        levels_t = levels_t[mask]
        
        if len(freqs_t) > 0:
            sc2 = ax4.scatter([t_idx] * len(freqs_t), freqs_t, c=levels_t, 
                       cmap='viridis', s=8, vmin=-50, vmax=10, alpha=0.7)
    
    # Add theoretical chirp overlay
    ax4.plot(time_axis, chirp_freq, 'r-', linewidth=2, alpha=0.8, label='Expected chirp')
    
    ax4.set_xlabel('Time (ms)', fontsize=10)
    ax4.set_ylabel('Frequency (Hz)', fontsize=10)
    ax4.set_title('Multi-Resolution Spectrogram - RIGHT', fontsize=11)
    ax4.set_yscale('log')
    ax4.set_ylim(50, 10000)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    # Add colorbar
    cbar2 = plt.colorbar(sc2, ax=ax4)
    cbar2.set_label('Level (dB SPL)', fontsize=9)
    
    # Plot 5: Component count over time (both channels)
    ax5 = axes[2, 0]
    n_components_l = (freqs_l[0] > 0).sum(dim=1).cpu().numpy()
    n_components_r = (freqs_r[0] > 0).sum(dim=1).cpu().numpy()
    ax5.plot(time_axis, n_components_l, linewidth=2, label='Left', alpha=0.8)
    ax5.plot(time_axis, n_components_r, linewidth=2, label='Right', alpha=0.8, linestyle='--')
    ax5.set_xlabel('Time (ms)', fontsize=10)
    ax5.set_ylabel('N Components', fontsize=10)
    ax5.set_title('Relevant Component Count vs Time', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Level distribution (both channels)
    ax6 = axes[2, 1]
    levels_flat_l = levels_l[0][freqs_l[0] > 0].cpu().numpy()
    levels_flat_r = levels_r[0][freqs_r[0] > 0].cpu().numpy()
    ax6.hist(levels_flat_l, bins=40, alpha=0.5, edgecolor='black', label='Left', color='blue')
    ax6.hist(levels_flat_r, bins=40, alpha=0.5, edgecolor='black', label='Right', color='orange')
    ax6.axvline(spectrum.threshold_absolute.item(), color='r', linestyle='--', linewidth=2,
               label=f'Abs threshold ({spectrum.threshold_absolute.item():.0f} dB)')
    ax6.set_xlabel('Level (dB SPL)', fontsize=10)
    ax6.set_ylabel('Count', fontsize=10)
    ax6.set_title('Component Level Distribution', fontsize=11)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Moore2016Spectrum: Multi-Resolution FFT Analysis (Logarithmic Chirp 100-8000 Hz)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    output_path = TEST_FIGURES_DIR / 'moore2016_spectrum_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    test_window_creation()
    test_frequency_bands()
    test_relevant_filtering()
    test_stereo_processing()
    test_visualization()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
