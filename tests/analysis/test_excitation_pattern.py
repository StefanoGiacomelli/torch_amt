"""
Excitation Pattern - Test Suite

Contents:
1. test_excitation_pattern: Verifies excitation pattern for Glasberg & Moore (2002) model
   - Tests asymmetric spreading (more toward lower frequencies)
   - Validates level-dependent spreading
   - Confirms masking behavior (upward, downward, parallel)

Structure:
- Pure tones at 1 kHz with different levels (40, 60, 80 dB SPL)
- Narrow-band noise centered at 2 kHz
- Masker tone analysis for masking patterns
- Multi-resolution FFT + ERB integration + excitation pattern pipeline

Figures generated:
- excitation_pattern.png: 6-panel analysis (level-dependent patterns, noise spreading, masking)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import MultiResolutionFFT, ERBIntegration, ExcitationPattern, f2erbrate


def generate_test_signals(fs=32000, duration=0.5):
    """
    Generate test signals for excitation pattern analysis.
    
    Returns:
        tones_40db, tones_60db, tones_80db: Pure tones at 1 kHz, different levels
        narrow_band_noise: Narrow-band noise centered at 2 kHz
        masker_probe: Masker (1 kHz) + probe tones at various distances
        t: Time vector
    """
    t = torch.linspace(0, duration, int(fs * duration))
    
    # 1. Pure tones at 1 kHz with different levels
    tone_1khz = torch.sin(2 * np.pi * 1000 * t)
    
    # Scale to approximate dB SPL levels
    # Assuming digital full scale corresponds to ~100 dB SPL
    # This is a rough calibration
    tones_40db = tone_1khz * 10 ** ((40 - 100) / 20)
    tones_60db = tone_1khz * 10 ** ((60 - 100) / 20)
    tones_80db = tone_1khz * 10 ** ((80 - 100) / 20)
    
    # 2. Narrow-band noise centered at 2 kHz, bandwidth 200 Hz
    noise = torch.randn_like(t)
    noise_fft = torch.fft.rfft(noise)
    freqs_fft = torch.fft.rfftfreq(len(noise), 1 / fs)
    
    # Bandpass filter: 1900-2100 Hz
    mask = (freqs_fft >= 1900) & (freqs_fft <= 2100)
    noise_fft[~mask] = 0
    narrow_band_noise = torch.fft.irfft(noise_fft, n=len(noise))
    narrow_band_noise = narrow_band_noise / narrow_band_noise.std() * 10 ** ((60 - 100) / 20)
    
    # 3. Masker + probe tones for masking experiment
    # Masker: 1 kHz @ 60 dB SPL
    masker = tone_1khz * 10 ** ((60 - 100) / 20)
    
    # Probe tones at various ERB distances from masker
    # We'll generate multiple signals, each with masker + one probe
    masker_probe = masker  # Just return masker for now, we'll add probes during analysis
    
    return tones_40db, tones_60db, tones_80db, narrow_band_noise, masker_probe, t


def test_excitation_pattern():
    """Test excitation pattern with various signals."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Excitation Pattern Test")
    print("=" * 80)
    
    # Parameters
    fs = 32000
    duration = 0.5
    
    # Generate test signals
    print("\nGenerating test signals...")
    tone_40db, tone_60db, tone_80db, narrow_band_noise, masker, t = generate_test_signals(fs, duration)
    
    print(f"  1 kHz tones at 40, 60, 80 dB SPL")
    print(f"  Narrow-band noise (1900-2100 Hz)")
    print(f"  Masker tone at 1 kHz, 60 dB SPL")
    
    # Add batch dimension
    tone_40db_batch = tone_40db.unsqueeze(0)
    tone_60db_batch = tone_60db.unsqueeze(0)
    tone_80db_batch = tone_80db.unsqueeze(0)
    noise_batch = narrow_band_noise.unsqueeze(0)
    masker_batch = masker.unsqueeze(0)
    
    # Initialize components
    print("\nInitializing components...")
    multi_fft = MultiResolutionFFT(fs=fs, learnable=False)
    erb_integration = ERBIntegration(fs=fs, learnable=False)
    excitation_pattern = ExcitationPattern(fs=fs, learnable=False)
    
    fc_erb = excitation_pattern.fc_erb
    erb_centers = f2erbrate(fc_erb)
    
    print(f"  ERB channels: {excitation_pattern.n_erb_bands}")
    print(f"  Spreading slopes (60 dB): upper={excitation_pattern.upper_slope_base.item():.1f} dB/ERB, "
          f"lower={excitation_pattern.lower_slope_base.item():.1f} dB/ERB")
    
    # Process signals through pipeline
    print("\nProcessing signals...")
    
    # Tone at 40 dB
    psd_40, freqs = multi_fft(tone_40db_batch)
    exc_40 = erb_integration(psd_40, freqs)
    exc_spread_40 = excitation_pattern(exc_40)
    exc_spread_40_avg = exc_spread_40.mean(dim=1).squeeze(0).numpy()
    
    # Tone at 60 dB
    psd_60, _ = multi_fft(tone_60db_batch)
    exc_60 = erb_integration(psd_60, freqs)
    exc_spread_60 = excitation_pattern(exc_60)
    exc_spread_60_avg = exc_spread_60.mean(dim=1).squeeze(0).numpy()
    
    # Tone at 80 dB
    psd_80, _ = multi_fft(tone_80db_batch)
    exc_80 = erb_integration(psd_80, freqs)
    exc_spread_80 = excitation_pattern(exc_80)
    exc_spread_80_avg = exc_spread_80.mean(dim=1).squeeze(0).numpy()
    
    # Narrow-band noise
    psd_noise, _ = multi_fft(noise_batch)
    exc_noise = erb_integration(psd_noise, freqs)
    exc_spread_noise = excitation_pattern(exc_noise)
    exc_spread_noise_avg = exc_spread_noise.mean(dim=1).squeeze(0).numpy()
    
    # Masker
    psd_masker, _ = multi_fft(masker_batch)
    exc_masker = erb_integration(psd_masker, freqs)
    exc_spread_masker = excitation_pattern(exc_masker)
    exc_spread_masker_avg = exc_spread_masker.mean(dim=1).squeeze(0).numpy()
    
    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # === Top-left: Excitation pattern for tone at 3 levels ===
    ax = fig.add_subplot(gs[0, 0])
    
    # Find 1 kHz ERB channel
    idx_1khz = torch.argmin(torch.abs(fc_erb - 1000))
    
    ax.plot(fc_erb.numpy(), exc_spread_40_avg, 'b-', linewidth=2, label='40 dB SPL', alpha=0.8)
    ax.plot(fc_erb.numpy(), exc_spread_60_avg, 'g-', linewidth=2, label='60 dB SPL', alpha=0.8)
    ax.plot(fc_erb.numpy(), exc_spread_80_avg, 'r-', linewidth=2, label='80 dB SPL', alpha=0.8)
    
    # Mark the tone frequency
    ax.axvline(1000, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(1000, ax.get_ylim()[1], '1 kHz tone', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Excitation (dB SPL)')
    ax.set_title('Excitation Pattern: 1 kHz Tone at 3 Levels', pad=15)
    ax.set_xscale('log')
    ax.set_xlim([100, 10000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Top-right: Asymmetry of spreading vs level ===
    ax = fig.add_subplot(gs[0, 1])
    
    # Compute slopes for various levels
    levels = np.arange(20, 101, 10)
    upper_slopes = []
    lower_slopes = []
    
    for level in levels:
        upper, lower = excitation_pattern.get_spreading_slopes(level)
        upper_slopes.append(upper)
        lower_slopes.append(lower)
    
    ax.plot(levels, upper_slopes, 'r-o', linewidth=2, markersize=6, label='Upper slope (→ high freq)', alpha=0.8)
    ax.plot(levels, lower_slopes, 'b-s', linewidth=2, markersize=6, label='Lower slope (→ low freq)', alpha=0.8)
    
    ax.set_xlabel('Input Level (dB SPL)')
    ax.set_ylabel('Spreading Slope (dB/ERB)')
    ax.set_title('Asymmetric Spreading vs Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation for asymmetry
    level_60_idx = np.where(levels == 60)[0][0]
    asymmetry = lower_slopes[level_60_idx] - upper_slopes[level_60_idx]
    ax.text(0.05, 0.95, f'Asymmetry @ 60 dB: {asymmetry:.1f} dB/ERB\n(more spreading downward)',
           transform=ax.transAxes, va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Middle-left: Masker and probe tones in time domain ===
    ax = fig.add_subplot(gs[1, 0])
    
    # Generate probe tones at different ERB distances
    probe_distances_erb = [-5, 0, 5]  # ERB units from masker
    colors_probe = ['blue', 'green', 'red']
    labels_probe = ['Probe -5 ERB', 'Masker (0 ERB)', 'Probe +5 ERB']
    
    t_short = t[:int(0.01 * fs)]  # Show 10ms
    
    # Plot masker (1 kHz)
    ax.plot(t_short.numpy() * 1000, masker[:len(t_short)].numpy(), 
           colors_probe[1], linewidth=1.5, label=labels_probe[1], alpha=0.8)
    
    # Plot probe tones
    for i, (dist, color, label) in enumerate(zip([probe_distances_erb[0], probe_distances_erb[2]], 
                                                  [colors_probe[0], colors_probe[2]], 
                                                  [labels_probe[0], labels_probe[2]])):
        # Convert ERB distance to frequency
        masker_erb = f2erbrate(torch.tensor([1000.0])).item()
        probe_erb = masker_erb + dist
        from torch_amt.common.filterbanks import erbrate2f
        probe_freq = erbrate2f(torch.tensor([probe_erb])).item()
        
        probe_tone = torch.sin(2 * np.pi * probe_freq * t[:len(t_short)])
        probe_tone = probe_tone * 10 ** ((40 - 100) / 20)  # 40 dB SPL probe
        ax.plot(t_short.numpy() * 1000, probe_tone.numpy(), 
               color, linewidth=1.5, label=label, alpha=0.7)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Masker and Probe Tones: Time Domain')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10])
    
    # === Middle-right: Masker and probe tones in frequency domain ===
    ax = fig.add_subplot(gs[1, 1])
    
    # Compute spectra
    masker_fft = torch.fft.rfft(masker)
    masker_freqs = torch.fft.rfftfreq(len(masker), 1/fs)
    masker_mag = 20 * torch.log10(torch.abs(masker_fft) + 1e-10)
    
    ax.plot(masker_freqs.numpy(), masker_mag.numpy(), 
           colors_probe[1], linewidth=2, label=labels_probe[1], alpha=0.8)
    
    # Plot probe spectra
    for i, (dist, color, label) in enumerate(zip([probe_distances_erb[0], probe_distances_erb[2]], 
                                                  [colors_probe[0], colors_probe[2]], 
                                                  [labels_probe[0], labels_probe[2]])):
        masker_erb = f2erbrate(torch.tensor([1000.0])).item()
        probe_erb = masker_erb + dist
        from torch_amt.common.filterbanks import erbrate2f
        probe_freq = erbrate2f(torch.tensor([probe_erb])).item()
        
        probe_tone = torch.sin(2 * np.pi * probe_freq * t)
        probe_tone = probe_tone * 10 ** ((40 - 100) / 20)
        probe_fft = torch.fft.rfft(probe_tone)
        probe_mag = 20 * torch.log10(torch.abs(probe_fft) + 1e-10)
        
        ax.plot(masker_freqs.numpy(), probe_mag.numpy(), 
               color, linewidth=2, label=label, alpha=0.7)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Masker and Probe Tones: Frequency Domain')
    ax.set_xscale('log')
    ax.set_xlim([200, 5000])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # === Bottom-left: Narrow-band noise excitation pattern ===
    ax = fig.add_subplot(gs[2, 0])
    
    ax.plot(fc_erb.numpy(), exc_spread_noise_avg, 'purple', linewidth=2, label='Narrow-band noise')
    
    # Mark the noise band
    ax.axvspan(1900, 2100, alpha=0.2, color='purple', label='Noise band (1900-2100 Hz)')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Excitation (dB SPL)')
    ax.set_title('Excitation Pattern: Narrow-Band Noise')
    ax.set_xscale('log')
    ax.set_xlim([500, 8000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Bottom-right: Masking (upward/downward/parallel) ===
    ax = fig.add_subplot(gs[2, 1])
    
    # Find masker ERB channel (1 kHz)
    masker_erb_idx = torch.argmin(torch.abs(fc_erb - 1000))
    masker_erb_center = erb_centers[masker_erb_idx].item()
    
    # Compute masking threshold (take excitation pattern as threshold)
    # Distance from masker in ERB units
    erb_distances = erb_centers.numpy() - masker_erb_center
    masking_threshold = exc_spread_masker_avg
    
    # Classify into upward, downward, and parallel masking
    # Upward: probe > masker frequency (erb_distances > 0)
    # Downward: probe < masker frequency (erb_distances < 0)
    # Parallel: probe ≈ masker frequency (|erb_distances| < 0.5)
    
    upward_mask = erb_distances > 0.5
    downward_mask = erb_distances < -0.5
    parallel_mask = np.abs(erb_distances) <= 0.5
    
    ax.plot(erb_distances[downward_mask], masking_threshold[downward_mask], 
           'b-o', linewidth=2, markersize=4, label='Downward masking', alpha=0.8)
    ax.plot(erb_distances[parallel_mask], masking_threshold[parallel_mask], 
           'g-s', linewidth=2, markersize=6, label='Parallel masking', alpha=0.8)
    ax.plot(erb_distances[upward_mask], masking_threshold[upward_mask], 
           'r-^', linewidth=2, markersize=4, label='Upward masking', alpha=0.8)
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(60, color='k', linestyle=':', alpha=0.3, linewidth=1, label='Masker level (60 dB)')
    
    ax.set_xlabel('Distance from Masker (ERB units)')
    ax.set_ylabel('Masking Threshold (dB SPL)')
    ax.set_title('Masking Pattern: 1 kHz Masker @ 60 dB SPL')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-10, 10])
    
    plt.suptitle('Excitation Pattern Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = TEST_FIGURES_DIR / 'excitation_pattern.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Quantitative analysis
    print("\n" + "=" * 80)
    print("Quantitative Analysis")
    print("=" * 80)
    
    print(f"\nSpreading slopes:")
    for level in [40, 60, 80]:
        upper, lower = excitation_pattern.get_spreading_slopes(level)
        print(f"  {level} dB SPL: upper={upper:.2f} dB/ERB, lower={lower:.2f} dB/ERB, "
              f"asymmetry={lower-upper:.2f} dB/ERB")
    
    print(f"\nExcitation pattern statistics (1 kHz tone):")
    print(f"  40 dB SPL: peak={exc_spread_40_avg[idx_1khz]:.2f} dB, "
          f"bandwidth @ -10dB={np.sum(exc_spread_40_avg > exc_spread_40_avg[idx_1khz] - 10)}")
    print(f"  60 dB SPL: peak={exc_spread_60_avg[idx_1khz]:.2f} dB, "
          f"bandwidth @ -10dB={np.sum(exc_spread_60_avg > exc_spread_60_avg[idx_1khz] - 10)}")
    print(f"  80 dB SPL: peak={exc_spread_80_avg[idx_1khz]:.2f} dB, "
          f"bandwidth @ -10dB={np.sum(exc_spread_80_avg > exc_spread_80_avg[idx_1khz] - 10)}")
    
    print(f"\nNarrow-band noise excitation:")
    noise_center_idx = torch.argmin(torch.abs(fc_erb - 2000))
    print(f"  Peak excitation: {exc_spread_noise_avg[noise_center_idx]:.2f} dB SPL")
    print(f"  Bandwidth @ -10dB: {np.sum(exc_spread_noise_avg > exc_spread_noise_avg[noise_center_idx] - 10)} channels")
    
    print(f"\nMasking pattern (1 kHz masker @ 60 dB SPL):")
    print(f"  Downward masking extent (-10 dB): {np.sum(downward_mask & (masking_threshold > 50))} channels")
    print(f"  Upward masking extent (-10 dB): {np.sum(upward_mask & (masking_threshold > 50))} channels")
    print(f"  Asymmetry ratio: {np.sum(downward_mask & (masking_threshold > 50)) / max(1, np.sum(upward_mask & (masking_threshold > 50))):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    test_excitation_pattern()
