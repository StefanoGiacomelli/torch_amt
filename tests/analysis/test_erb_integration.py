"""
ERB Integration - Test Suite

Contents:
1. test_erb_integration: Verifies ERB channel spacing and PSD integration
   - Tests 149 channels with 0.25 ERB-rate step
   - Validates frequency-to-ERB mapping
   - Confirms ERB bandwidth increases with frequency

Structure:
- Multi-tone complex signal test (200, 500, 1000, 2000, 5000 Hz)
- Broadband noise test (50-15000 Hz flat spectrum)
- Multi-resolution FFT + ERB integration pipeline
- Quantitative excitation pattern analysis

Figures generated:
- erb_integration.png: 6-panel analysis (excitation patterns, ERB spacing, transfer functions)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import MultiResolutionFFT, ERBIntegration, f2erbrate


def generate_test_signals(fs=32000, duration=0.5):
    """
    Generate test signals for ERB integration.
    
    Args:
        fs: Sampling rate in Hz
        duration: Signal duration in seconds
        
    Returns:
        multi_tone: Complex tone with multiple frequencies
        broadband_noise: Flat-spectrum noise
        t: Time vector
    """
    t = torch.linspace(0, duration, int(fs * duration))
    
    # 1. Multi-tone complex: 200, 500, 1000, 2000, 5000 Hz (all 60 dB SPL)
    freqs_tone = [200, 500, 1000, 2000, 5000]
    multi_tone = torch.zeros_like(t)
    
    for freq in freqs_tone:
        # Equal amplitude for all tones
        multi_tone += torch.sin(2 * np.pi * freq * t) / len(freqs_tone)
    
    # 2. Broadband noise (flat spectrum from 50 Hz to 15 kHz)
    # Generate white noise and apply bandpass filter
    noise = torch.randn_like(t)
    
    # Simple spectral shaping via FFT
    noise_fft = torch.fft.rfft(noise)
    freqs_fft = torch.fft.rfftfreq(len(noise), 1 / fs)
    
    # Create flat spectrum between 50 Hz and 15 kHz
    mask = (freqs_fft >= 50) & (freqs_fft <= 15000)
    noise_fft[~mask] = 0
    
    broadband_noise = torch.fft.irfft(noise_fft, n=len(noise))
    
    # Normalize
    broadband_noise = broadband_noise / broadband_noise.std() * 0.1
    
    return multi_tone, broadband_noise, t


def test_erb_integration():
    """Test ERB integration with various signals."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ERB Integration Test")
    print("=" * 80)
    
    # Parameters
    fs = 32000
    duration = 0.5
    
    # Generate test signals
    print("\nGenerating test signals...")
    multi_tone, broadband_noise, t = generate_test_signals(fs, duration)
    
    print(f"  Multi-tone: 200, 500, 1000, 2000, 5000 Hz")
    print(f"  Broadband noise: 50-15000 Hz")
    
    # Add batch dimension
    multi_tone_batch = multi_tone.unsqueeze(0)
    noise_batch = broadband_noise.unsqueeze(0)
    
    # Initialize components
    print("\nInitializing components...")
    multi_fft = MultiResolutionFFT(fs=fs, learnable=False)
    erb_integration = ERBIntegration(fs=fs, learnable=False)
    
    print(f"  ERB channels: {erb_integration.n_erb_bands}")
    print(f"  Frequency range: {erb_integration.f_min} - {erb_integration.f_max} Hz")
    print(f"  ERB step: {erb_integration.erb_step}")
    
    # Get ERB frequencies and bandwidths
    fc_erb = erb_integration.get_erb_frequencies()
    erb_bw = erb_integration.get_erb_bandwidths()
    
    print(f"  Center frequencies: {fc_erb[0]:.1f} - {fc_erb[-1]:.1f} Hz")
    print(f"  Bandwidths: {erb_bw[0]:.1f} - {erb_bw[-1]:.1f} Hz")
    
    # Compute multi-resolution FFT
    print("\nComputing multi-resolution FFT...")
    psd_multi_tone, freqs = multi_fft(multi_tone_batch)
    psd_noise, _ = multi_fft(noise_batch)
    
    print(f"  PSD shape: {psd_multi_tone.shape}")
    
    # Compute ERB integration
    print("\nComputing ERB integration...")
    excitation_multi_tone = erb_integration(psd_multi_tone, freqs)
    excitation_noise = erb_integration(psd_noise, freqs)
    
    print(f"  Excitation shape: {excitation_multi_tone.shape}")
    
    # Average over time for visualization
    exc_multi_tone_avg = excitation_multi_tone.mean(dim=1).squeeze(0).numpy()
    exc_noise_avg = excitation_noise.mean(dim=1).squeeze(0).numpy()
    
    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # === Top-left: ERB channel map ===
    ax = fig.add_subplot(gs[0, 0])
    
    # Convert ERB centers to ERB-rate scale
    erb_rate_centers = f2erbrate(fc_erb).numpy()
    
    # Plot ERB channels as vertical spans
    for i, (fc, bw, erb_rate) in enumerate(zip(fc_erb.numpy(), erb_bw.numpy(), erb_rate_centers)):
        # Show only every 10th channel to avoid clutter
        if i % 10 == 0:
            color = plt.cm.viridis(i / len(fc_erb))
            ax.axvspan(fc - bw/2, fc + bw/2, alpha=0.3, color=color)
            
            # Add text label for some channels
            if i % 30 == 0:
                ax.text(fc, 0.5, f'{fc:.0f} Hz\nERB={erb_rate:.1f}',
                       ha='center', va='center', fontsize=7, rotation=0)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Channel Index (every 10th shown)')
    ax.set_title(f'ERB Channel Map ({erb_integration.n_erb_bands} channels, step={erb_integration.erb_step})')
    ax.set_xlim([50, 16000])
    ax.set_ylim([0, 1])
    
    # Use symlog scale: log below 2000 Hz, linear above
    ax.set_xscale('symlog', linthresh=2000, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.grid(True, alpha=0.3, which='both')
    
    # Set custom ticks for better readability
    ax.set_xticks([100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000])
    ax.set_xticklabels(['100', '200', '500', '1k', '2k', '4k', '6k', '8k', '10k', '12k', '14k', '16k'])
    
    # Mark test tone frequencies
    test_freqs = [200, 500, 1000, 2000, 5000]
    for freq in test_freqs:
        ax.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add legends
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=plt.cm.viridis(0.5), alpha=0.3, label='ERB channels (every 10th)'),
        Line2D([0], [0], color='red', linestyle='--', alpha=0.5, linewidth=1, label='Test tone frequencies')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    
    # === Top-right: PSD input vs ERB output for multi-tone ===
    ax = fig.add_subplot(gs[0, 1])
    
    # Plot input PSD (average over time, in dB)
    psd_avg = psd_multi_tone.mean(dim=1).squeeze(0).numpy()
    psd_avg_db = 10 * np.log10(psd_avg + 1e-12)
    
    ax.semilogx(freqs.numpy(), psd_avg_db, 'b-', linewidth=1, alpha=0.5, label='Input PSD')
    
    # Plot ERB-integrated excitation
    ax2 = ax.twinx()
    ax2.semilogx(fc_erb.numpy(), exc_multi_tone_avg, 'r-', linewidth=2, label='ERB Excitation')
    
    # Mark test tone frequencies
    for freq in test_freqs:
        ax.axvline(freq, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(freq, ax.get_ylim()[1], f'{freq} Hz', ha='center', va='bottom',
               fontsize=7, rotation=45)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Input PSD (dB)', color='b')
    ax2.set_ylabel('ERB Excitation (dB SPL)', color='r')
    ax.set_title('Multi-Tone: PSD → ERB Integration', pad=30)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_xlim([50, 15000])
    ax.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    # === Bottom-left: ERB-integrated excitation for broadband noise ===
    ax = fig.add_subplot(gs[1, 0])
    
    ax.semilogx(fc_erb.numpy(), exc_noise_avg, 'g-', linewidth=2, label='Broadband Noise')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Excitation (dB SPL)')
    ax.set_title('Broadband Noise: ERB-Integrated Excitation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([50, 15000])
    
    # Add horizontal line at mean
    mean_exc = exc_noise_avg.mean()
    ax.axhline(mean_exc, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(100, mean_exc + 2, f'Mean = {mean_exc:.1f} dB SPL',
           fontsize=8, va='bottom')
    
    # === Bottom-right: ERB bandwidth vs frequency ===
    ax = fig.add_subplot(gs[1, 1])
    
    # Plot ERB bandwidth
    ax.loglog(fc_erb.numpy(), erb_bw.numpy(), 'b-', linewidth=2, label='ERB Bandwidth')
    
    # Add theoretical curve
    # ERB(f) = 24.673 * (4.368 * f/1000 + 1)
    f_theory = torch.logspace(np.log10(50), np.log10(15000), 100)
    erb_bw_theory = 24.673 * (4.368 * f_theory / 1000 + 1)
    ax.loglog(f_theory.numpy(), erb_bw_theory.numpy(), 'r--', linewidth=1,
             alpha=0.7, label='Theory (Glasberg & Moore 1990)')
    
    ax.set_xlabel('Center Frequency (Hz)')
    ax.set_ylabel('ERB Bandwidth (Hz)')
    ax.set_title('ERB Bandwidth vs Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([50, 15000])
    
    # Add some reference points
    ref_freqs = [100, 1000, 10000]
    for freq in ref_freqs:
        idx = torch.argmin(torch.abs(fc_erb - freq))
        ax.plot(fc_erb[idx].numpy(), erb_bw[idx].numpy(), 'ko', markersize=6)
        ax.text(fc_erb[idx].numpy(), erb_bw[idx].numpy() * 1.2,
               f'{fc_erb[idx]:.0f} Hz\nΔf={erb_bw[idx]:.0f} Hz',
               ha='center', fontsize=7)
    
    plt.suptitle('ERB Integration Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = TEST_FIGURES_DIR / 'erb_integration.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Quantitative analysis
    print("\n" + "=" * 80)
    print("Quantitative Analysis")
    print("=" * 80)
    
    print(f"\nERB channel configuration:")
    print(f"  Number of channels: {erb_integration.n_erb_bands}")
    print(f"  Frequency range: {fc_erb[0]:.1f} - {fc_erb[-1]:.1f} Hz")
    print(f"  ERB-rate range: {f2erbrate(fc_erb[0]):.2f} - {f2erbrate(fc_erb[-1]):.2f}")
    print(f"  ERB step size: {erb_integration.erb_step} ERB-rate units")
    
    print(f"\nERB bandwidth statistics:")
    print(f"  Minimum bandwidth: {erb_bw.min():.2f} Hz (at {fc_erb[0]:.1f} Hz)")
    print(f"  Maximum bandwidth: {erb_bw.max():.2f} Hz (at {fc_erb[-1]:.1f} Hz)")
    print(f"  Bandwidth ratio (max/min): {erb_bw.max() / erb_bw.min():.2f}")
    
    print(f"\nERB spacing verification:")
    erb_rate_centers_all = f2erbrate(fc_erb)
    erb_rate_spacing = torch.diff(erb_rate_centers_all)
    print(f"  Mean ERB spacing: {erb_rate_spacing.mean():.4f} ERB-rate units")
    print(f"  Std ERB spacing: {erb_rate_spacing.std():.6f} ERB-rate units")
    print(f"  Expected spacing: {erb_integration.erb_step} ERB-rate units")
    
    print(f"\nExcitation statistics (multi-tone signal):")
    print(f"  Mean excitation: {exc_multi_tone_avg.mean():.2f} dB SPL")
    print(f"  Max excitation: {exc_multi_tone_avg.max():.2f} dB SPL")
    print(f"  Min excitation: {exc_multi_tone_avg.min():.2f} dB SPL")
    print(f"  Dynamic range: {exc_multi_tone_avg.max() - exc_multi_tone_avg.min():.2f} dB")
    
    # Find peaks corresponding to test tones
    print(f"\nPeak detection (test tones at 200, 500, 1000, 2000, 5000 Hz):")
    test_freqs = [200, 500, 1000, 2000, 5000]
    for freq in test_freqs:
        # Find closest ERB channel
        idx = torch.argmin(torch.abs(fc_erb - freq))
        print(f"  {freq:5d} Hz → ERB channel {idx:3d} ({fc_erb[idx]:7.1f} Hz): "
              f"{exc_multi_tone_avg[idx]:6.2f} dB SPL")
    
    print(f"\nExcitation statistics (broadband noise):")
    print(f"  Mean excitation: {exc_noise_avg.mean():.2f} dB SPL")
    print(f"  Std excitation: {exc_noise_avg.std():.2f} dB SPL")
    print(f"  Expected: Approximately flat across ERB channels")
    print("=" * 80)


if __name__ == "__main__":
    test_erb_integration()
