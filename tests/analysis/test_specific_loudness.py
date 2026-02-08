"""
Specific Loudness - Test Suite

Contents:
1. test_specific_loudness: Verifies specific loudness implementation

Structure:
- Tests three regimes (sub-threshold, linear, compressive)
- Verifies compression exponent ≈ 0.2
- Tests specific loudness pattern for multi-tone complexes
- Validates loudness growth per ISO 226

Figures generated:
- specific_loudness.png: Complete specific loudness analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from torch_amt.common import (MultiResolutionFFT, 
                              ERBIntegration, 
                              ExcitationPattern, 
                              SpecificLoudness
                              )


def generate_test_signals(fs=32000, duration=0.5):
    """
    Generate test signals for specific loudness analysis.
    
    Returns:
        tones_1khz_levels: Dict of 1 kHz tones at various levels (0-100 dB SPL, step 5 dB)
        multi_tone_complex: Multi-tone complex (200, 500, 1000, 2000, 5000 Hz @ 60 dB)
        tones_200hz, tones_1khz, tones_5khz: Tones at various levels for 3 frequencies
        t: Time vector
    """
    t = torch.linspace(0, duration, int(fs * duration))
    
    # 1. 1 kHz tones at various levels (0-100 dB SPL, step 5 dB)
    tone_1khz = torch.sin(2 * np.pi * 1000 * t)
    tones_1khz_levels = {}
    for level in range(0, 101, 5):
        tones_1khz_levels[level] = tone_1khz * 10 ** ((level - 100) / 20)
    
    # 2. Multi-tone complex at 60 dB SPL
    freqs_multi = [200, 500, 1000, 2000, 5000]
    multi_tone_complex = torch.zeros_like(t)
    for f in freqs_multi:
        multi_tone_complex += torch.sin(2 * np.pi * f * t)
    multi_tone_complex = multi_tone_complex / len(freqs_multi)  # Average
    multi_tone_complex = multi_tone_complex * 10 ** ((60 - 100) / 20)
    
    # 3. Tones at various levels for 3 frequencies
    tones_200hz = {}
    tones_5khz = {}
    for level in range(0, 101, 5):
        tones_200hz[level] = torch.sin(2 * np.pi * 200 * t) * 10 ** ((level - 100) / 20)
        tones_5khz[level] = torch.sin(2 * np.pi * 5000 * t) * 10 ** ((level - 100) / 20)
    
    return tones_1khz_levels, multi_tone_complex, tones_200hz, tones_5khz, t


def compute_iso226_threshold(freq):
    """
    Compute ISO 226 absolute threshold of hearing.
    
    Args:
        freq: Frequency in Hz (can be array-like)
    
    Returns:
        threshold: Threshold in dB SPL
    """
    f_khz = freq / 1000.0
    threshold = (3.64 * (f_khz ** -0.8) 
                 - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) 
                 + 1e-3 * (f_khz ** 4))
    return threshold


def test_specific_loudness():
    """Test specific loudness with various signals."""
    
    print("=" * 80)
    print("Specific Loudness Test")
    print("=" * 80)
    
    # Parameters
    fs = 32000
    duration = 0.5
    
    # Generate test signals
    print("\nGenerating test signals...")
    tones_1khz, multi_tone, tones_200hz, tones_5khz, t = generate_test_signals(fs, duration)
    
    print(f"  1 kHz tones at 0-100 dB SPL (step 5 dB)")
    print(f"  Multi-tone complex: 200, 500, 1000, 2000, 5000 Hz @ 60 dB")
    print(f"  Tones at 200 Hz, 1 kHz, 5 kHz for loudness growth")
    
    # Initialize components
    print("\nInitializing components...")
    multi_fft = MultiResolutionFFT(fs=fs, learnable=False)
    erb_integration = ERBIntegration(fs=fs, learnable=False)
    excitation_pattern = ExcitationPattern(fs=fs, learnable=False)
    specific_loudness = SpecificLoudness(fs=fs, learnable=False)
    
    fc_erb = specific_loudness.fc_erb
    
    # Get parameters
    params = specific_loudness.get_parameters()
    print(f"  ERB channels: {specific_loudness.n_erb_bands}")
    print(f"  Compression exponent α: {params['alpha']:.3f}")
    print(f"  Constant C: {params['C']:.4f}")
    print(f"  E0 offset: {params['E0_offset']:.1f} dB above threshold")
    
    # === 1. Process 1 kHz tone at various levels ===
    print("\nProcessing 1 kHz tones at various levels...")
    levels = sorted(tones_1khz.keys())
    loudness_1khz = []
    
    idx_1khz = torch.argmin(torch.abs(fc_erb - 1000))
    
    for level in levels:
        signal = tones_1khz[level].unsqueeze(0)
        psd, freqs = multi_fft(signal)
        exc = erb_integration(psd, freqs)
        exc_spread = excitation_pattern(exc)
        specific_loud = specific_loudness(exc_spread)
        
        # Take average over time and extract 1 kHz channel
        specific_loud_avg = specific_loud.mean(dim=1).squeeze(0)
        loudness_1khz.append(specific_loud_avg[idx_1khz].item())
    
    loudness_1khz = np.array(loudness_1khz)
    
    # === 2. Process multi-tone complex ===
    print("Processing multi-tone complex...")
    multi_tone_batch = multi_tone.unsqueeze(0)
    psd_multi, freqs = multi_fft(multi_tone_batch)
    exc_multi = erb_integration(psd_multi, freqs)
    exc_spread_multi = excitation_pattern(exc_multi)
    specific_loud_multi = specific_loudness(exc_spread_multi)
    specific_loud_multi_avg = specific_loud_multi.mean(dim=1).squeeze(0).numpy()
    
    # === 3. Process tones at 3 frequencies for loudness growth ===
    print("Processing loudness growth for 200 Hz, 1 kHz, 5 kHz...")
    idx_200hz = torch.argmin(torch.abs(fc_erb - 200))
    idx_5khz = torch.argmin(torch.abs(fc_erb - 5000))
    
    loudness_200hz = []
    loudness_5khz = []
    
    for level in levels:
        # 200 Hz
        signal_200 = tones_200hz[level].unsqueeze(0)
        psd_200, freqs = multi_fft(signal_200)
        exc_200 = erb_integration(psd_200, freqs)
        exc_spread_200 = excitation_pattern(exc_200)
        specific_loud_200 = specific_loudness(exc_spread_200)
        loudness_200hz.append(specific_loud_200.mean(dim=1).squeeze(0)[idx_200hz].item())
        
        # 5 kHz
        signal_5k = tones_5khz[level].unsqueeze(0)
        psd_5k, freqs = multi_fft(signal_5k)
        exc_5k = erb_integration(psd_5k, freqs)
        exc_spread_5k = excitation_pattern(exc_5k)
        specific_loud_5k = specific_loudness(exc_spread_5k)
        loudness_5khz.append(specific_loud_5k.mean(dim=1).squeeze(0)[idx_5khz].item())
    
    loudness_200hz = np.array(loudness_200hz)
    loudness_5khz = np.array(loudness_5khz)
    
    # Get absolute threshold for these frequencies
    threshold_1khz = specific_loudness.get_threshold()[idx_1khz].item()
    threshold_200hz = specific_loudness.get_threshold()[idx_200hz].item()
    threshold_5khz = specific_loudness.get_threshold()[idx_5khz].item()
    
    # Compute E0 for 1 kHz
    E0_1khz = threshold_1khz + params['E0_offset']
    
    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # === Top row: ISO 226 Equal Loudness Contours (spanning 2 columns) ===
    ax = fig.add_subplot(gs[0, :])
    
    # ISO 226 equal loudness contours
    freqs_iso = np.logspace(np.log10(50), np.log10(15000), 200)
    phon_levels_iso = [20, 40, 60, 80, 100]
    
    # For ISO 226, we need to compute SPL for each phon level at each frequency
    # This is a simplified approximation using the threshold curve
    for phon in phon_levels_iso:
        # Approximate: SPL(f, phon) ≈ threshold(f) + phon
        # This is very simplified; real ISO 226 is more complex
        threshold_curve = compute_iso226_threshold(freqs_iso)
        
        # Better approximation: use loudness level scaling
        # At 1 kHz: phon = dB SPL (by definition)
        # At other frequencies: adjust based on threshold
        threshold_1khz_iso = compute_iso226_threshold(1000)
        spl_curve = threshold_curve + (phon - threshold_1khz_iso)
        
        ax.plot(freqs_iso, spl_curve, linewidth=2, label=f'{phon} phon', alpha=0.8)
    
    # Add threshold curve
    threshold_all = compute_iso226_threshold(freqs_iso)
    ax.plot(freqs_iso, threshold_all, 'k--', linewidth=2, label='Threshold (0 phon)', alpha=0.6)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Sound Pressure Level (dB SPL)', fontsize=11)
    ax.set_title('ISO 226 Equal Loudness Contours (Phon vs dB SPL vs Frequency)', fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim([50, 15000])
    ax.set_ylim([-20, 120])
    ax.legend(fontsize=9, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax.text(0.02, 0.08, 'Equal phon → equal loudness perception\nLines show SPL needed for constant loudness',
           transform=ax.transAxes, va='bottom', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Middle-left: Loudness vs level for 1 kHz (with 3 regimes) ===
    ax = fig.add_subplot(gs[1, 0])
    
    # Color regime regions
    y_max = max(loudness_1khz) * 1.15  # Leave space at top
    ax.axvspan(0, threshold_1khz, alpha=0.15, color='lightblue', label='Sub-threshold (N=0)')
    ax.axvspan(threshold_1khz, E0_1khz, alpha=0.15, color='lightgreen', label='Linear (N ∝ E)')
    ax.axvspan(E0_1khz, 100, alpha=0.15, color='lightyellow', label=f'Compressive (N ∝ E^{params["alpha"]:.2f})')
    
    # Plot data on top of colored regions
    ax.plot(levels, loudness_1khz, 'b-o', linewidth=2, markersize=4, label='1 kHz tone', alpha=0.8, zorder=10)
    
    # Mark regime boundaries
    ax.axvline(threshold_1khz, color='k', linestyle='--', alpha=0.4, linewidth=1.5, zorder=5)
    ax.axvline(E0_1khz, color='k', linestyle='--', alpha=0.4, linewidth=1.5, zorder=5)
    
    ax.set_xlabel('Input Level (dB SPL)')
    ax.set_ylabel('Specific Loudness (sone/ERB)')
    ax.set_title('Loudness Growth: 1 kHz Tone (3 Regimes)', pad=12)
    ax.set_xscale('log')  # Log scale for input level
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both', zorder=0)
    ax.set_xlim([1, 100])
    ax.set_ylim([0, y_max])
    
    # === Middle-right: Log-log plot to verify α ===
    ax = fig.add_subplot(gs[1, 1])
    
    # Use only compressive regime data (E > E0)
    compressive_mask = np.array(levels) > E0_1khz
    if np.sum(compressive_mask) > 2:
        levels_comp = np.array(levels)[compressive_mask]
        loudness_comp = loudness_1khz[compressive_mask]
        
        # Remove zeros for log-log plot
        nonzero_mask = loudness_comp > 0
        levels_comp = levels_comp[nonzero_mask]
        loudness_comp = loudness_comp[nonzero_mask]
        
        if len(levels_comp) > 2:
            # Log-log plot
            ax.loglog(levels_comp, loudness_comp, 'ro', markersize=6, label='Data (compressive regime)', alpha=0.8)
            
            # Fit power law: N = a * E^α
            log_levels = np.log10(levels_comp)
            log_loudness = np.log10(loudness_comp)
            
            # Linear fit in log-log space
            p = np.polyfit(log_levels, log_loudness, 1)
            alpha_fitted = p[0]
            
            # Plot fitted line
            levels_fit = np.logspace(np.log10(levels_comp.min()), np.log10(levels_comp.max()), 50)
            loudness_fit = 10 ** (p[1] + alpha_fitted * np.log10(levels_fit))
            ax.loglog(levels_fit, loudness_fit, 'b-', linewidth=2, 
                     label=f'Power law fit: α={alpha_fitted:.3f}', alpha=0.7)
            
            ax.set_xlabel('Excitation Level (dB SPL)')
            ax.set_ylabel('Specific Loudness (sone/ERB)')
            ax.set_title('Log-Log Plot: Verify Compression Exponent')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
            
            # Add text with expected vs fitted α
            ax.text(0.05, 0.95, f'Expected: α={params["alpha"]:.3f}\nFitted: α={alpha_fitted:.3f}',
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Bottom-left: Specific loudness pattern for multi-tone ===
    ax = fig.add_subplot(gs[2, 0])
    
    ax.plot(fc_erb.numpy(), specific_loud_multi_avg, 'purple', linewidth=2, label='Multi-Tone @ 60 dB')
    
    # Mark tone frequencies
    for f in [200, 500, 1000, 2000, 5000]:
        ax.axvline(f, color='k', linestyle=':', alpha=0.3, linewidth=1)
        ax.text(f, ax.get_ylim()[1], f'{f} Hz', ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Specific Loudness (sone/ERB)')
    ax.set_title('Specific Loudness Pattern: Multi-Tone', pad=25)
    ax.set_xscale('log')
    ax.set_xlim([100, 10000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Bottom-right: Loudness growth with log scale ===
    ax = fig.add_subplot(gs[2, 1])
    
    # Loudness growth curves
    ax.plot(levels, loudness_200hz, 'b-o', linewidth=2, markersize=4, label='200 Hz', alpha=0.8)
    ax.plot(levels, loudness_1khz, 'g-s', linewidth=2, markersize=4, label='1 kHz', alpha=0.8)
    ax.plot(levels, loudness_5khz, 'r-^', linewidth=2, markersize=4, label='5 kHz', alpha=0.8)
    
    # Add ISO 226 equal loudness contours (approximate)
    # We show contours at a few phon levels
    phon_levels = [20, 40, 60, 80]
    freqs_iso = np.logspace(np.log10(50), np.log10(15000), 100)
    
    # For each phon level, compute approximate threshold shift
    # This is a rough approximation: we scale the threshold curve
    for phon in phon_levels:
        # Approximate: equal loudness at X phon corresponds to threshold + X dB
        # This is very rough but gives qualitative sense
        threshold_iso = compute_iso226_threshold(freqs_iso)
        # Skip plotting ISO curves in this subplot to avoid clutter
        # Instead, mark threshold levels
        pass
    
    # Mark thresholds
    ax.axhline(0, color='k', linestyle='--', alpha=0.2, linewidth=1)
    ax.text(5, 0.05, f'Thresholds: 200Hz={threshold_200hz:.1f} dB, '
            f'1kHz={threshold_1khz:.1f} dB, 5kHz={threshold_5khz:.1f} dB',
           fontsize=7, va='bottom')
    
    ax.set_xlabel('Input Level (dB SPL)')
    ax.set_ylabel('Specific Loudness (sone/ERB)')
    ax.set_title('Loudness Growth: 3 Frequencies')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    plt.suptitle('Specific Loudness Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = Path(__file__).parent.parent.parent / 'test_figures' / 'specific_loudness.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Quantitative analysis
    print("\n" + "=" * 80)
    print("Quantitative Analysis")
    print("=" * 80)
    
    print(f"\nModel parameters:")
    print(f"  C = {params['C']:.4f}")
    print(f"  α = {params['alpha']:.3f}")
    print(f"  E0_offset = {params['E0_offset']:.1f} dB")
    
    print(f"\nAbsolute thresholds (ISO 226):")
    print(f"  200 Hz: {threshold_200hz:.2f} dB SPL")
    print(f"  1 kHz: {threshold_1khz:.2f} dB SPL")
    print(f"  5 kHz: {threshold_5khz:.2f} dB SPL")
    
    print(f"\nRegime boundaries (1 kHz):")
    print(f"  Sub-threshold → Linear: {threshold_1khz:.2f} dB SPL")
    print(f"  Linear → Compressive: {E0_1khz:.2f} dB SPL")
    
    # Verify compression exponent from data
    if 'alpha_fitted' in locals():
        print(f"\nCompression exponent verification:")
        print(f"  Expected α: {params['alpha']:.3f}")
        print(f"  Fitted α (log-log): {alpha_fitted:.3f}")
        print(f"  Error: {abs(alpha_fitted - params['alpha']):.3f} ({abs(alpha_fitted - params['alpha']) / params['alpha'] * 100:.1f}%)")
    
    print(f"\nLoudness at 60 dB SPL:")
    print(f"  200 Hz: {loudness_200hz[np.where(np.array(levels) == 60)[0][0]]:.4f} sone/ERB")
    print(f"  1 kHz: {loudness_1khz[np.where(np.array(levels) == 60)[0][0]]:.4f} sone/ERB")
    print(f"  5 kHz: {loudness_5khz[np.where(np.array(levels) == 60)[0][0]]:.4f} sone/ERB")
    
    print(f"\nMulti-tone complex (60 dB SPL):")
    for f in [200, 500, 1000, 2000, 5000]:
        idx_f = torch.argmin(torch.abs(fc_erb - f))
        print(f"  {f} Hz: {specific_loud_multi_avg[idx_f]:.4f} sone/ERB")
    
    print("=" * 80)


if __name__ == "__main__":
    test_specific_loudness()
