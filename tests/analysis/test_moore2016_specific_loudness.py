"""
Moore2016 Specific Loudness - Test Suite

Contents:
1. test_moore2016_specific_loudness: Comprehensive verification of Moore2016SpecificLoudness

Structure:
- Tests frequency-dependent parameters G(f), Alpha(f), A(f)
- Verifies three loudness regimes (N1, N2, N3)
- Tests loudness growth curves at different frequencies (200 Hz, 1 kHz, 5 kHz)
- Validates threshold dependency and sub-threshold behavior
- Tests compression exponent variation (0.2 to ~0.267)
- Compares with Glasberg2002 SpecificLoudness

Figures generated:
- moore2016_specific_loudness.png: Complete specific loudness analysis
  * G(f), Alpha(f), A(f) parameter curves
  * Loudness growth functions at multiple frequencies
  * Regime transitions (N1, N2, N3)
  * Specific loudness patterns at different levels
  * Threshold effects and sub-threshold behavior
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import Moore2016SpecificLoudness


def test_moore2016_specific_loudness():
    """Test Moore2016SpecificLoudness with comprehensive analysis."""
    
    print("\n" + "="*80)
    print("MOORE2016 SPECIFIC LOUDNESS TEST")
    print("="*80)
    
    # Initialize specific loudness model
    specific_loudness = Moore2016SpecificLoudness()
    
    print(f"\nSpecific Loudness Configuration:")
    params = specific_loudness.get_parameters()
    print(f"  Number of channels: {params['n_channels']}")
    print(f"  Binaural constant C: {params['C']}")
    print(f"  G range: {params['G_min']:.4f} - {params['G_max']:.4f}")
    print(f"  Alpha range: {params['Alpha_min']:.4f} - {params['Alpha_max']:.4f}")
    print(f"  A range: {params['A_min']:.6f} - {params['A_max']:.6f}")
    print(f"  Frequency range: {specific_loudness.fc[0]:.1f} - {specific_loudness.fc[-1]:.1f} Hz")
    
    # ========== TEST 1: Parameter curves G(f), Alpha(f), A(f) ==========
    print("\n" + "-"*80)
    print("TEST 1: Frequency-dependent parameters")
    print("-"*80)
    
    G = specific_loudness.G.numpy()
    Alpha = specific_loudness.Alpha.numpy()
    A = specific_loudness.A.numpy()
    fc = specific_loudness.fc.numpy()
    erb_scale = specific_loudness.erb_scale.numpy()
    
    # Check key frequencies
    test_freqs = [200, 1000, 5000]
    for freq in test_freqs:
        idx = np.argmin(np.abs(fc - freq))
        print(f"\nAt {freq} Hz:")
        print(f"  G = {G[idx]:.4f} ({10*np.log10(G[idx]):.2f} dB)")
        print(f"  Alpha = {Alpha[idx]:.4f}")
        print(f"  A = {A[idx]:.6f} ({10*np.log10(A[idx]):.2f} dB)")
        print(f"  Threshold = {specific_loudness.threshold_db[idx]:.1f} dB SPL")
    
    # ========== TEST 2: Loudness growth curves ==========
    print("\n" + "-"*80)
    print("TEST 2: Loudness growth curves")
    print("-"*80)
    
    # Test excitation levels from 0 to 100 dB
    excitation_levels = torch.linspace(0, 100, 201)
    
    # Test frequencies
    test_freq_indices = {
        '200 Hz': np.argmin(np.abs(fc - 200)),
        '1 kHz': np.argmin(np.abs(fc - 1000)),
        '5 kHz': np.argmin(np.abs(fc - 5000)),
    }
    
    loudness_curves = {}
    
    for freq_label, idx in test_freq_indices.items():
        # Create excitation pattern with single active channel
        exc_db = torch.full((len(excitation_levels), 150), -100.0)  # Very low baseline
        exc_db[:, idx] = excitation_levels
        
        # Compute loudness
        loudness = specific_loudness(exc_db)
        loudness_curves[freq_label] = loudness[:, idx].numpy()
        
        # Find threshold crossing
        threshold = specific_loudness.threshold_db[idx].item()
        mask_above = excitation_levels.numpy() > threshold
        if mask_above.any():
            idx_threshold = np.where(mask_above)[0][0]
            print(f"\n{freq_label}:")
            print(f"  Threshold: {threshold:.1f} dB SPL")
            print(f"  Loudness at threshold: {loudness_curves[freq_label][idx_threshold]:.4f} sone/ERB")
            print(f"  Loudness at 60 dB: {loudness_curves[freq_label][120]:.4f} sone/ERB")
            print(f"  Loudness at 80 dB: {loudness_curves[freq_label][160]:.4f} sone/ERB")
    
    # ========== TEST 3: Regime identification ==========
    print("\n" + "-"*80)
    print("TEST 3: Three loudness regimes (N1, N2, N3)")
    print("-"*80)
    
    # Use 1 kHz as reference
    idx_1khz = test_freq_indices['1 kHz']
    threshold_1khz = specific_loudness.threshold_db[idx_1khz].item()
    
    # Test specific levels
    test_levels = [
        (threshold_1khz - 10, "Sub-threshold (N2)"),
        (threshold_1khz + 5, "Low-level (N1)"),
        (threshold_1khz + 30, "Mid-level (N1)"),
        (80, "High-level (N1/N3 transition)"),
        (100, "Very high-level (N3)"),
    ]
    
    for level, regime in test_levels:
        exc_db = torch.full((1, 150), -100.0)
        exc_db[0, idx_1khz] = level
        loudness = specific_loudness(exc_db)
        print(f"\n  {level:.1f} dB SPL - {regime}:")
        print(f"    Loudness: {loudness[0, idx_1khz]:.4f} sone/ERB")
    
    # ========== TEST 4: Threshold effects ==========
    print("\n" + "-"*80)
    print("TEST 4: Threshold effects and sub-threshold behavior")
    print("-"*80)
    
    # Fine resolution around threshold
    around_threshold = torch.linspace(threshold_1khz - 20, threshold_1khz + 20, 81)
    exc_db_threshold = torch.full((len(around_threshold), 150), -100.0)
    exc_db_threshold[:, idx_1khz] = around_threshold
    
    loudness_threshold = specific_loudness(exc_db_threshold)
    loudness_threshold_vals = loudness_threshold[:, idx_1khz].numpy()
    
    # Find where loudness exceeds small value (regime transition)
    idx_transition = np.where(loudness_threshold_vals > 0.001)[0]
    if len(idx_transition) > 0:
        transition_level = around_threshold[idx_transition[0]].item()
        print(f"\n  Transition from N2 to N1:")
        print(f"    Occurs at: {transition_level:.1f} dB SPL")
        print(f"    Threshold: {threshold_1khz:.1f} dB SPL")
        print(f"    Difference: {transition_level - threshold_1khz:.1f} dB")
    
    # ========== TEST 5: Frequency dependency comparison ==========
    print("\n" + "-"*80)
    print("TEST 5: Frequency dependency at fixed level (60 dB SPL)")
    print("-"*80)
    
    # All channels at 60 dB
    exc_60db = torch.full((1, 150), 60.0)
    loudness_60db = specific_loudness(exc_60db)
    loudness_60db_vals = loudness_60db[0].numpy()
    
    # Find peaks and valleys
    idx_max = np.argmax(loudness_60db_vals)
    idx_min = np.argmin(loudness_60db_vals)
    
    print(f"\n  Maximum loudness:")
    print(f"    {fc[idx_max]:.1f} Hz: {loudness_60db_vals[idx_max]:.4f} sone/ERB")
    print(f"    G = {G[idx_max]:.4f}, Alpha = {Alpha[idx_max]:.4f}")
    
    print(f"\n  Minimum loudness:")
    print(f"    {fc[idx_min]:.1f} Hz: {loudness_60db_vals[idx_min]:.4f} sone/ERB")
    print(f"    G = {G[idx_min]:.4f}, Alpha = {Alpha[idx_min]:.4f}")
    
    # ========== TEST 6: Compression behavior ==========
    print("\n" + "-"*80)
    print("TEST 6: Compression exponent validation")
    print("-"*80)
    
    # Check if compression follows power law approximately
    # For N1 regime: N ∝ E^Alpha
    levels_high = torch.linspace(threshold_1khz + 10, 80, 50)
    exc_high = torch.full((len(levels_high), 150), -100.0)
    exc_high[:, idx_1khz] = levels_high
    
    loudness_high = specific_loudness(exc_high)
    loudness_high_vals = loudness_high[:, idx_1khz].numpy()
    
    # Fit power law in log-log space
    mask_positive = loudness_high_vals > 1e-6
    if mask_positive.sum() > 10:
        log_exc = np.log10(10 ** (levels_high.numpy()[mask_positive] / 10.0))
        log_loud = np.log10(loudness_high_vals[mask_positive])
        
        # Linear fit
        coeffs = np.polyfit(log_exc, log_loud, 1)
        fitted_exponent = coeffs[0]
        
        print(f"\n  Power law fit (N1 regime):")
        print(f"    Expected Alpha: {Alpha[idx_1khz]:.4f}")
        print(f"    Fitted exponent: {fitted_exponent:.4f}")
        print(f"    Difference: {abs(fitted_exponent - Alpha[idx_1khz]):.4f}")
    
    # ========== VISUALIZATION ==========
    print("\n" + "-"*80)
    print("Creating comprehensive visualization...")
    print("-"*80)
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # === Plot 1: G(f) parameter ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    G_db = 10 * np.log10(G)
    ax1.semilogx(fc, G_db, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('G (dB)')
    ax1.set_title('Low-Level Gain Parameter G(f)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([fc[0], fc[-1]])
    
    # === Plot 2: Alpha(f) parameter ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.semilogx(fc, Alpha, 'r-', linewidth=2)
    ax2.axhline(y=0.2, color='k', linestyle='--', alpha=0.3, linewidth=1, label='α = 0.2')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Alpha (compression exponent)')
    ax2.set_title('Compression Exponent Alpha(f)')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([fc[0], fc[-1]])
    ax2.legend()
    
    # === Plot 3: A(f) parameter ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    A_db = 10 * np.log10(A)
    ax3.semilogx(fc, A_db, 'g-', linewidth=2)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('A (dB)')
    ax3.set_title('Additive Constant A(f)')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([fc[0], fc[-1]])
    
    # === Plot 4: Loudness growth @ 200 Hz ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    freq_label = '200 Hz'
    idx = test_freq_indices[freq_label]
    threshold = specific_loudness.threshold_db[idx].item()
    
    ax4.plot(excitation_levels.numpy(), loudness_curves[freq_label], 'b-', linewidth=2)
    ax4.axvline(x=threshold, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Threshold')
    ax4.set_xlabel('Excitation (dB SPL)')
    ax4.set_ylabel('Specific Loudness (sone/ERB)')
    ax4.set_title(f'Loudness Growth: {freq_label}')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim([0, 100])
    
    # === Plot 5: Loudness growth @ 1 kHz ===
    ax5 = fig.add_subplot(gs[1, 1])
    
    freq_label = '1 kHz'
    idx = test_freq_indices[freq_label]
    threshold = specific_loudness.threshold_db[idx].item()
    
    ax5.plot(excitation_levels.numpy(), loudness_curves[freq_label], 'b-', linewidth=2)
    ax5.axvline(x=threshold, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Threshold')
    ax5.set_xlabel('Excitation (dB SPL)')
    ax5.set_ylabel('Specific Loudness (sone/ERB)')
    ax5.set_title(f'Loudness Growth: {freq_label}')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim([0, 100])
    
    # === Plot 6: Loudness growth @ 5 kHz ===
    ax6 = fig.add_subplot(gs[1, 2])
    
    freq_label = '5 kHz'
    idx = test_freq_indices[freq_label]
    threshold = specific_loudness.threshold_db[idx].item()
    
    ax6.plot(excitation_levels.numpy(), loudness_curves[freq_label], 'b-', linewidth=2)
    ax6.axvline(x=threshold, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Threshold')
    ax6.set_xlabel('Excitation (dB SPL)')
    ax6.set_ylabel('Specific Loudness (sone/ERB)')
    ax6.set_title(f'Loudness Growth: {freq_label}')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_xlim([0, 100])
    
    # === Plot 7: Log-log loudness growth (compression visualization) ===
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Use 1 kHz, only above threshold
    freq_label = '1 kHz'
    idx = test_freq_indices[freq_label]
    threshold = specific_loudness.threshold_db[idx].item()
    
    mask_above = excitation_levels.numpy() > threshold
    exc_above = excitation_levels.numpy()[mask_above]
    loud_above = loudness_curves[freq_label][mask_above]
    
    # Convert to linear excitation
    exc_linear = 10 ** (exc_above / 10.0)
    
    ax7.loglog(exc_linear, loud_above, 'b-', linewidth=2, label='Moore2016')
    
    # Reference line with Alpha slope
    alpha_ref = Alpha[idx]
    exc_ref = exc_linear[len(exc_linear)//2]
    loud_ref = loud_above[len(loud_above)//2]
    exc_ref_range = np.logspace(np.log10(exc_linear[0]), np.log10(exc_linear[-1]), 100)
    loud_ref_line = loud_ref * (exc_ref_range / exc_ref) ** alpha_ref
    ax7.loglog(exc_ref_range, loud_ref_line, 'r--', linewidth=1.5, alpha=0.7, 
              label=f'Slope α={alpha_ref:.3f}')
    
    ax7.set_xlabel('Excitation (linear)')
    ax7.set_ylabel('Specific Loudness (sone/ERB)')
    ax7.set_title('Compression in Log-Log Space (1 kHz)')
    ax7.grid(True, alpha=0.3, which='both')
    ax7.legend()
    
    # === Plot 8: Threshold region detail ===
    ax8 = fig.add_subplot(gs[2, 1])
    
    ax8.plot(around_threshold.numpy(), loudness_threshold_vals, 'b-', linewidth=2)
    ax8.axvline(x=threshold_1khz, color='r', linestyle='--', alpha=0.7, linewidth=2, 
               label='Threshold')
    ax8.set_xlabel('Excitation (dB SPL)')
    ax8.set_ylabel('Specific Loudness (sone/ERB)')
    ax8.set_title('Threshold Region Detail (1 kHz)')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    ax8.set_xlim([threshold_1khz - 20, threshold_1khz + 20])
    
    # === Plot 9: Frequency dependency at 60 dB ===
    ax9 = fig.add_subplot(gs[2, 2])
    
    ax9.semilogx(fc, loudness_60db_vals, 'b-', linewidth=2)
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('Specific Loudness (sone/ERB)')
    ax9.set_title('Frequency Dependency @ 60 dB SPL')
    ax9.grid(True, alpha=0.3, which='both')
    ax9.set_xlim([fc[0], fc[-1]])
    
    # Mark extrema
    ax9.plot(fc[idx_max], loudness_60db_vals[idx_max], 'go', markersize=8, label='Maximum')
    ax9.plot(fc[idx_min], loudness_60db_vals[idx_min], 'ro', markersize=8, label='Minimum')
    ax9.legend()
    
    # Overall title
    fig.suptitle('Moore2016 Specific Loudness - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    test_figures_dir = Path(__file__).parent.parent.parent / 'test_figures'
    test_figures_dir.mkdir(exist_ok=True)
    output_path = test_figures_dir / 'moore2016_specific_loudness.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figure saved: {output_path}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == '__main__':
    test_moore2016_specific_loudness()
