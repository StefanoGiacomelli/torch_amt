"""
Moore2016 Binaural Loudness - Test Suite

Contents:
1. test_moore2016_binauralloudness: Comprehensive verification of binaural loudness processing

Structure:
- Tests Gaussian spatial smoothing (kernel shape, energy conservation)
- Verifies sech-based binaural inhibition (diotic, monaural, dichotic)
- Tests ILD (Interaural Level Difference) effects from -40 to +40 dB
- Validates inhibition factors across multiple frequencies (100, 500, 1k, 5k Hz)
- Compares loudness with and without binaural inhibition

Figures generated:
- moore2016_binauralloudness.png: Complete binaural loudness analysis
  * Gaussian kernel shape and normalization
  * Spatial smoothing effects at multiple frequencies
  * Inhibition curves and factors
  * ILD-dependent loudness modulation
  * Component contributions (left, right, total)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import erbrate2f, SpatialSmoothing, BinauralInhibition, Moore2016BinauralLoudness


def test_moore2016_binauralloudness():
    """Test Moore2016 binaural loudness processing with comprehensive analysis."""
    
    print("\n" + "="*80)
    print("MOORE2016 BINAURAL LOUDNESS TEST")
    print("="*80)
    
    # Initialize modules
    spatial_smoothing = SpatialSmoothing(kernel_width=18.0, sigma=0.08)
    binaural_inhibition = BinauralInhibition(p=1.5978)
    binaural_loudness = Moore2016BinauralLoudness()
    
    print(f"\nModule Configuration:")
    smooth_params = spatial_smoothing.get_parameters()
    inhib_params = binaural_inhibition.get_parameters()
    print(f"  Spatial Smoothing:")
    print(f"    Kernel width: {smooth_params['kernel_width']} ERB")
    print(f"    Sigma: {smooth_params['sigma']:.4f}")
    print(f"    Kernel size: {smooth_params['kernel_size']} samples")
    print(f"  Binaural Inhibition:")
    print(f"    Exponent p: {inhib_params['p']:.4f}")
    
    # ERB scale and frequencies
    erb_scale = torch.arange(1.75, 39.25, 0.25)
    fc = erbrate2f(erb_scale).numpy()
    
    # Test frequencies
    test_freqs = [100, 500, 1000, 5000]
    test_freq_indices = {freq: np.argmin(np.abs(fc - freq)) for freq in test_freqs}
    
    # ========== TEST 1: Gaussian kernel ==========
    print("\n" + "-"*80)
    print("TEST 1: Gaussian spatial smoothing kernel")
    print("-"*80)
    
    kernel = spatial_smoothing.gaussian_kernel.numpy()
    g = np.arange(-18.0, 18.25, 0.25)
    
    print(f"\nKernel properties:")
    print(f"  Number of samples: {len(kernel)}")
    print(f"  Sum (normalization): {kernel.sum():.6f}")
    print(f"  Max value: {kernel.max():.6f} at g=0")
    print(f"  Symmetry check: {np.allclose(kernel[:72], kernel[-72:][::-1])}")
    print(f"  Support: g ∈ [{g[0]:.1f}, {g[-1]:.1f}] ERB")
    
    # ========== TEST 2: Spatial smoothing effect ==========
    print("\n" + "-"*80)
    print("TEST 2: Spatial smoothing effect at multiple frequencies")
    print("-"*80)
    
    smoothing_results = {}
    
    for freq in test_freqs:
        idx = test_freq_indices[freq]
        
        # Create delta function at frequency
        spec_loud = torch.zeros(1, 150)
        spec_loud[0, idx] = 1.0
        
        # Apply smoothing
        smoothed = spatial_smoothing(spec_loud)
        
        smoothing_results[freq] = {
            'original': spec_loud[0].numpy(),
            'smoothed': smoothed[0].numpy(),
            'energy_in': spec_loud.sum().item(),
            'energy_out': smoothed.sum().item()
        }
        
        print(f"\n  {freq} Hz:")
        print(f"    Original energy: {smoothing_results[freq]['energy_in']:.6f}")
        print(f"    Smoothed energy: {smoothing_results[freq]['energy_out']:.6f}")
        print(f"    Energy loss: {(1 - smoothing_results[freq]['energy_out']) * 100:.2f}%")
        print(f"    Peak spread: {(smoothed[0] > 0.01).sum().item()} channels")
    
    # ========== TEST 3: Inhibition curve ==========
    print("\n" + "-"*80)
    print("TEST 3: Binaural inhibition curve (sech function)")
    print("-"*80)
    
    # Test ratios from 0.01 to 100 (log scale)
    ratios = torch.logspace(-2, 2, 200)
    
    # Create loudness patterns with varying ratios
    left_const = torch.ones(200, 150)
    right_var = left_const * ratios.unsqueeze(1)
    
    inhib_left, inhib_right = binaural_inhibition(left_const, right_var)
    
    # Extract inhibition at center channel
    inhib_left_center = inhib_left[:, 75].numpy()
    inhib_right_center = inhib_right[:, 75].numpy()
    ratios_np = ratios.numpy()
    
    print(f"\nInhibition factor ranges:")
    print(f"  Left (NR/NL varies): {inhib_left_center.min():.4f} - {inhib_left_center.max():.4f}")
    print(f"  Right (NL/NR varies): {inhib_right_center.min():.4f} - {inhib_right_center.max():.4f}")
    
    # Key points
    idx_diotic = np.argmin(np.abs(ratios_np - 1.0))
    idx_monaural_low = np.argmin(np.abs(ratios_np - 0.01))
    idx_monaural_high = np.argmin(np.abs(ratios_np - 100))
    
    print(f"\nKey inhibition values:")
    print(f"  Diotic (ratio=1): {inhib_left_center[idx_diotic]:.4f}")
    print(f"  Monaural L>>R (ratio→0): {inhib_left_center[idx_monaural_low]:.4f}")
    print(f"  Monaural R>>L (ratio→∞): {inhib_left_center[idx_monaural_high]:.4f}")
    
    # ========== TEST 4: Diotic stimuli (L=R) ==========
    print("\n" + "-"*80)
    print("TEST 4: Diotic stimuli (L=R)")
    print("-"*80)
    
    # Create identical L/R patterns
    spec_loud_diotic = torch.randn(1, 150).abs() * 2.0
    
    loudness_diotic, loud_L_diotic, loud_R_diotic = binaural_loudness(
        spec_loud_diotic, spec_loud_diotic
    )
    
    # Check inhibition factors
    left_smooth = spatial_smoothing(spec_loud_diotic)
    right_smooth = spatial_smoothing(spec_loud_diotic)
    inhib_L_diotic, inhib_R_diotic = binaural_inhibition(left_smooth, right_smooth)
    
    print(f"\nDiotic condition:")
    print(f"  Input L=R: {spec_loud_diotic[0, 75]:.4f} sone/ERB @ 1kHz")
    print(f"  Inhibition factors: L={inhib_L_diotic[0, 75]:.4f}, R={inhib_R_diotic[0, 75]:.4f}")
    print(f"  Total loudness: {loudness_diotic.item():.4f} sone")
    print(f"  Left contribution: {loud_L_diotic.item():.4f} sone")
    print(f"  Right contribution: {loud_R_diotic.item():.4f} sone")
    
    # ========== TEST 5: Monaural stimuli ==========
    print("\n" + "-"*80)
    print("TEST 5: Monaural stimuli (L=0 or R=0)")
    print("-"*80)
    
    spec_loud_mono = torch.randn(1, 150).abs() * 2.0
    
    # Left ear only
    loudness_mono_L, loud_L_mono_L, loud_R_mono_L = binaural_loudness(
        spec_loud_mono, torch.zeros_like(spec_loud_mono)
    )
    
    # Right ear only
    loudness_mono_R, loud_L_mono_R, loud_R_mono_R = binaural_loudness(
        torch.zeros_like(spec_loud_mono), spec_loud_mono
    )
    
    # Check inhibition
    left_smooth_mono = spatial_smoothing(spec_loud_mono)
    right_smooth_zero = spatial_smoothing(torch.zeros_like(spec_loud_mono))
    inhib_L_mono, inhib_R_mono = binaural_inhibition(left_smooth_mono, right_smooth_zero)
    
    print(f"\nMonaural left (R=0):")
    print(f"  Inhibition factor L: {inhib_L_mono[0, 75]:.4f} (expect ~2.0)")
    print(f"  Total loudness: {loudness_mono_L.item():.4f} sone")
    print(f"  Left contribution: {loud_L_mono_L.item():.4f} sone")
    print(f"  Right contribution: {loud_R_mono_L.item():.4f} sone")
    
    print(f"\nMonaural right (L=0):")
    print(f"  Inhibition factor R: {inhib_R_mono[0, 75]:.4f} (expect ~2.0)")
    print(f"  Total loudness: {loudness_mono_R.item():.4f} sone")
    print(f"  Left contribution: {loud_L_mono_R.item():.4f} sone")
    print(f"  Right contribution: {loud_R_mono_R.item():.4f} sone")
    
    print(f"\nSymmetry check:")
    print(f"  L-only vs R-only ratio: {loudness_mono_L.item() / loudness_mono_R.item():.4f}")
    
    # ========== TEST 6: Dichotic stimuli - ILD variation ==========
    print("\n" + "-"*80)
    print("TEST 6: Dichotic stimuli - ILD variation (-40 to +40 dB)")
    print("-"*80)
    
    # ILD range
    ild_values = np.linspace(-40, 40, 81)
    
    # Test at multiple frequencies
    ild_results = {freq: {'loudness': [], 'loud_L': [], 'loud_R': [], 
                          'inhib_L': [], 'inhib_R': []} 
                   for freq in test_freqs}
    
    for freq in test_freqs:
        idx = test_freq_indices[freq]
        
        for ild in ild_values:
            # Create specific loudness at frequency with ILD
            # Reference level: 60 dB, ILD = L - R
            level_left_db = 60.0
            level_right_db = 60.0 - ild
            
            spec_loud_left = torch.zeros(1, 150)
            spec_loud_right = torch.zeros(1, 150)
            
            # Convert dB to specific loudness (approximate)
            spec_loud_left[0, idx] = 10 ** (level_left_db / 20.0) / 100.0
            spec_loud_right[0, idx] = 10 ** (level_right_db / 20.0) / 100.0
            
            # Compute loudness
            loudness, loud_L, loud_R = binaural_loudness(spec_loud_left, spec_loud_right)
            
            # Get inhibition factors
            left_smooth = spatial_smoothing(spec_loud_left)
            right_smooth = spatial_smoothing(spec_loud_right)
            inhib_L, inhib_R = binaural_inhibition(left_smooth, right_smooth)
            
            ild_results[freq]['loudness'].append(loudness.item())
            ild_results[freq]['loud_L'].append(loud_L.item())
            ild_results[freq]['loud_R'].append(loud_R.item())
            ild_results[freq]['inhib_L'].append(inhib_L[0, idx].item())
            ild_results[freq]['inhib_R'].append(inhib_R[0, idx].item())
    
    # Print some key values
    for freq in test_freqs:
        idx_ild_0 = np.argmin(np.abs(ild_values - 0))
        idx_ild_20 = np.argmin(np.abs(ild_values - 20))
        idx_ild_m20 = np.argmin(np.abs(ild_values - (-20)))
        
        print(f"\n  {freq} Hz:")
        print(f"    ILD=0 dB: Loudness={ild_results[freq]['loudness'][idx_ild_0]:.4f} sone")
        print(f"    ILD=+20 dB (L>R): Loudness={ild_results[freq]['loudness'][idx_ild_20]:.4f} sone")
        print(f"    ILD=-20 dB (L<R): Loudness={ild_results[freq]['loudness'][idx_ild_m20]:.4f} sone")
    
    # ========== TEST 7: With/without inhibition comparison ==========
    print("\n" + "-"*80)
    print("TEST 7: Loudness with and without binaural inhibition")
    print("-"*80)
    
    # Create test patterns
    spec_loud_test = torch.randn(1, 150).abs() * 2.0
    
    # With inhibition
    loudness_with, loud_L_with, loud_R_with = binaural_loudness(
        spec_loud_test, spec_loud_test
    )
    
    # Without inhibition (simple sum / 4)
    loudness_without = (spec_loud_test.sum() / 4.0 + spec_loud_test.sum() / 4.0).item()
    
    print(f"\nDiotic test pattern:")
    print(f"  With inhibition: {loudness_with.item():.4f} sone")
    print(f"  Without inhibition: {loudness_without:.4f} sone")
    print(f"  Inhibition effect: {(1 - loudness_with.item() / loudness_without) * 100:.1f}% reduction")
    
    # ========== VISUALIZATION ==========
    print("\n" + "-"*80)
    print("Creating comprehensive visualization...")
    print("-"*80)
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    freq_colors = {freq: colors[i] for i, freq in enumerate(test_freqs)}
    
    # === Plot 1: Gaussian kernel ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.plot(g, kernel, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Distance g (ERB)')
    ax1.set_ylabel('Weight W(g)')
    ax1.set_title('Gaussian Spatial Smoothing Kernel')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'σ = {smooth_params["sigma"]:.4f}\nΣW = {kernel.sum():.6f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Plot 2: Spatial smoothing examples ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    for freq in test_freqs:
        original = smoothing_results[freq]['original']
        smoothed = smoothing_results[freq]['smoothed']
        
        # Plot only non-zero region
        mask = (original > 0) | (smoothed > 0.01)
        if mask.any():
            ax2.plot(fc[mask], smoothed[mask], '-', linewidth=2, 
                    color=freq_colors[freq], label=f'{freq} Hz smoothed')
            ax2.plot(fc[mask], original[mask], '--', linewidth=1.5, 
                    color=freq_colors[freq], alpha=0.4, label=f'{freq} Hz original')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Specific Loudness (sone/ERB)')
    ax2.set_title('Spatial Smoothing Effect')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=7, ncol=2)
    
    # === Plot 3: Inhibition curve ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    ax3.semilogx(ratios_np, inhib_left_center, 'b-', linewidth=2, label='Left inhib (NR/NL)')
    ax3.semilogx(ratios_np, inhib_right_center, 'r-', linewidth=2, label='Right inhib (NL/NR)')
    ax3.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label='Max inhib (diotic)')
    ax3.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Min inhib (monaural)')
    ax3.axvline(x=1.0, color='k', linestyle=':', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Loudness Ratio')
    ax3.set_ylabel('Inhibition Factor')
    ax3.set_title('Binaural Inhibition: sech Function')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=8)
    ax3.set_ylim([0.9, 2.1])
    
    # === Plot 4: Diotic inhibition factors ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.semilogx(fc, inhib_L_diotic[0].numpy(), 'b-', linewidth=2, label='Left')
    ax4.semilogx(fc, inhib_R_diotic[0].numpy(), 'r--', linewidth=2, label='Right')
    ax4.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Inhibition Factor')
    ax4.set_title('Diotic Inhibition Factors (L=R)')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()
    ax4.set_ylim([0.95, 1.15])
    
    # === Plot 5: Dichotic inhibition factors for different ILDs ===
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Plot inhibition factors at 1 kHz for different ILDs
    freq_plot = 1000
    ild_plot_values = [-40, -20, 0, 20, 40]
    
    for ild_val in ild_plot_values:
        idx_ild = np.argmin(np.abs(ild_values - ild_val))
        inhib_L = ild_results[freq_plot]['inhib_L'][idx_ild]
        inhib_R = ild_results[freq_plot]['inhib_R'][idx_ild]
        
        ax5.scatter([ild_val], [inhib_L], marker='o', s=100, 
                   label=f'ILD={ild_val:+d} dB: L={inhib_L:.2f}')
        ax5.scatter([ild_val], [inhib_R], marker='s', s=100, alpha=0.6)
    
    ax5.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, linewidth=1, label='Max inhib')
    ax5.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Min inhib')
    ax5.set_xlabel('ILD (dB, L-R)')
    ax5.set_ylabel('Inhibition Factor')
    ax5.set_title(f'Dichotic Inhibition @ {freq_plot} Hz')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=7)
    ax5.set_ylim([0.8, 2.2])
    
    # === Plot 6: Loudness vs ILD for all frequencies ===
    ax6 = fig.add_subplot(gs[1, 2])
    
    for freq in test_freqs:
        loudness_vals = ild_results[freq]['loudness']
        ax6.plot(ild_values, loudness_vals, '-', linewidth=2, 
                color=freq_colors[freq], label=f'{freq} Hz')
    
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax6.set_xlabel('ILD (dB, L-R)')
    ax6.set_ylabel('Total Loudness (sone)')
    ax6.set_title('Loudness vs ILD')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # === Plot 7: Left/Right contributions vs ILD ===
    ax7 = fig.add_subplot(gs[2, 0])
    
    freq_plot = 1000
    ax7.plot(ild_values, ild_results[freq_plot]['loud_L'], 'b-', linewidth=2, label='Left')
    ax7.plot(ild_values, ild_results[freq_plot]['loud_R'], 'r-', linewidth=2, label='Right')
    ax7.plot(ild_values, ild_results[freq_plot]['loudness'], 'k--', linewidth=2, label='Total')
    ax7.axvline(x=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
    ax7.set_xlabel('ILD (dB, L-R)')
    ax7.set_ylabel('Loudness (sone)')
    ax7.set_title(f'Loudness Components @ {freq_plot} Hz')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # === Plot 8: Component comparison bar plot ===
    ax8 = fig.add_subplot(gs[2, 1])
    
    conditions = ['Diotic\n(L=R)', 'Monaural\nLeft', 'Monaural\nRight']
    loudness_vals = [loudness_diotic.item(), loudness_mono_L.item(), loudness_mono_R.item()]
    left_vals = [loud_L_diotic.item(), loud_L_mono_L.item(), 0]
    right_vals = [loud_R_diotic.item(), 0, loud_R_mono_R.item()]
    
    x = np.arange(len(conditions))
    width = 0.25
    
    ax8.bar(x - width, left_vals, width, label='Left', color='b', alpha=0.8)
    ax8.bar(x, right_vals, width, label='Right', color='r', alpha=0.8)
    ax8.bar(x + width, loudness_vals, width, label='Total', color='k', alpha=0.6)
    
    ax8.set_ylabel('Loudness (sone)')
    ax8.set_title('Loudness Components by Condition')
    ax8.set_xticks(x)
    ax8.set_xticklabels(conditions)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # === Plot 9: With/without inhibition comparison ===
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Compare at ILD=0 for all frequencies
    loudness_with_inhib = []
    loudness_without_inhib = []
    
    for freq in test_freqs:
        idx_ild_0 = np.argmin(np.abs(ild_values - 0))
        
        # With inhibition
        loud_with = ild_results[freq]['loudness'][idx_ild_0]
        loudness_with_inhib.append(loud_with)
        
        # Without inhibition (approximate: 2 × monaural)
        # Get reference level
        level_db = 60.0
        idx = test_freq_indices[freq]
        spec_loud_val = 10 ** (level_db / 20.0) / 100.0
        loud_without = 2 * spec_loud_val / 4.0  # Approximate
        loudness_without_inhib.append(loud_without)
    
    x = np.arange(len(test_freqs))
    width = 0.35
    
    ax9.bar(x - width/2, loudness_with_inhib, width, label='With inhibition', 
            color='green', alpha=0.8)
    ax9.bar(x + width/2, loudness_without_inhib, width, label='Without inhibition', 
            color='orange', alpha=0.8)
    
    ax9.set_ylabel('Loudness (sone)')
    ax9.set_title('Effect of Binaural Inhibition (ILD=0)')
    ax9.set_xticks(x)
    ax9.set_xticklabels([f'{f} Hz' for f in test_freqs])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle('Moore2016 Binaural Loudness - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    test_figures_dir = Path(__file__).parent.parent.parent / 'test_figures'
    test_figures_dir.mkdir(exist_ok=True)
    output_path = test_figures_dir / 'moore2016_binauralloudness.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figure saved: {output_path}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == '__main__':
    test_moore2016_binauralloudness()
