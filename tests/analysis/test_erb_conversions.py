"""
ERB Conversions - Test Suite

Contents:
1. test_erb_conversion_comparison: Compares two ERB-rate formulas from Glasberg & Moore (1990)
   - fc2erb/erb2fc: Natural logarithm version (filterbank spacing)
   - f2erbrate/erbrate2f: Log10 version (loudness models)

Structure:
- Forward conversion (frequency → ERB-rate) comparison
- Inverse conversion (ERB-rate → frequency) comparison
- Filter spacing analysis using both methods
- Roundtrip accuracy verification

Figures generated:
- erb_conversion_comparison.png: 2x2 grid comparing forward/inverse conversions
- erb_spacing_comparison.png: 2x2 grid analyzing filter spacing differences
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import fc2erb, erb2fc, f2erbrate, erbrate2f


def test_erb_conversion_comparison():
    """Compare fc2erb vs f2erbrate and erb2fc vs erbrate2f."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ERB Conversion Methods Comparison")
    print("=" * 80)
    
    # Frequency range: 20 Hz to 20 kHz
    freqs = torch.logspace(torch.log10(torch.tensor(20.0)),
                           torch.log10(torch.tensor(20000.0)),
                           1000)
    
    # Method 1: Natural log version (fc2erb)
    erb_natlog = fc2erb(freqs)
    freqs_recovered_natlog = erb2fc(erb_natlog)
    
    # Method 2: Log10 version (f2erbrate)
    erb_log10 = f2erbrate(freqs)
    freqs_recovered_log10 = erbrate2f(erb_log10)
    
    # Check roundtrip accuracy
    error_natlog = torch.abs(freqs - freqs_recovered_natlog).max().item()
    error_log10 = torch.abs(freqs - freqs_recovered_log10).max().item()
    
    print(f"\nRoundtrip accuracy:")
    print(f"  fc2erb -> erb2fc: max error = {error_natlog:.2e} Hz")
    print(f"  f2erbrate -> erbrate2f: max error = {error_log10:.2e} Hz")
    
    # Compare ERB-rate values at specific frequencies
    test_freqs = torch.tensor([100.0, 500.0, 1000.0, 4000.0, 10000.0])
    print(f"\nERB-rate values at specific frequencies:")
    print(f"{'Frequency (Hz)':<15} {'fc2erb (natlog)':<20} {'f2erbrate (log10)':<20} {'Difference':<15}")
    print("-" * 70)
    for f in test_freqs:
        erb_nl = fc2erb(f).item()
        erb_l10 = f2erbrate(f).item()
        diff = erb_l10 - erb_nl
        print(f"{f.item():<15.0f} {erb_nl:<20.4f} {erb_l10:<20.4f} {diff:<15.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('ERB Conversion Methods Comparison\n'
                 'Natural log (fc2erb) vs Log10 (f2erbrate)', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Frequency to ERB-rate
    # Left: Natural log
    ax = axes[0, 0]
    ax.semilogx(freqs.numpy(), erb_natlog.numpy(), 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('ERB-rate')
    ax.set_title('fc2erb (natural log)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 20000])
    
    # Right: Log10
    ax = axes[0, 1]
    ax.semilogx(freqs.numpy(), erb_log10.numpy(), 'r-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('ERB-rate (Cams)')
    ax.set_title('f2erbrate (log10)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 20000])
    
    # Row 2: ERB-rate to Frequency (inverse functions)
    # Test range: 0 to 40 ERBs
    erb_range = torch.linspace(0, 40, 1000)
    
    # Left: Natural log inverse
    ax = axes[1, 0]
    freqs_from_natlog = erb2fc(erb_range)
    ax.semilogy(erb_range.numpy(), freqs_from_natlog.numpy(), 'b-', linewidth=2)
    ax.set_xlabel('ERB-rate')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('erb2fc (natural log inverse)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([10, 20000])
    
    # Right: Log10 inverse
    ax = axes[1, 1]
    freqs_from_log10 = erbrate2f(erb_range)
    ax.semilogy(erb_range.numpy(), freqs_from_log10.numpy(), 'r-', linewidth=2)
    ax.set_xlabel('ERB-rate (Cams)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('erbrate2f (log10 inverse)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([10, 20000])
    
    plt.tight_layout()
    plt.savefig(TEST_FIGURES_DIR / 'erb_conversion_comparison.png', dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: erb_conversion_comparison.png")
    
    # Additional analysis: ERB spacing comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('ERB Filter Spacing Comparison\n'
                  'Using different conversion methods', 
                  fontsize=14, fontweight='bold')
    
    # Generate ERB-spaced frequencies using both methods
    erb_min = fc2erb(torch.tensor(50.0)).item()
    erb_max = fc2erb(torch.tensor(15000.0)).item()
    erb_step = 0.25  # Glasberg2002 uses 0.25
    
    # Method 1: Natural log
    erb_numbers_nl = torch.arange(erb_min, erb_max + erb_step, erb_step)
    fc_spaced_nl = erb2fc(erb_numbers_nl)
    
    # Method 2: Log10
    erb_min_l10 = f2erbrate(torch.tensor(50.0)).item()
    erb_max_l10 = f2erbrate(torch.tensor(15000.0)).item()
    erb_numbers_l10 = torch.arange(erb_min_l10, erb_max_l10 + erb_step, erb_step)
    fc_spaced_l10 = erbrate2f(erb_numbers_l10)
    
    print(f"\nERB-spaced filter centers (step = {erb_step}):")
    print(f"  Natural log method: {len(fc_spaced_nl)} filters from {fc_spaced_nl[0]:.1f} to {fc_spaced_nl[-1]:.1f} Hz")
    print(f"  Log10 method: {len(fc_spaced_l10)} filters from {fc_spaced_l10[0]:.1f} to {fc_spaced_l10[-1]:.1f} Hz")
    
    # Top-left: Filter center frequencies (linear scale)
    ax = axes2[0, 0]
    ax.plot(fc_spaced_nl.numpy(), 'b.-', markersize=3, linewidth=1, label='natlog', alpha=0.7)
    ax.plot(fc_spaced_l10.numpy(), 'r.-', markersize=3, linewidth=1, label='log10', alpha=0.7)
    ax.set_xlabel('Filter Index')
    ax.set_ylabel('Center Frequency (Hz)')
    ax.set_title('ERB-Spaced Filter Centers (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Filter center frequencies (log scale)
    ax = axes2[0, 1]
    ax.semilogy(fc_spaced_nl.numpy(), 'b.-', markersize=3, linewidth=1, label='natlog', alpha=0.7)
    ax.semilogy(fc_spaced_l10.numpy(), 'r.-', markersize=3, linewidth=1, label='log10', alpha=0.7)
    ax.set_xlabel('Filter Index')
    ax.set_ylabel('Center Frequency (Hz)')
    ax.set_title('ERB-Spaced Filter Centers (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Frequency spacing between adjacent filters
    spacing_nl = torch.diff(fc_spaced_nl)
    spacing_l10 = torch.diff(fc_spaced_l10)
    
    ax = axes2[1, 0]
    ax.plot(spacing_nl.numpy(), 'b.-', markersize=3, linewidth=1, label='natlog', alpha=0.7)
    ax.plot(spacing_l10.numpy(), 'r.-', markersize=3, linewidth=1, label='log10', alpha=0.7)
    ax.set_xlabel('Filter Index')
    ax.set_ylabel('Spacing (Hz)')
    ax.set_title('Frequency Spacing Between Adjacent Filters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Difference in center frequencies between methods
    # Need to interpolate since different number of filters
    min_len = min(len(fc_spaced_nl), len(fc_spaced_l10))
    ax = axes2[1, 1]
    diff_fc = fc_spaced_l10[:min_len] - fc_spaced_nl[:min_len]
    ax.plot(diff_fc.numpy(), 'g.-', markersize=3, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Filter Index')
    ax.set_ylabel('Frequency Difference (Hz)\n(log10 - natlog)')
    ax.set_title('Difference in Filter Centers')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(TEST_FIGURES_DIR / 'erb_spacing_comparison.png', dpi=600, bbox_inches='tight')
    print(f"✓ Saved: erb_spacing_comparison.png")

if __name__ == '__main__':
    test_erb_conversion_comparison()
