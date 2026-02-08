"""
FastKing2019 Modulation Filterbank Design - Test Suite

This test suite is identical to test_king2019_modfilt_design.py but uses
FastKing2019ModulationFilterbank (FFT-based) instead of the original IIR implementation.

Contents:
1. test_modulation_filterbank_qfactor: Tests Q-factor based configuration
2. test_modulation_filterbank_fixed: Tests fixed number configuration  
3. test_modulation_bandwidth_analysis: Analyzes bandwidth characteristics

Structure:
- Modulation frequency range: 2-150 Hz
- Q-factor based: Automatic spacing based on quality factor
- Fixed number: User-specified number of filters

Figures generated:
- modulation_filterbank_qfactor_fast_king2019.png: Q-factor configurations
- modulation_filterbank_fixed_fast_king2019.png: Fixed number configurations
- modulation_filterbank_bandwidth_fast_king2019.png: Bandwidth analysis

Note: The Fast version uses FFT-based convolution which is ~250x faster but
      introduces ~15% output error and ~13% gradient error compared to exact IIR.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfreqz
from torch_amt.common.modulation import FastKing2019ModulationFilterbank


def test_modulation_filterbank_qfactor():
    """Test modulation filterbank with different Q-factors."""
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("Fast Modulation Filterbank Design - Q-factor Configuration")
    print("="*80)
    
    # Test different Q-factors around reference value (Q=1.0)
    q_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(len(q_values), 2, figsize=(16, 4*len(q_values)))
    fig.suptitle('Fast Modulation Filterbank Design - Q-factor Control (FFT-based)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, q in enumerate(q_values):
        print(f"\n{'='*60}")
        print(f"Q-factor = {q}")
        print(f"{'='*60}")
        
        mod_bank = FastKing2019ModulationFilterbank(fs=44100, mflow=2.0, mfhigh=150.0, qfactor=q)
        
        print(f"  Sampling rate: {mod_bank.fs} Hz")
        print(f"  Modulation range: {mod_bank.mflow} - {mod_bank.mfhigh} Hz")
        print(f"  Q-factor: {mod_bank.qfactor}")
        print(f"  Number of filters: {len(mod_bank.mfc)}")
        print(f"  Center frequencies: {mod_bank.mfc.numpy()}")
        
        # Calculate step factor
        step_mfc = ((np.sqrt(4 * q**2 + 1) + 1) / (np.sqrt(4 * q**2 + 1) - 1))
        print(f"  Step factor: {step_mfc:.4f}")
        
        # Left column: Center frequencies (lollipop plot)
        ax_left = axes[idx, 0]
        markerline, stemlines, baseline = ax_left.stem(np.arange(len(mod_bank.mfc)), 
                                                       mod_bank.mfc.numpy(), basefmt=' ')
        markerline.set_markerfacecolor('C0')
        markerline.set_markeredgecolor('C0')
        markerline.set_markersize(8)
        
        # Add frequency values as text labels above each lollipop
        for i, fc in enumerate(mod_bank.mfc.numpy()):
            ax_left.text(i, fc, f'{fc:.1f}Hz', 
                        ha='center', va='bottom', fontsize=7, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                 alpha=0.7, edgecolor='none'))
        
        ax_left.set_xlabel('Filter Index', fontsize=10)
        ax_left.set_ylabel('Center Frequency [Hz]', fontsize=10)
        ax_left.set_title(f'Q={q}: {len(mod_bank.mfc)} filters, step={step_mfc:.3f}', 
                         fontsize=11, fontweight='bold')
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim([0, mod_bank.mfhigh * 1.1])
        
        # Add text with frequency ratios
        if len(mod_bank.mfc) > 1:
            ratios = mod_bank.mfc[1:] / mod_bank.mfc[:-1]
            ratio_mean = ratios.mean().item()
            ratio_std = ratios.std().item()
            ax_left.text(0.02, 0.98, f'Ratio: {ratio_mean:.3f} ± {ratio_std:.3f}',
                        transform=ax_left.transAxes, verticalalignment='top',
                        fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        # Right column: Frequency responses
        ax_right = axes[idx, 1]
        
        colors = plt.cm.viridis(np.linspace(0, 1, mod_bank.num_filters))
        
        for i in range(mod_bank.num_filters):
            sos = mod_bank.sos_stack[i]
            sos_np = sos.detach().cpu().numpy()
            w, h = sosfreqz(sos_np, worN=8192, fs=mod_bank.fs)
            
            # Plot magnitude response in dB
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            ax_right.semilogx(w, mag_db, linewidth=1.5, 
                             label=f'{mod_bank.mfc[i]:.1f} Hz',
                             alpha=0.8, color=colors[i])
        
        ax_right.set_xlabel('Frequency [Hz]', fontsize=10)
        ax_right.set_ylabel('Magnitude [dB]', fontsize=10)
        ax_right.set_title(f'Q={q}: Frequency Responses', fontsize=11, fontweight='bold')
        ax_right.grid(True, alpha=0.3, which='both')
        ax_right.set_xlim([mod_bank.mflow / 2, mod_bank.mfhigh * 2])
        ax_right.set_ylim([-50, 5])
        ax_right.axhline(y=-3, color='r', linestyle='--', alpha=0.5, linewidth=1.5, 
                        label='-3dB')
        ax_right.legend(loc='lower left', fontsize=7, ncol=2, framealpha=0.9)
        
        # Check logarithmic spacing
        if len(mod_bank.mfc) > 2:
            ratios = mod_bank.mfc[1:] / mod_bank.mfc[:-1]
            print(f"  Frequency ratios (log spacing check):")
            print(f"    Mean: {ratios.mean():.4f}")
            print(f"    Std: {ratios.std():.4f}")
            print(f"    Min: {ratios.min():.4f}")
            print(f"    Max: {ratios.max():.4f}")
            
            if ratios.std() < 0.1:
                print(f"  ✓ Logarithmic spacing verified (std < 0.1)")
            else:
                print(f"  ⚠ Non-uniform spacing detected (std ≥ 0.1)")
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_qfactor_fast_king2019.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


def test_modulation_filterbank_fixed():
    """Test modulation filterbank with fixed number of filters."""
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("Fast Modulation Filterbank Design - Fixed Number Configuration")
    print("="*80)
    
    # Test different numbers of filters
    nmod_values = [5, 10, 20, 41]
    
    fig, axes = plt.subplots(len(nmod_values), 2, figsize=(16, 4*len(nmod_values)))
    fig.suptitle('Fast Modulation Filterbank Design - Fixed Number Control (FFT-based)', 
                 fontsize=16, fontweight='bold')
    
    for idx, nmod in enumerate(nmod_values):
        print(f"\n{'='*60}")
        print(f"Number of filters = {nmod}")
        print(f"{'='*60}")
        
        mod_bank = FastKing2019ModulationFilterbank(fs=44100, mflow=2.0, mfhigh=150.0, 
                                                     qfactor=1.0, nmod=nmod)
        
        print(f"  Sampling rate: {mod_bank.fs} Hz")
        print(f"  Modulation range: {mod_bank.mflow} - {mod_bank.mfhigh} Hz")
        print(f"  Q-factor (reference): {mod_bank.qfactor}")
        print(f"  Number of filters: {len(mod_bank.mfc)}")
        print(f"  Center frequencies: {mod_bank.mfc.numpy()}")
        
        # Left column: Center frequencies (lollipop plot)
        ax_left = axes[idx, 0]
        markerline, stemlines, baseline = ax_left.stem(np.arange(len(mod_bank.mfc)), 
                                                       mod_bank.mfc.numpy(), basefmt=' ')
        markerline.set_markerfacecolor('C1')
        markerline.set_markeredgecolor('C1')
        markerline.set_markersize(8)
        
        # Add frequency values as text labels above each lollipop
        for i, fc in enumerate(mod_bank.mfc.numpy()):
            # Show labels for fewer filters to avoid overlap
            if nmod <= 20 or i % 2 == 0:
                ax_left.text(i, fc, f'{fc:.1f}Hz', 
                            ha='center', va='bottom', fontsize=7, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                                     alpha=0.7, edgecolor='none'))
        
        ax_left.set_xlabel('Filter Index', fontsize=10)
        ax_left.set_ylabel('Center Frequency [Hz]', fontsize=10)
        ax_left.set_title(f'N={nmod} filters (linear log-spacing)', 
                         fontsize=11, fontweight='bold')
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim([0, mod_bank.mfhigh * 1.1])
        
        # Add text with frequency ratios
        if len(mod_bank.mfc) > 1:
            ratios = mod_bank.mfc[1:] / mod_bank.mfc[:-1]
            ratio_mean = ratios.mean().item()
            ratio_std = ratios.std().item()
            ax_left.text(0.02, 0.98, f'Ratio: {ratio_mean:.3f} ± {ratio_std:.3f}',
                        transform=ax_left.transAxes, verticalalignment='top',
                        fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        # Right column: Frequency responses
        ax_right = axes[idx, 1]
        
        colors = plt.cm.plasma(np.linspace(0, 1, mod_bank.num_filters))
        
        for i in range(mod_bank.num_filters):
            sos = mod_bank.sos_stack[i]
            sos_np = sos.detach().cpu().numpy()
            w, h = sosfreqz(sos_np, worN=8192, fs=mod_bank.fs)
            
            # Plot magnitude response in dB
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            # Show labels for every 3rd filter to avoid clutter
            show_label = (i % 3 == 0) if nmod > 20 else True
            ax_right.semilogx(w, mag_db, linewidth=1.5, 
                             label=f'{mod_bank.mfc[i]:.1f} Hz' if show_label else None,
                             alpha=0.7, color=colors[i])
        
        ax_right.set_xlabel('Frequency [Hz]', fontsize=10)
        ax_right.set_ylabel('Magnitude [dB]', fontsize=10)
        ax_right.set_title(f'N={nmod}: Frequency Responses', fontsize=11, fontweight='bold')
        ax_right.grid(True, alpha=0.3, which='both')
        ax_right.set_xlim([mod_bank.mflow / 2, mod_bank.mfhigh * 2])
        ax_right.set_ylim([-50, 5])
        ax_right.axhline(y=-3, color='r', linestyle='--', alpha=0.5, linewidth=1.5, 
                        label='-3dB')
        if nmod <= 20:
            ax_right.legend(loc='lower left', fontsize=7, ncol=2, framealpha=0.9)
        else:
            ax_right.legend(loc='lower left', fontsize=6, ncol=3, framealpha=0.9)
        
        # Check logarithmic spacing
        if len(mod_bank.mfc) > 2:
            ratios = mod_bank.mfc[1:] / mod_bank.mfc[:-1]
            print(f"  Frequency ratios (log spacing check):")
            print(f"    Mean: {ratios.mean():.4f}")
            print(f"    Std: {ratios.std():.4f}")
            print(f"    Min: {ratios.min():.4f}")
            print(f"    Max: {ratios.max():.4f}")
            
            if ratios.std() < 0.1:
                print(f"  ✓ Logarithmic spacing verified (std < 0.1)")
            else:
                print(f"  ⚠ Non-uniform spacing detected (std ≥ 0.1)")
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_fixed_fast_king2019.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


def test_modulation_bandwidth_analysis():
    """Analyze bandwidth and Q-factor of modulation filters for all configurations."""
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("Fast Modulation Filter Bandwidth Analysis - All Configurations")
    print("="*80)
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Fast Modulation Filter Bandwidth Analysis - All Configurations (FFT-based)', 
                 fontsize=16, fontweight='bold')
    
    # =========================================================================
    # LEFT COLUMN: Q-factor based configurations
    # =========================================================================
    print("\n" + "="*60)
    print("Q-factor Based Configurations")
    print("="*60)
    
    q_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    colors_q = plt.cm.viridis(np.linspace(0, 1, len(q_values)))
    
    for idx, q in enumerate(q_values):
        mod_bank = FastKing2019ModulationFilterbank(fs=44100, mflow=2.0, mfhigh=150.0, qfactor=q)
        
        print(f"\nQ={q}: {len(mod_bank.mfc)} filters")
        
        bandwidths = []
        center_freqs = []
        
        for i in range(mod_bank.num_filters):
            sos = mod_bank.sos_stack[i]
            sos_np = sos.detach().cpu().numpy()
            w, h = sosfreqz(sos_np, worN=16384, fs=mod_bank.fs)
            
            # Find -3dB bandwidth
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            peak_db = mag_db.max()
            cutoff_db = peak_db - 3
            
            # Find frequencies where magnitude crosses -3dB
            above_cutoff = mag_db > cutoff_db
            crossings = np.diff(above_cutoff.astype(int))
            
            if np.sum(crossings != 0) >= 2:
                lower_idx = np.where(crossings > 0)[0][0] if np.any(crossings > 0) else 0
                upper_idx = np.where(crossings < 0)[0][0] if np.any(crossings < 0) else len(w) - 1
                
                f_lower = w[lower_idx]
                f_upper = w[upper_idx]
                bw = f_upper - f_lower
                fc = mod_bank.mfc[i].item()
                
                bandwidths.append(bw)
                center_freqs.append(fc)
        
        if len(center_freqs) > 0:
            # Plot 1 (top-left): Bandwidth vs Center Frequency
            axes[0, 0].plot(center_freqs, bandwidths, 'o-', linewidth=2, markersize=7, 
                           color=colors_q[idx], label=f'Q={q} ({len(center_freqs)} filters)',
                           alpha=0.8)
            
            # Plot 2 (bottom-left): Q-factor vs Center Frequency
            q_factors = np.array(center_freqs) / np.array(bandwidths)
            axes[1, 0].plot(center_freqs, q_factors, 's-', linewidth=2, markersize=7, 
                           color=colors_q[idx], label=f'Q={q}', alpha=0.8)
    
    # Configure top-left plot (Bandwidth - Q-based)
    axes[0, 0].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[0, 0].set_ylabel('Bandwidth [Hz]', fontsize=11)
    axes[0, 0].set_title('Bandwidth vs Center Frequency (Q-factor Control)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    # Configure bottom-left plot (Q-factor - Q-based)
    axes[1, 0].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[1, 0].set_ylabel('Q-factor (fc/BW)', fontsize=11)
    axes[1, 0].set_title('Q-factor vs Center Frequency (Q-factor Control)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    # Add target Q lines
    for q in q_values:
        axes[1, 0].axhline(y=q, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    axes[1, 0].legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    # =========================================================================
    # RIGHT COLUMN: Fixed number configurations
    # =========================================================================
    print("\n" + "="*60)
    print("Fixed Number Configurations")
    print("="*60)
    
    nmod_values = [5, 10, 20, 41]
    colors_n = plt.cm.plasma(np.linspace(0, 1, len(nmod_values)))
    
    for idx, nmod in enumerate(nmod_values):
        mod_bank = FastKing2019ModulationFilterbank(fs=44100, mflow=2.0, mfhigh=150.0, 
                                                     qfactor=1.0, nmod=nmod)
        
        print(f"\nN={nmod}: {len(mod_bank.mfc)} filters")
        
        bandwidths = []
        center_freqs = []
        
        for i in range(mod_bank.num_filters):
            sos = mod_bank.sos_stack[i]
            sos_np = sos.detach().cpu().numpy()
            w, h = sosfreqz(sos_np, worN=16384, fs=mod_bank.fs)
            
            # Find -3dB bandwidth
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            peak_db = mag_db.max()
            cutoff_db = peak_db - 3
            
            # Find frequencies where magnitude crosses -3dB
            above_cutoff = mag_db > cutoff_db
            crossings = np.diff(above_cutoff.astype(int))
            
            if np.sum(crossings != 0) >= 2:
                lower_idx = np.where(crossings > 0)[0][0] if np.any(crossings > 0) else 0
                upper_idx = np.where(crossings < 0)[0][0] if np.any(crossings < 0) else len(w) - 1
                
                f_lower = w[lower_idx]
                f_upper = w[upper_idx]
                bw = f_upper - f_lower
                fc = mod_bank.mfc[i].item()
                
                bandwidths.append(bw)
                center_freqs.append(fc)
        
        if len(center_freqs) > 0:
            # Plot 3 (top-right): Bandwidth vs Center Frequency
            axes[0, 1].plot(center_freqs, bandwidths, 'o-', linewidth=2, markersize=7, 
                           color=colors_n[idx], label=f'N={nmod} ({len(center_freqs)} filters)',
                           alpha=0.8)
            
            # Plot 4 (bottom-right): Q-factor vs Center Frequency
            q_factors = np.array(center_freqs) / np.array(bandwidths)
            axes[1, 1].plot(center_freqs, q_factors, 's-', linewidth=2, markersize=7, 
                           color=colors_n[idx], label=f'N={nmod}', alpha=0.8)
    
    # Configure top-right plot (Bandwidth - Fixed N)
    axes[0, 1].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[0, 1].set_ylabel('Bandwidth [Hz]', fontsize=11)
    axes[0, 1].set_title('Bandwidth vs Center Frequency (Fixed Number Control)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    # Configure bottom-right plot (Q-factor - Fixed N)
    axes[1, 1].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[1, 1].set_ylabel('Q-factor (fc/BW)', fontsize=11)
    axes[1, 1].set_title('Q-factor vs Center Frequency (Fixed Number Control)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    # Add reference Q=1.0 line
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2,
                      label='Q=1.0 (reference)')
    axes[1, 1].legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_bandwidth_fast_king2019.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FAST MODULATION FILTERBANK DESIGN VERIFICATION (FFT-based)")
    print("="*80)
    print("\nModulation frequency range: 2-150 Hz")
    print("  2-10 Hz: Syllable rate, speech rhythm")
    print("  10-50 Hz: Phoneme rate, fine temporal information")
    print("  50-150 Hz: Pitch and fine modulation information")
    print("\nImplementation: FastKing2019ModulationFilterbank")
    print("  - FFT-based convolution (~250x faster)")
    print("  - ~15% output error, ~13% gradient error vs exact IIR")
    print("  - Ideal for inference/feature extraction")
    print("="*80)
    
    test_modulation_filterbank_qfactor()
    test_modulation_filterbank_fixed()
    test_modulation_bandwidth_analysis()
    
    print("\n\n" + "="*80)
    print("ALL DESIGN VERIFICATION TESTS COMPLETED!")
    print("Generated plots:")
    print("  1. modulation_filterbank_qfactor_fast_king2019.png - Q-factor control (5 configs)")
    print("  2. modulation_filterbank_fixed_fast_king2019.png - Fixed number control (4 configs)")
    print("  3. modulation_filterbank_bandwidth_fast_king2019.png - Bandwidth analysis")
    print("="*80)
