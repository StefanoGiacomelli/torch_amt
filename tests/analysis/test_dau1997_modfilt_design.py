"""
Dau1997 Modulation Filterbank Design - Test Suite

Contents:
1. test_modulation_filterbank_qfactor: Tests Q-factor control (1.0-3.0)
2. test_modulation_filterbank_umf_variation: Tests upper modulation frequency variation
3. test_modulation_filterbank_maxmfc: Tests max_mfc parameter control
4. test_modulation_bandwidth_analysis: Analyzes filter bandwidths
5. test_hybrid_spacing_verification: Verifies linear+exponential spacing

Structure:
- Hybrid spacing: Linear (0-10 Hz, step=5Hz) + Exponential (>10 Hz)
- fc-dependent: umf = min(0.25*fc, 150 Hz)
- Includes mfc=0: Explicit lowpass filter at 2.5 Hz
- Default Q = 2.0 (vs 1.0 for King2019)

Figures generated:
- modulation_filterbank_qfactor_dau1997.png: Q-factor configurations
- modulation_filterbank_umf_variation_dau1997.png: umf control
- modulation_filterbank_maxmfc_variation_dau1997.png: max_mfc control
- modulation_filterbank_bandwidth_dau1997.png: Bandwidth analysis
- modulation_filterbank_spacing_dau1997.png: Hybrid spacing verification
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_amt.common import ModulationFilterbank


def test_modulation_filterbank_qfactor():
    """Test modulation filterbank with different Q-factors."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("DAU1997 Modulation Filterbank Design - Q-factor Configuration")
    print("="*80)
    
    # Test different Q-factors (default for DAU1997 is Q=2.0)
    q_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    fc = torch.tensor([1000.0])  # Single reference auditory channel
    
    fig, axes = plt.subplots(len(q_values), 2, figsize=(16, 4*len(q_values)))
    fig.suptitle('DAU1997 Modulation Filterbank Design - Q-factor Control', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, q in enumerate(q_values):
        print(f"\n{'='*60}")
        print(f"Q-factor = {q}, fc = {fc[0]:.0f} Hz")
        print(f"{'='*60}")
        
        modfb = ModulationFilterbank(fs=44100, fc=fc, Q=q, max_mfc=150.0)
        mfc = modfb.mfc[0]  # Get mfc for first (only) channel
        
        print(f"  Sampling rate: {modfb.fs} Hz")
        print(f"  Auditory channel fc: {fc[0]:.0f} Hz")
        print(f"  Q-factor: {q}")
        print(f"  Number of filters: {len(mfc)}")
        print(f"  Center frequencies: {mfc.numpy()}")
        
        # Calculate ex factor for exponential region
        ex = (1 + 1/(2*q)) / (1 - 1/(2*q))
        print(f"  Exponential factor (ex): {ex:.4f}")
        
        # Identify linear vs exponential regions
        linear_mask = mfc <= 10.0
        n_linear = linear_mask.sum().item()
        n_exp = len(mfc) - n_linear
        print(f"  Linear region (<= 10 Hz): {n_linear} filters")
        print(f"  Exponential region (> 10 Hz): {n_exp} filters")
        
        # Left column: Center frequencies (lollipop plot)
        ax_left = axes[idx, 0]
        
        # Plot with colors for linear vs exponential
        colors = ['green' if m <= 10.0 else 'orange' for m in mfc.numpy()]
        markerline, stemlines, baseline = ax_left.stem(np.arange(len(mfc)), 
                                                       mfc.numpy(), basefmt=' ')
        # Set colors for stems
        stemlines.set_colors(colors)
        markerline.set_markerfacecolor('none')
        markerline.set_markeredgecolor('black')
        markerline.set_markersize(8)
        markerline.set_markeredgewidth(1.5)
        
        # Add frequency values as text labels
        for i, fc_val in enumerate(mfc.numpy()):
            color = 'green' if fc_val <= 10.0 else 'orange'
            ax_left.text(i, fc_val, f'{fc_val:.1f}Hz', 
                        ha='center', va='bottom', fontsize=7, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                 alpha=0.5, edgecolor='black'))
        
        ax_left.set_xlabel('Filter Index', fontsize=10)
        ax_left.set_ylabel('Center Frequency [Hz]', fontsize=10)
        ax_left.set_title(f'Q={q}: {len(mfc)} filters (green=linear, orange=exponential)', 
                         fontsize=11)
        ax_left.set_xticks(np.arange(len(mfc)))
        ax_left.set_xticklabels([str(i) for i in range(len(mfc))])
        ax_left.grid(True, alpha=0.3)
        ax_left.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
                       label='10 Hz transition')
        ax_left.legend(fontsize=9, loc='center left')
        
        # Right column: Frequency spacing analysis
        ax_right = axes[idx, 1]
        
        if len(mfc) > 1:
            # Calculate frequency steps
            steps = mfc[1:] - mfc[:-1]
            ratios = mfc[1:] / mfc[:-1]
            
            # Plot steps and ratios
            x_indices = np.arange(1, len(mfc))
            
            ax_right_twin = ax_right.twinx()
            
            # Steps (bar plot)
            bars = ax_right.bar(x_indices, steps.numpy(), alpha=0.6, 
                               color=['green' if mfc[i] <= 10.0 else 'orange' 
                                      for i in range(1, len(mfc))],
                               label='Frequency step [Hz]')
            
            # Ratios (line plot)
            line = ax_right_twin.plot(x_indices, ratios.numpy(), 'r-o', 
                                     linewidth=2, markersize=6, 
                                     label=f'Frequency ratio (target ex={ex:.3f})')
            ax_right_twin.axhline(y=ex, color='red', linestyle='--', alpha=0.3, 
                                 linewidth=2)
            
            ax_right.set_xlabel('Filter Index', fontsize=10)
            ax_right.set_ylabel('Frequency Step [Hz]', fontsize=10, color='black')
            ax_right_twin.set_ylabel('Frequency Ratio', fontsize=10, color='red')
            ax_right.set_title(f'Q={q}: Spacing Analysis', fontsize=11)
            ax_right.set_xticks(x_indices)
            ax_right.set_xticklabels([str(int(i)) for i in x_indices])
            ax_right.grid(True, alpha=0.3)
            ax_right.tick_params(axis='y', labelcolor='black')
            ax_right_twin.tick_params(axis='y', labelcolor='red')
            
            # Combined legend
            lines1, labels1 = ax_right.get_legend_handles_labels()
            lines2, labels2 = ax_right_twin.get_legend_handles_labels()
            ax_right.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center left')
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_qfactor_dau1997.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


def test_modulation_filterbank_umf_variation():
    """Test modulation filterbank with different upper modulation frequencies (umf)."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("DAU1997 Modulation Filterbank Design - umf Variation")
    print("="*80)
    
    # Test different upper modulation frequencies
    # Keep fc fixed at 1000 Hz, vary umf by changing max_mfc
    umf_values = [10, 25, 75, 100, 150]  # Hz
    fc = torch.tensor([1000.0])  # Fixed auditory channel
    q = 2.0  # Default for DAU1997
    
    fig, axes = plt.subplots(len(umf_values), 2, figsize=(16, 4*len(umf_values)))
    fig.suptitle('DAU1997 Modulation Filterbank Design - umf (Upper Modulation Frequency) Control', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, umf in enumerate(umf_values):
        print(f"\n{'='*60}")
        print(f"Upper modulation freq (umf) = {umf} Hz, fc = 1000 Hz, Q = {q}")
        print(f"{'='*60}")
        
        modfb = ModulationFilterbank(fs=44100, fc=fc, Q=q, max_mfc=umf)
        mfc = modfb.mfc[0]
        
        print(f"  Upper modulation freq (umf): {umf:.1f} Hz")
        print(f"  Number of filters: {len(mfc)}")
        print(f"  Center frequencies: {mfc.numpy()}")
        
        # Left column: Center frequencies (lollipop plot)
        ax_left = axes[idx, 0]
        
        colors = ['green' if m <= 10.0 else 'orange' for m in mfc.numpy()]
        markerline, stemlines, baseline = ax_left.stem(np.arange(len(mfc)), 
                                                       mfc.numpy(), basefmt=' ')
        stemlines.set_colors(colors)
        markerline.set_markerfacecolor('none')
        markerline.set_markeredgecolor('black')
        markerline.set_markersize(8)
        
        # Add frequency values (skip some if too many)
        show_all = len(mfc) <= 10
        for i, mfc_val in enumerate(mfc.numpy()):
            if show_all or i % 2 == 0 or i == len(mfc) - 1:
                color = 'green' if mfc_val <= 10.0 else 'orange'
                ax_left.text(i, mfc_val, f'{mfc_val:.1f}Hz', 
                            ha='center', va='bottom', fontsize=7, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                     alpha=0.5, edgecolor='black'))
        
        ax_left.set_xlabel('Filter Index', fontsize=10)
        ax_left.set_ylabel('Center Frequency [Hz]', fontsize=10)
        ax_left.set_title(f'umf={umf:.1f}Hz → {len(mfc)} filters', 
                         fontsize=11)
        ax_left.set_xticks(np.arange(len(mfc)))
        ax_left.set_xticklabels([str(i) for i in range(len(mfc))])
        ax_left.grid(True, alpha=0.3)
        ax_left.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax_left.set_ylim([0, max(umf * 1.1, 160)])
        
        # Right column: Frequency responses of modulation filters
        ax_right = axes[idx, 1]
        
        # Generate frequency responses for all filters
        freqs = np.logspace(-1, np.log10(200), 500)  # 0.1 to 200 Hz
        
        for i, mfc_val in enumerate(mfc.numpy()):
            if mfc_val == 0:
                # Lowpass filter at 2.5 Hz
                fc_lp = 2.5
                H = 1 / np.sqrt(1 + (freqs / fc_lp)**2)
                color = 'blue'
                label = f'LP (2.5 Hz)'
            else:
                # Bandpass filter with Q=2.0
                if mfc_val < 10:
                    bw = 5.0
                else:
                    bw = mfc_val / q
                # Second-order bandpass response
                H = (freqs / mfc_val) / np.sqrt((1 - (freqs / mfc_val)**2)**2 + (freqs / (mfc_val * q))**2)
                color = 'green' if mfc_val <= 10.0 else 'orange'
                label = f'{mfc_val:.1f} Hz'
            
            # Convert to dB
            H_dB = 20 * np.log10(H + 1e-10)  # Add small value to avoid log(0)
            ax_right.plot(freqs, H_dB, linewidth=1.5, color=color, alpha=0.7, label=label)
        
        ax_right.set_xlabel('Modulation Frequency [Hz]', fontsize=10)
        ax_right.set_ylabel('Amplitude Response [dB]', fontsize=10)
        ax_right.set_title(f'umf={umf:.1f}Hz: Filter Frequency Responses', fontsize=11)
        ax_right.set_xscale('log')
        ax_right.grid(True, alpha=0.3, which='both')
        ax_right.set_xlim([0.1, 200])
        ax_right.set_ylim([-40, 10])
        ax_right.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_right.legend(fontsize=7, ncol=2, loc='upper right')
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_umf_variation_dau1997.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


def test_modulation_filterbank_maxmfc():
    """Test modulation filterbank with different max_mfc values."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("DAU1997 Modulation Filterbank Design - max_mfc Variation")
    print("="*80)
    
    # Test different max modulation frequencies
    max_mfc_values = [100, 150, 200, 250]
    fc = torch.tensor([1000.0])  # Reference auditory channel
    q = 2.0
    
    fig, axes = plt.subplots(len(max_mfc_values), 2, figsize=(16, 4*len(max_mfc_values)))
    fig.suptitle('DAU1997 Modulation Filterbank Design - max_mfc (Maximum Modulation Frequency) Control', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, max_mfc in enumerate(max_mfc_values):
        print(f"\n{'='*60}")
        print(f"max_mfc = {max_mfc} Hz, fc = 1000 Hz, Q = {q}")
        print(f"{'='*60}")
        
        modfb = ModulationFilterbank(fs=44100, fc=fc, Q=q, max_mfc=max_mfc)
        mfc = modfb.mfc[0]
        
        umf = min(1000 * 0.25, max_mfc)
        
        print(f"  Upper modulation freq (umf): {umf:.1f} Hz")
        print(f"  Number of filters: {len(mfc)}")
        print(f"  Center frequencies: {mfc.numpy()}")
        
        # Left column: Center frequencies
        ax_left = axes[idx, 0]
        
        colors = ['green' if m <= 10.0 else 'orange' for m in mfc.numpy()]
        markerline, stemlines, baseline = ax_left.stem(np.arange(len(mfc)), 
                                                       mfc.numpy(), basefmt=' ')
        stemlines.set_colors(colors)
        markerline.set_markerfacecolor('none')
        markerline.set_markeredgecolor('black')
        markerline.set_markersize(8)
        
        for i, mfc_val in enumerate(mfc.numpy()):
            color = 'green' if mfc_val <= 10.0 else 'orange'
            ax_left.text(i, mfc_val, f'{mfc_val:.1f}Hz', 
                        ha='center', va='bottom', fontsize=7, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                 alpha=0.5, edgecolor='black'))
        
        ax_left.set_xlabel('Filter Index', fontsize=10)
        ax_left.set_ylabel('Center Frequency [Hz]', fontsize=10)
        ax_left.set_title(f'max_mfc={max_mfc}Hz → {len(mfc)} filters', 
                         fontsize=11)
        ax_left.set_xticks(np.arange(len(mfc)))
        ax_left.set_xticklabels([str(i) for i in range(len(mfc))])
        ax_left.grid(True, alpha=0.3)
        ax_left.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Right column: Cumulative filter coverage
        ax_right = axes[idx, 1]
        
        ax_right.plot(np.arange(len(mfc)), mfc.numpy(), 'o-', linewidth=2, markersize=8,
                     color='blue', label='mfc values')
        ax_right.axhline(y=max_mfc, color='red', linestyle='--', linewidth=2, 
                        label=f'max_mfc = {max_mfc} Hz')
        ax_right.fill_between(np.arange(len(mfc)), 0, mfc.numpy(), alpha=0.3, color='blue')
        
        ax_right.set_xlabel('Filter Index', fontsize=10)
        ax_right.set_ylabel('Modulation Frequency [Hz]', fontsize=10)
        ax_right.set_title(f'max_mfc={max_mfc}Hz: Coverage', fontsize=11)
        ax_right.set_xticks(np.arange(len(mfc)))
        ax_right.set_xticklabels([str(i) for i in range(len(mfc))])
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_maxmfc_variation_dau1997.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


def test_modulation_bandwidth_analysis():
    """Analyze bandwidth and Q-factor for Q-variation and fc-variation."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("DAU1997 Modulation Filter Bandwidth Analysis")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('DAU1997 Modulation Filter Bandwidth Analysis', 
                 fontsize=16, fontweight='bold')
    
    # =========================================================================
    # LEFT COLUMN: Q-factor variation
    # =========================================================================
    print("\n" + "="*60)
    print("Q-factor Variation (fc=1000Hz)")
    print("="*60)
    
    q_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    fc = torch.tensor([1000.0])
    colors_q = plt.cm.viridis(np.linspace(0, 1, len(q_values)))
    
    for idx, q in enumerate(q_values):
        modfb = ModulationFilterbank(fs=44100, fc=fc, Q=q, max_mfc=150.0)
        mfc = modfb.mfc[0].numpy()
        
        print(f"\nQ={q}: {len(mfc)} filters")
        
        # Estimate bandwidth from Q and mfc
        # For DAU1997: bw = 5 Hz for mfc<10, bw = mfc/Q for mfc>=10
        bandwidths = []
        for m in mfc[1:]:  # Skip mfc=0 (lowpass)
            if m < 10:
                bw = 5.0
            else:
                bw = m / q
            bandwidths.append(bw)
        
        center_freqs = mfc[1:].tolist()  # Skip mfc=0
        
        if len(center_freqs) > 0:
            # Plot 1: Bandwidth vs mfc
            axes[0, 0].plot(center_freqs, bandwidths, 'o-', linewidth=2, markersize=7,
                           color=colors_q[idx], label=f'Q={q} ({len(center_freqs)} filters)',
                           alpha=0.8)
            
            # Plot 2: Q-factor vs mfc
            q_factors = np.array(center_freqs) / np.array(bandwidths)
            axes[1, 0].plot(center_freqs, q_factors, 's-', linewidth=2, markersize=7,
                           color=colors_q[idx], label=f'Q={q}', alpha=0.8)
    
    # Configure left column
    axes[0, 0].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[0, 0].set_ylabel('Bandwidth [Hz]', fontsize=11)
    axes[0, 0].set_title('Bandwidth vs mfc (Q-factor Control)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    axes[1, 0].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[1, 0].set_ylabel('Q-factor (mfc/BW)', fontsize=11)
    axes[1, 0].set_title('Q-factor vs mfc (Q-factor Control)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    for q in q_values:
        axes[1, 0].axhline(y=q, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    axes[1, 0].legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    # =========================================================================
    # RIGHT COLUMN: fc variation
    # =========================================================================
    print("\n" + "="*60)
    print("Auditory fc Variation (Q=2.0)")
    print("="*60)
    
    fc_values = [300, 600, 1000, 3000, 8000]
    q = 2.0
    colors_fc = plt.cm.plasma(np.linspace(0, 1, len(fc_values)))
    
    for idx, fc_val in enumerate(fc_values):
        fc = torch.tensor([float(fc_val)])
        modfb = ModulationFilterbank(fs=44100, fc=fc, Q=q, max_mfc=150.0)
        mfc = modfb.mfc[0].numpy()
        
        umf = min(fc_val * 0.25, 150.0)
        print(f"\nfc={fc_val}Hz (umf={umf:.1f}Hz): {len(mfc)} filters")
        
        # Estimate bandwidth
        bandwidths = []
        for m in mfc[1:]:
            if m < 10:
                bw = 5.0
            else:
                bw = m / q
            bandwidths.append(bw)
        
        center_freqs = mfc[1:].tolist()
        
        if len(center_freqs) > 0:
            # Plot 3: Bandwidth vs mfc
            axes[0, 1].plot(center_freqs, bandwidths, 'o-', linewidth=2, markersize=7,
                           color=colors_fc[idx], 
                           label=f'fc={fc_val}Hz (umf={umf:.0f}Hz, {len(center_freqs)} filters)',
                           alpha=0.8)
            
            # Plot 4: Q-factor vs mfc
            q_factors = np.array(center_freqs) / np.array(bandwidths)
            axes[1, 1].plot(center_freqs, q_factors, 's-', linewidth=2, markersize=7,
                           color=colors_fc[idx], label=f'fc={fc_val}Hz', alpha=0.8)
    
    # Configure right column
    axes[0, 1].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[0, 1].set_ylabel('Bandwidth [Hz]', fontsize=11)
    axes[0, 1].set_title('Bandwidth vs mfc (Auditory fc Control)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=8, loc='upper left', framealpha=0.9)
    
    axes[1, 1].set_xlabel('Center Frequency [Hz]', fontsize=11)
    axes[1, 1].set_ylabel('Q-factor (mfc/BW)', fontsize=11)
    axes[1, 1].set_title('Q-factor vs mfc (Auditory fc Control)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].axhline(y=q, color='r', linestyle='--', alpha=0.5, linewidth=2,
                      label=f'Q={q} (reference)')
    axes[1, 1].legend(fontsize=8, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_bandwidth_dau1997.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


def test_hybrid_spacing_verification():
    """Verify hybrid spacing: linear (0-10 Hz) + exponential (>10 Hz)."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("DAU1997 Hybrid Spacing Verification")
    print("="*80)
    
    fc = torch.tensor([1000.0])
    q = 2.0
    
    modfb = ModulationFilterbank(fs=44100, fc=fc, Q=q, max_mfc=150.0)
    mfc = modfb.mfc[0].numpy()
    
    print(f"\nConfiguration: fc=1000Hz, Q={q}, max_mfc=150Hz")
    print(f"Number of filters: {len(mfc)}")
    print(f"mfc values: {mfc}")
    
    # Calculate expected ex for exponential region
    ex = (1 + 1/(2*q)) / (1 - 1/(2*q))
    print(f"\nExpected exponential factor (ex): {ex:.4f}")
    
    # Analyze spacing
    linear_mfc = mfc[mfc <= 10.0]
    exp_mfc = mfc[mfc > 10.0]
    
    print(f"\nLinear region (<= 10 Hz): {len(linear_mfc)} filters")
    print(f"  Values: {linear_mfc}")
    if len(linear_mfc) > 1:
        linear_steps = linear_mfc[1:] - linear_mfc[:-1]
        print(f"  Steps: {linear_steps}")
        print(f"  Mean step: {linear_steps.mean():.2f} Hz (expected: 5 Hz)")
    
    print(f"\nExponential region (> 10 Hz): {len(exp_mfc)} filters")
    print(f"  Values: {exp_mfc}")
    if len(exp_mfc) > 1:
        exp_ratios = exp_mfc[1:] / exp_mfc[:-1]
        print(f"  Ratios: {exp_ratios}")
        print(f"  Mean ratio: {exp_ratios.mean():.4f} (expected: {ex:.4f})")
        print(f"  Std ratio: {exp_ratios.std():.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DAU1997 Hybrid Spacing Verification (fc=1000Hz, Q=2.0)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: All mfc values with colored regions
    ax1 = axes[0, 0]
    colors = ['blue' if m == 0 else 'green' if m <= 10.0 else 'orange' for m in mfc]
    markerline, stemlines, baseline = ax1.stem(np.arange(len(mfc)), mfc, basefmt=' ')
    stemlines.set_colors(colors)
    markerline.set_markerfacecolor('none')
    markerline.set_markeredgecolor('black')
    markerline.set_markersize(10)
    markerline.set_markeredgewidth(2)
    
    ax1.axhspan(0, 10, alpha=0.2, color='green', label='Linear region (≤10 Hz)')
    ax1.axhspan(10, 150, alpha=0.2, color='orange', label='Exponential region (>10 Hz)')
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10 Hz transition')
    
    # Add frequency labels to lollipops
    for i, mfc_val in enumerate(mfc):
        color = 'blue' if mfc_val == 0 else 'green' if mfc_val <= 10.0 else 'orange'
        ax1.text(i, mfc_val, f'{mfc_val:.1f}Hz', 
                ha='center', va='bottom', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                         alpha=0.5, edgecolor='black'))
    
    ax1.set_xlabel('Filter Index', fontsize=11)
    ax1.set_ylabel('Center Frequency [Hz]', fontsize=11)
    ax1.set_title('Modulation Filter Center Frequencies', fontsize=12)
    ax1.set_xticks(np.arange(len(mfc)))
    ax1.set_xticklabels([str(i) for i in range(len(mfc))])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')
    
    # Plot 2: Linear region detail
    ax2 = axes[0, 1]
    if len(linear_mfc) > 0:
        ax2.plot(np.arange(len(linear_mfc)), linear_mfc, 'go-', linewidth=2, 
                markersize=10, label='Linear spacing')
        for i, val in enumerate(linear_mfc):
            # Offset text slightly higher for better visibility
            y_offset = 0.5  # Hz offset
            ax2.text(i, val + y_offset, f'{val:.1f} Hz', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Filter Index (Linear Region)', fontsize=11)
    ax2.set_ylabel('Center Frequency [Hz]', fontsize=11)
    ax2.set_title('Linear Region Detail (step = 5 Hz)', fontsize=12)
    ax2.set_xticks(np.arange(len(linear_mfc)))
    ax2.set_xticklabels([str(i) for i in range(len(linear_mfc))])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Frequency steps
    ax3 = axes[1, 0]
    if len(mfc) > 1:
        steps = mfc[1:] - mfc[:-1]
        x_indices = np.arange(1, len(mfc))
        colors_steps = ['green' if mfc[i] <= 10.0 else 'orange' for i in range(1, len(mfc))]
        ax3.bar(x_indices, steps, color=colors_steps, alpha=0.7, edgecolor='black')
        ax3.axhline(y=5, color='green', linestyle='--', linewidth=2, label='5 Hz (linear step)')
        ax3.set_xlabel('Filter Index', fontsize=11)
        ax3.set_ylabel('Frequency Step [Hz]', fontsize=11)
        ax3.set_title('Frequency Steps Between Adjacent Filters', fontsize=12)
        ax3.set_xticks(x_indices)
        ax3.set_xticklabels([str(int(i)) for i in x_indices])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(fontsize=9)
    
    # Plot 4: Frequency ratios (exponential region)
    ax4 = axes[1, 1]
    if len(mfc) > 1:
        ratios = mfc[1:] / mfc[:-1]
        x_indices = np.arange(1, len(mfc))
        colors_ratios = ['green' if mfc[i] <= 10.0 else 'orange' for i in range(1, len(mfc))]
        ax4.plot(x_indices, ratios, 'o-', linewidth=2, markersize=8)
        for i, (x, r, c) in enumerate(zip(x_indices, ratios, colors_ratios)):
            ax4.plot(x, r, 'o', markersize=8, color=c)
        
        ax4.axhline(y=ex, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected ratio ex={ex:.3f}')
        ax4.set_xlabel('Filter Index', fontsize=11)
        ax4.set_ylabel('Frequency Ratio (mfc[i]/mfc[i-1])', fontsize=11)
        ax4.set_title('Frequency Ratios (should be constant in exponential region)', 
                     fontsize=12)
        ax4.set_xticks(x_indices)
        ax4.set_xticklabels([str(int(i)) for i in x_indices])
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'modulation_filterbank_spacing_dau1997.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DAU1997 MODULATION FILTERBANK DESIGN VERIFICATION")
    print("="*80)
    print("\nKey features:")
    print("  - Hybrid spacing: Linear (0-10 Hz, step=5Hz) + Exponential (>10 Hz)")
    print("  - fc-dependent: umf = min(0.25*fc, 150 Hz)")
    print("  - Includes mfc=0: Lowpass filter at 2.5 Hz")
    print("  - Default Q = 2.0 (vs 1.0 for King2019)")
    print("="*80)
    
    test_modulation_filterbank_qfactor()
    test_modulation_filterbank_umf_variation()
    test_modulation_filterbank_maxmfc()
    test_modulation_bandwidth_analysis()
    test_hybrid_spacing_verification()
    
    print("\n\n" + "="*80)
    print("ALL DESIGN VERIFICATION TESTS COMPLETED!")
    print("Generated plots:")
    print("  1. modulation_filterbank_qfactor_dau1997.png - Q-factor control (5 configs)")
    print("  2. modulation_filterbank_umf_variation_dau1997.png - umf control (5 configs)")
    print("  3. modulation_filterbank_maxmfc_variation_dau1997.png - max_mfc control (4 configs)")
    print("  4. modulation_filterbank_bandwidth_dau1997.png - Bandwidth analysis")
    print("  5. modulation_filterbank_spacing_dau1997.png - Hybrid spacing verification")
    print("="*80)
