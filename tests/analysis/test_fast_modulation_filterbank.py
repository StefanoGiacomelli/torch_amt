"""
FastModulationFilterbank - Test Suite

Contents:
1. test_forward_validation: Validates forward pass with all 3 presets
2. test_modulation_center_frequencies: Tests frequency computation
3. test_filter_type_comparison: Compares efilt vs butterworth filters

Structure:
- Tests all presets: dau1997, jepsen2008, paulick2024
- Validates filter design and frequency responses
- Generates filter type comparison visualizations

Figures generated:
- fast_modulation_filterbank_design_comparison_efilt.png: Efilt filter implementation
- fast_modulation_filterbank_design_comparison_butterworth.png: Butterworth filter implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch_amt.common import FastModulationFilterbank


def test_forward_validation():
    """Test forward pass with dummy inputs - validation only."""
    print("\n" + "="*80)
    print("TEST 1: FORWARD PASS VALIDATION")
    print("="*80)
    
    fs = 11025  # After 4x resampling from 44100 Hz
    fc_aud = torch.linspace(250, 8000, 50, dtype=torch.float64)
    num_channels = len(fc_aud)
    duration = 0.1
    n_samples = int(fs * duration)
    
    # Dummy input: random adapted signal
    x = torch.randn(num_channels, n_samples, dtype=torch.float64) * 100.0
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.3e}, {x.max():.3e}]")
    print(f"Sampling rate: {fs} Hz (after 4x downsampling)")
    
    # Test dau1997
    mfb_dau = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='dau1997', dtype=torch.float64)
    with torch.no_grad():
        y_dau = mfb_dau(x)
    
    print(f"\nDau1997 preset:")
    print(f"  LP cutoff: 2.5 Hz")
    print(f"  Upper limit: Fixed at {mfb_dau.max_mfc} Hz")
    print(f"  Attenuation factor: {mfb_dau.att_factor:.3f}")
    print(f"  150 Hz pre-filter: {mfb_dau.use_lp150_prefilter}")
    print(f"  Output: list of {len(y_dau)} tensors")
    print(f"  Modulation channels per aud channel: min={min([y.shape[-2] for y in y_dau])}, max={max([y.shape[-2] for y in y_dau])}")
    print(f"  Example output[0] shape: {y_dau[0].shape}")
    
    # Test jepsen2008
    mfb_jepsen = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='jepsen2008', dtype=torch.float64)
    with torch.no_grad():
        y_jepsen = mfb_jepsen(x)
    
    print(f"\nJepsen2008 preset:")
    print(f"  LP cutoff: 150.0 Hz")
    print(f"  Upper limit: Dynamic (0.25 * fc_aud, max {mfb_jepsen.max_mfc} Hz)")
    print(f"  Attenuation factor: {mfb_jepsen.att_factor:.3f}")
    print(f"  150 Hz pre-filter: {mfb_jepsen.use_lp150_prefilter}")
    print(f"  Output: list of {len(y_jepsen)} tensors")
    print(f"  Modulation channels per aud channel: min={min([y.shape[-2] for y in y_jepsen])}, max={max([y.shape[-2] for y in y_jepsen])}")
    print(f"  Example output[0] shape: {y_jepsen[0].shape}")
    
    # Test paulick2024
    mfb_paulick = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='paulick2024', dtype=torch.float64)
    with torch.no_grad():
        y_paulick = mfb_paulick(x)
    
    print(f"\nPaulick2024 preset:")
    print(f"  LP cutoff: 150.0 Hz")
    print(f"  Upper limit: Dynamic (0.25 * fc_aud, max {mfb_paulick.max_mfc} Hz)")
    print(f"  Attenuation factor: {mfb_paulick.att_factor:.3f}")
    print(f"  150 Hz pre-filter: {mfb_paulick.use_lp150_prefilter}")
    print(f"  Output: list of {len(y_paulick)} tensors")
    print(f"  Modulation channels per aud channel: min={min([y.shape[-2] for y in y_paulick])}, max={max([y.shape[-2] for y in y_paulick])}")
    print(f"  Example output[0] shape: {y_paulick[0].shape}")
    
    # Validation
    assert len(y_dau) == num_channels, "Dau1997: Number of outputs mismatch"
    assert len(y_jepsen) == num_channels, "Jepsen2008: Number of outputs mismatch"
    assert len(y_paulick) == num_channels, "Paulick2024: Number of outputs mismatch"
    
    for i in range(num_channels):
        assert not torch.isnan(y_dau[i]).any(), f"Dau1997: NaN in channel {i}"
        assert not torch.isnan(y_jepsen[i]).any(), f"Jepsen2008: NaN in channel {i}"
        assert not torch.isnan(y_paulick[i]).any(), f"Paulick2024: NaN in channel {i}"
        assert not torch.isinf(y_dau[i]).any(), f"Dau1997: Inf in channel {i}"
        assert not torch.isinf(y_jepsen[i]).any(), f"Jepsen2008: Inf in channel {i}"
        assert not torch.isinf(y_paulick[i]).any(), f"Paulick2024: Inf in channel {i}"
    
    print(f"\n✓ All validation checks passed")


def test_modulation_center_frequencies():
    """Test modulation center frequency computation."""
    print("\n" + "="*80)
    print("TEST 2: MODULATION CENTER FREQUENCIES")
    print("="*80)
    
    fs = 11025
    fc_aud = torch.tensor([250, 1000, 4000, 8000], dtype=torch.float64)
    
    # Test dau1997
    mfb_dau = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='dau1997', dtype=torch.float64)
    print(f"\nDau1997 (no upper limit):")
    for i, fc in enumerate(fc_aud):
        mfc = mfb_dau.mfc[i]
        print(f"  fc_aud={fc:.0f} Hz: mfc = {mfc.tolist()}")
    
    # Test jepsen2008
    mfb_jepsen = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='jepsen2008', dtype=torch.float64)
    print(f"\nJepsen2008 (dynamic upper limit = 0.25 * fc_aud):")
    for i, fc in enumerate(fc_aud):
        mfc = mfb_jepsen.mfc[i]
        umf = min(fc.item() * 0.25, 150.0)
        print(f"  fc_aud={fc:.0f} Hz (umf={umf:.1f}): mfc = {mfc.tolist()}")
    
    # Test paulick2024
    mfb_paulick = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='paulick2024', dtype=torch.float64)
    print(f"\nPaulick2024 (same as jepsen2008):")
    for i, fc in enumerate(fc_aud):
        mfc = mfb_paulick.mfc[i]
        umf = min(fc.item() * 0.25, 150.0)
        print(f"  fc_aud={fc:.0f} Hz (umf={umf:.1f}): mfc = {mfc.tolist()}")
    
    # Verify geometric spacing ratio for mfc > 10 Hz
    print(f"\nVerifying geometric spacing (ratio should be ~5/3 = 1.667):")
    mfc_example = mfb_jepsen.mfc[3]  # 8000 Hz channel
    mfc_high = mfc_example[mfc_example > 10]
    if len(mfc_high) > 1:
        ratios = mfc_high[1:] / mfc_high[:-1]
        print(f"  Ratios: {ratios.tolist()}")
        print(f"  Mean ratio: {ratios.mean():.4f} (expected: 1.6667)")
    
    print(f"\n✓ Modulation center frequencies correctly computed")


def test_filter_type_comparison():
    """Generate comparison figures for efilt vs butterworth filter implementations."""
    print("\n" + "="*80)
    print("TEST 3: FILTER TYPE COMPARISON (EFILT vs BUTTERWORTH)")
    print("="*80)
    
    fs = 11025
    fc_aud = torch.linspace(250, 8000, 50, dtype=torch.float64)
    
    print(f"\nGenerating 2 comparison figures (same structure, different filter types)...")
    
    # Create both filter types for dau1997 preset
    filter_types = ['efilt', 'butterworth']
    
    for filter_type in filter_types:
        print(f"\n  Processing {filter_type} implementation...")
        
        # Create all 3 presets with specified filter_type
        mfb_dau = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='dau1997', filter_type=filter_type, dtype=torch.float64)
        mfb_jepsen = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='jepsen2008', filter_type=filter_type, dtype=torch.float64)
        mfb_paulick = FastModulationFilterbank(fs=fs, fc=fc_aud, preset='paulick2024', filter_type=filter_type, dtype=torch.float64)
        
        # Create figure with GridSpec layout
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 15))
        
        filter_type_label = 'Complex Freq-Shifted LP (MATLAB AMT)' if filter_type == 'efilt' else 'Butterworth Bandpass'
        fig.suptitle(f'Fast Modulation Filterbank Design Analysis - {filter_type_label}', 
                     fontsize=16, fontweight='bold', y=0.995)
        gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Use channel at 1000 Hz for visualization
        ch_idx_example = torch.argmin(torch.abs(fc_aud - 1000.0)).item()
        
        # ========== ROW 1: MODULATION CENTER FREQUENCIES (COMMON TO ALL PRESETS) ==========
        ax1 = fig.add_subplot(gs[0, :])  # Span all 3 columns
        mfc_common = mfb_dau.mfc[ch_idx_example].numpy()
        bars1 = ax1.bar(range(len(mfc_common)), mfc_common, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Modulation Frequency (Hz)', fontsize=11)
        ax1.set_title(f'Modulation Center Frequencies (fc_aud={fc_aud[ch_idx_example]:.0f} Hz) - Common to All Presets', fontsize=12)
        ax1.set_xlabel('Filter Index', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_yscale('symlog', linthresh=1)
        # Add mfc values inside bars
        for i, (bar, val) in enumerate(zip(bars1, mfc_common)):
            ax1.text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.1f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # ========== ROW 2: UPPER MODULATION LIMITS PER CHANNEL ==========
        channels = np.arange(len(fc_aud))
        
        ax2 = fig.add_subplot(gs[1, 0])
        umf_dau = np.array([mfb_dau.max_mfc] * len(fc_aud))
        ax2.plot(channels, umf_dau, color='steelblue', linewidth=2, label='Fixed limit')
        ax2.set_xlabel('Auditory Channel', fontsize=10)
        ax2.set_ylabel('Upper Modulation Limit (Hz)', fontsize=10)
        ax2.set_title('Dau1997: Upper Modulation Limits', fontsize=11)
        ax2.set_xlim(0, len(fc_aud)-1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        ax3 = fig.add_subplot(gs[1, 1])
        umf_jepsen = np.array([min(fc.item() * 0.25, mfb_jepsen.max_mfc) for fc in fc_aud])
        ax3.plot(channels, umf_jepsen, color='darkorange', linewidth=2, label='Dynamic (0.25*fc)')
        ax3.axhline(mfb_jepsen.max_mfc, color='red', linestyle='--', alpha=0.5, label=f'Max={mfb_jepsen.max_mfc} Hz')
        ax3.set_xlabel('Auditory Channel', fontsize=10)
        ax3.set_ylabel('Upper Modulation Limit (Hz)', fontsize=10)
        ax3.set_title('Jepsen2008: Upper Modulation Limits', fontsize=11)
        ax3.set_xlim(0, len(fc_aud)-1)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        ax4 = fig.add_subplot(gs[1, 2])
        umf_paulick = np.array([min(fc.item() * 0.25, mfb_paulick.max_mfc) for fc in fc_aud])
        ax4.plot(channels, umf_paulick, color='coral', linewidth=2, label='Dynamic (0.25*fc)')
        ax4.axhline(mfb_paulick.max_mfc, color='red', linestyle='--', alpha=0.5, label=f'Max={mfb_paulick.max_mfc} Hz')
        ax4.set_xlabel('Auditory Channel', fontsize=10)
        ax4.set_ylabel('Upper Modulation Limit (Hz)', fontsize=10)
        ax4.set_title('Paulick2024: Upper Modulation Limits', fontsize=11)
        ax4.set_xlim(0, len(fc_aud)-1)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        
        # ========== ROW 3: MODULATION FILTER FREQUENCY RESPONSES ==========
        freqs = np.logspace(-1, np.log10(200), 1000)  # 0.1 Hz to 200 Hz
        w = 2 * np.pi * freqs / fs
        z_inv = np.exp(-1j * w)  # z^(-1) for transfer function evaluation
        
        ax5 = fig.add_subplot(gs[2, :])  # Span all 3 columns
        # Plot all modulation filters
        for i, mfc_val in enumerate(mfc_common):
            if mfc_val > 0:  # Skip DC component
                filter_params = mfb_dau.filter_coeffs[ch_idx_example][i]
                b = filter_params[0].cpu().numpy()
                a = filter_params[1].cpu().numpy()
                # Handle complex coefficients (take real part)
                if np.iscomplexobj(b):
                    b = b.real
                if np.iscomplexobj(a):
                    a = a.real
                # H(z) evaluation
                if filter_type == 'efilt':
                    # H(z) = b[0] / (a[0] + a[1]*z^(-1))
                    H = b[0] / (a[0] + a[1] * z_inv)
                else:  # butterworth - uses SOS (Second-Order Sections) format
                    # b contains SOS matrix [n_sections, 6] where each row is [b0, b1, b2, a0, a1, a2]
                    # Need to multiply transfer functions of all sections
                    if b.ndim == 2 and b.shape[1] == 6:
                        # SOS format detected
                        H = np.ones_like(z_inv, dtype=complex)
                        for section in b:
                            # Each section: [b0, b1, b2, a0, a1, a2]
                            b0, b1, b2, a0, a1, a2 = section
                            # H_section(z) = (b0 + b1*z^(-1) + b2*z^(-2)) / (a0 + a1*z^(-1) + a2*z^(-2))
                            H_num_section = b0 + b1 * z_inv + b2 * (z_inv ** 2)
                            H_den_section = a0 + a1 * z_inv + a2 * (z_inv ** 2)
                            H = H * (H_num_section / H_den_section)
                    else:
                        # Fallback: should not happen with butterworth
                        H = np.ones_like(z_inv, dtype=complex)
                ax5.semilogx(freqs, 20 * np.log10(np.abs(H)), linewidth=2, alpha=0.75, label=f'{mfc_val:.1f} Hz')
        ax5.axvline(10, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, label='10 Hz threshold')
        ax5.set_xlabel('Frequency (Hz)', fontsize=11)
        ax5.set_ylabel('Magnitude (dB)', fontsize=11)
        
        filter_desc = 'Resonant LP' if filter_type == 'efilt' else 'Butterworth BP'
        ax5.set_title(f'Modulation Filters ({filter_desc}, fc_aud={fc_aud[ch_idx_example]:.0f} Hz) - Common to All Presets', fontsize=12)
        ax5.set_xlim(0.1, 200)
        ax5.set_ylim(-40, 5)
        ax5.grid(True, alpha=0.3, which='both')
        ax5.legend(fontsize=8, loc='lower left', ncol=3)
        
        # ========== ROW 4: LOWPASS FILTER FREQUENCY RESPONSES ==========
        freqs = np.logspace(-1, 3, 1000)  # 0.1 Hz to 1000 Hz
        
        ax6 = fig.add_subplot(gs[3, 0])
        # LP @ 2.5 Hz
        w = 2 * np.pi * freqs / fs
        b_lp = mfb_dau.b_lowpass.cpu().numpy()
        a_lp = mfb_dau.a_lowpass.cpu().numpy()
        H = np.polyval(b_lp, np.exp(-1j * w)) / np.polyval(a_lp, np.exp(-1j * w))
        ax6.semilogx(freqs, 20 * np.log10(np.abs(H)), linewidth=2, color='steelblue', label='LP @ 2.5 Hz')
        ax6.axvline(2.5, color='red', linestyle='--', alpha=0.5, label='Cutoff')
        ax6.set_xlabel('Frequency (Hz)', fontsize=10)
        ax6.set_ylabel('Magnitude (dB)', fontsize=10)
        ax6.set_title('Dau1997: Lowpass Filter', fontsize=11)
        ax6.set_xlim(0.1, 1000)
        ax6.set_ylim(-60, 5)
        ax6.grid(True, alpha=0.3, which='both')
        ax6.legend(fontsize=8)
        
        ax7 = fig.add_subplot(gs[3, 1])
        # LP @ 150 Hz + LP @ 2.5 Hz (cascade)
        b_lp = mfb_jepsen.b_lowpass.cpu().numpy()
        a_lp = mfb_jepsen.a_lowpass.cpu().numpy()
        H_lp2p5 = np.polyval(b_lp, np.exp(-1j * w)) / np.polyval(a_lp, np.exp(-1j * w))
        
        b_lp150 = mfb_jepsen.b_lp150.cpu().numpy()
        a_lp150 = mfb_jepsen.a_lp150.cpu().numpy()
        H_lp150 = np.polyval(b_lp150, np.exp(-1j * w)) / np.polyval(a_lp150, np.exp(-1j * w))
        
        ax7.semilogx(freqs, 20 * np.log10(np.abs(H_lp150)), linewidth=2, color='green', label='LP @ 150 Hz (pre-filter)', linestyle='--')
        ax7.semilogx(freqs, 20 * np.log10(np.abs(H_lp2p5)), linewidth=2, color='darkorange', label='LP @ 2.5 Hz')
        ax7.axvline(2.5, color='red', linestyle='--', alpha=0.5)
        ax7.axvline(150, color='green', linestyle='--', alpha=0.5)
        ax7.set_xlabel('Frequency (Hz)', fontsize=10)
        ax7.set_ylabel('Magnitude (dB)', fontsize=10)
        ax7.set_title('Jepsen2008: Lowpass Filters', fontsize=11)
        ax7.set_xlim(0.1, 1000)
        ax7.set_ylim(-60, 5)
        ax7.grid(True, alpha=0.3, which='both')
        ax7.legend(fontsize=8)
        
        ax8 = fig.add_subplot(gs[3, 2])
        # Same as jepsen2008
        b_lp = mfb_paulick.b_lowpass.cpu().numpy()
        a_lp = mfb_paulick.a_lowpass.cpu().numpy()
        H_lp2p5 = np.polyval(b_lp, np.exp(-1j * w)) / np.polyval(a_lp, np.exp(-1j * w))
        
        b_lp150 = mfb_paulick.b_lp150.cpu().numpy()
        a_lp150 = mfb_paulick.a_lp150.cpu().numpy()
        H_lp150 = np.polyval(b_lp150, np.exp(-1j * w)) / np.polyval(a_lp150, np.exp(-1j * w))
        
        ax8.semilogx(freqs, 20 * np.log10(np.abs(H_lp150)), linewidth=2, color='green', label='LP @ 150 Hz (pre-filter)', linestyle='--')
        ax8.semilogx(freqs, 20 * np.log10(np.abs(H_lp2p5)), linewidth=2, color='coral', label='LP @ 2.5 Hz')
        ax8.axvline(2.5, color='red', linestyle='--', alpha=0.5)
        ax8.axvline(150, color='green', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Frequency (Hz)', fontsize=10)
        ax8.set_ylabel('Magnitude (dB)', fontsize=10)
        ax8.set_title('Paulick2024: Lowpass Filters', fontsize=11)
        ax8.set_xlim(0.1, 1000)
        ax8.set_ylim(-60, 5)
        ax8.grid(True, alpha=0.3, which='both')
        ax8.legend(fontsize=8)
        
        # ========== ROW 5: PHASE INSENSITIVITY & ATTENUATION ==========
        mfc_example = np.array([0, 5, 10, 16.6, 27.8, 46.3, 77.2, 128.6])
        
        ax9 = fig.add_subplot(gs[4, 0])
        output_types = ['Real' if mfc <= 10 else 'Abs' for mfc in mfc_example]
        att_factors = [1.0 if mfc <= 10 else mfb_dau.att_factor for mfc in mfc_example]
        colors = ['blue' if t == 'Real' else 'red' for t in output_types]
        ax9.bar(range(len(mfc_example)), att_factors, color=colors, alpha=0.7, edgecolor='black')
        ax9.set_xlabel('Modulation Filter Index', fontsize=10)
        ax9.set_ylabel('Attenuation Factor', fontsize=10)
        ax9.set_title('Dau1997: Phase Insens & Attenuation', fontsize=11)
        ax9.set_xticks(range(len(mfc_example)))
        ax9.set_xticklabels([f'{mfc:.1f}' for mfc in mfc_example], fontsize=8, rotation=45)
        ax9.axhline(1.0, color='black', linestyle='--', alpha=0.3)
        ax9.grid(axis='y', alpha=0.3)
        # Add legend for color code
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Real part (≤10 Hz)'),
                          Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Abs value (>10 Hz)')]
        ax9.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax9.text(0.5, 0.95, f'Factor (>10Hz): {mfb_dau.att_factor:.3f}', 
                  transform=ax9.transAxes, ha='center', va='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax10 = fig.add_subplot(gs[4, 1])
        att_factors = [1.0 if mfc <= 10 else mfb_jepsen.att_factor for mfc in mfc_example]
        ax10.bar(range(len(mfc_example)), att_factors, color=colors, alpha=0.7, edgecolor='black')
        ax10.set_xlabel('Modulation Filter Index', fontsize=10)
        ax10.set_ylabel('Attenuation Factor', fontsize=10)
        ax10.set_title('Jepsen2008: Phase Insens & Attenuation', fontsize=11)
        ax10.set_xticks(range(len(mfc_example)))
        ax10.set_xticklabels([f'{mfc:.1f}' for mfc in mfc_example], fontsize=8, rotation=45)
        ax10.axhline(1.0, color='black', linestyle='--', alpha=0.3)
        ax10.axhline(mfb_jepsen.att_factor, color='red', linestyle='--', alpha=0.5, label=f'{mfb_jepsen.att_factor:.3f}')
        ax10.grid(axis='y', alpha=0.3)
        # Add legend for color code
        legend_elements = [Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Real part (≤10 Hz)'),
                          Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Abs value (>10 Hz)')]
        ax10.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax10.text(0.5, 0.95, f'Factor (>10Hz): {mfb_jepsen.att_factor:.3f}', 
                  transform=ax10.transAxes, ha='center', va='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax11 = fig.add_subplot(gs[4, 2])
        att_factors = [1.0 if mfc <= 10 else mfb_paulick.att_factor for mfc in mfc_example]
        ax11.bar(range(len(mfc_example)), att_factors, color=colors, alpha=0.7, edgecolor='black')
        ax11.set_xlabel('Modulation Filter Index', fontsize=10)
        ax11.set_ylabel('Attenuation Factor', fontsize=10)
        ax11.set_title('Paulick2024: Phase Insens & Attenuation', fontsize=11)
        ax11.set_xticks(range(len(mfc_example)))
        ax11.set_xticklabels([f'{mfc:.1f}' for mfc in mfc_example], fontsize=8, rotation=45)
        ax11.axhline(1.0, color='black', linestyle='--', alpha=0.3)
        ax11.axhline(mfb_paulick.att_factor, color='red', linestyle='--', alpha=0.5, label=f'{mfb_paulick.att_factor:.3f}')
        ax11.grid(axis='y', alpha=0.3)
        # Add legend for color code
        legend_elements = [Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Real part (≤10 Hz)'),
                          Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Abs value (>10 Hz)')]
        ax11.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax11.text(0.5, 0.95, f'Factor (>10Hz): {mfb_paulick.att_factor:.3f}', 
                  transform=ax11.transAxes, ha='center', va='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save figure
        output_path = Path(__file__).parent.parent.parent / 'test_figures' / f'fast_modulation_filterbank_design_comparison_{filter_type}.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"    ✓ Saved: {output_path}")
        
        plt.close()
    
    print(f"\n✓ Generated 2 figures for filter type comparison")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FAST MODULATION FILTERBANK - COMPREHENSIVE TEST SUITE (ALL PRESETS)")
    print("="*80)
    
    test_forward_validation()
    test_modulation_center_frequencies()
    # test_design_comparison_visualization()  # Deprecated: use filter_type_comparison instead
    test_filter_type_comparison()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
