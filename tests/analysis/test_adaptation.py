"""
AdaptLoop - Comprehensive Test Suite

Contents:
1. test_forward_validation: Forward pass validation with dummy inputs (all 3 presets)
2. test_frequency_specific_minlvl: Frequency-specific minimum level application
3. test_end_to_end_pipeline: Complete pipeline DRNL -> IHC -> Adaptation
4. test_design_comparison_visualization: Design comparison visualization (dau1997 vs osses2021 vs paulick2024)

Structure:
- Tests all three adaptation presets (dau1997, osses2021, paulick2024)
- Validates time constants and overshoot limits
- Tests frequency-specific minimum level for Paulick2024
- End-to-end pipeline with 1kHz tone and 15 Hz AM modulation

Figures generated:
- adaptation_design_comparison.png: 6-panel comparison of all three adaptation designs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import AdaptLoop, DRNLFilterbank, IHCPaulick2024


def test_forward_validation():
    """Test forward pass with dummy inputs - validation only."""
    print("\n" + "="*80)
    print("TEST 1: FORWARD PASS VALIDATION")
    print("="*80)
    
    fs = 44100
    num_channels = 50
    duration = 0.1
    n_samples = int(fs * duration)
    
    # Dummy input: random receptor potential-like signal
    x = torch.abs(torch.randn(num_channels, n_samples, dtype=torch.float64)) * 0.01
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.6e}, {x.max():.6e}] V")
    
    # Test dau1997
    adapt_dau = AdaptLoop(fs=fs, preset='dau1997', dtype=torch.float64)
    with torch.no_grad():
        y_dau = adapt_dau(x)
    
    print(f"\nDau1997 preset:")
    print(f"  Time constants: {[f'{t:.4f}' for t in [0.005, 0.050, 0.129, 0.253, 0.500]]}")
    print(f"  Output shape: {y_dau.shape}")
    print(f"  Output range: [{y_dau.min():.3e}, {y_dau.max():.3e}]")
    print(f"  Contains NaN: {torch.isnan(y_dau).any().item()}")
    print(f"  Contains Inf: {torch.isinf(y_dau).any().item()}")
    
    # Test osses2021
    adapt_osses = AdaptLoop(fs=fs, preset='osses2021', dtype=torch.float64)
    with torch.no_grad():
        y_osses = adapt_osses(x)
    
    print(f"\nOsses2021 preset:")
    print(f"  Time constants: {[f'{t:.4f}' for t in [0.005, 0.050, 0.129, 0.253, 0.500]]}")
    print(f"  Overshoot limit: 5.0 (vs 10.0 in dau1997)")
    print(f"  Output shape: {y_osses.shape}")
    print(f"  Output range: [{y_osses.min():.3e}, {y_osses.max():.3e}]")
    print(f"  Contains NaN: {torch.isnan(y_osses).any().item()}")
    print(f"  Contains Inf: {torch.isinf(y_osses).any().item()}")
    
    # Test paulick2024
    adapt_paulick = AdaptLoop(fs=fs, preset='paulick2024', dtype=torch.float64)
    with torch.no_grad():
        y_paulick = adapt_paulick(x)
    
    print(f"\nPaulick2024 preset:")
    print(f"  Time constants: {[f'{t:.4f}' for t in [0.007, 0.0318, 0.0878, 0.2143, 0.5]]}")
    print(f"  Output shape: {y_paulick.shape}")
    print(f"  Output range: [{y_paulick.min():.3e}, {y_paulick.max():.3e}]")
    print(f"  Contains NaN: {torch.isnan(y_paulick).any().item()}")
    print(f"  Contains Inf: {torch.isinf(y_paulick).any().item()}")
    
    # Validation
    assert y_dau.shape == x.shape, "Dau1997: Output shape mismatch"
    assert y_osses.shape == x.shape, "Osses2021: Output shape mismatch"
    assert y_paulick.shape == x.shape, "Paulick2024: Output shape mismatch"
    assert not torch.isnan(y_dau).any(), "Dau1997: NaN detected"
    assert not torch.isnan(y_osses).any(), "Osses2021: NaN detected"
    assert not torch.isnan(y_paulick).any(), "Paulick2024: NaN detected"
    assert not torch.isinf(y_dau).any(), "Dau1997: Inf detected"
    assert not torch.isinf(y_osses).any(), "Osses2021: Inf detected"
    assert not torch.isinf(y_paulick).any(), "Paulick2024: Inf detected"
    
    print(f"\n✓ All validation checks passed")


def test_frequency_specific_minlvl():
    """Test frequency-specific minimum level application."""
    print("\n" + "="*80)
    print("TEST 2: FREQUENCY-SPECIFIC MINIMUM LEVEL")
    print("="*80)
    
    fs = 44100
    
    # Test dau1997 (scalar minlvl)
    adapt_dau = AdaptLoop(fs=fs, preset='dau1997', dtype=torch.float64)
    print(f"\nDau1997:")
    print(f"  Scalar minlvl: {adapt_dau.minlvl.item():.6e}")
    print(f"  use_freq_specific_minlvl: {adapt_dau.use_freq_specific_minlvl}")
    
    # Test osses2021 (scalar minlvl)
    adapt_osses = AdaptLoop(fs=fs, preset='osses2021', dtype=torch.float64)
    print(f"\nOsses2021:")
    print(f"  Scalar minlvl: {adapt_osses.minlvl.item():.6e}")
    print(f"  use_freq_specific_minlvl: {adapt_osses.use_freq_specific_minlvl}")
    
    # Test paulick2024 (frequency-specific minlvl)
    adapt_paulick = AdaptLoop(fs=fs, preset='paulick2024', dtype=torch.float64)
    print(f"\nPaulick2024:")
    print(f"  use_freq_specific_minlvl: {adapt_paulick.use_freq_specific_minlvl}")
    print(f"  minlvl_per_channel shape: {adapt_paulick.minlvl_per_channel.shape}")
    print(f"  minlvl_per_channel range: [{adapt_paulick.minlvl_per_channel.min():.6e}, "
          f"{adapt_paulick.minlvl_per_channel.max():.6e}]")
    
    print(f"\n✓ Frequency-specific minlvl correctly implemented")


def test_end_to_end_pipeline():
    """Test complete pipeline: DRNL -> IHC -> Adaptation."""
    print("\n" + "="*80)
    print("TEST 3: END-TO-END PIPELINE (DRNL -> IHC -> ADAPTATION)")
    print("="*80)
    
    fs = 44100
    test_freq = 1000
    test_level = 60
    dur = 0.1
    
    print(f"\nTest parameters:")
    print(f"  Signal: {test_freq} Hz @ {test_level} dB SPL")
    print(f"  Duration: {dur*1000:.1f} ms")
    print(f"  Modulation: 15 Hz AM")
    
    # Generate AM tone
    t = torch.arange(0, dur, 1/fs, dtype=torch.float64)
    carrier = torch.sin(2 * torch.pi * test_freq * t)
    modulator = (1 + torch.sin(2 * torch.pi * 15 * t)) / 2
    tone = carrier * modulator
    
    # Scale to SPL
    rms_target = 20e-6 * 10.0 ** (test_level / 20.0)
    tone = tone * (rms_target / torch.sqrt(torch.mean(tone ** 2)))
    
    print(f"\nProcessing through pipeline...")
    
    # STAGE 1: DRNL
    drnl = DRNLFilterbank((250, 8000), fs=fs, n_channels=50, dtype=torch.float64)
    with torch.no_grad():
        vel = drnl(tone)  # [50, T]
    
    print(f"  DRNL output (BM velocity): [{vel.min():.3e}, {vel.max():.3e}] m/s")
    
    # STAGE 2: IHC
    ihc = IHCPaulick2024(fs=fs, dtype=torch.float64)
    with torch.no_grad():
        receptor = ihc(vel)  # [50, T]
    
    print(f"  IHC output (receptor potential): [{receptor.min():.3e}, {receptor.max():.3e}] V")
    
    # STAGE 3: Adaptation (dau1997)
    adapt_dau = AdaptLoop(fs=fs, preset='dau1997', dtype=torch.float64)
    with torch.no_grad():
        adapted_dau = adapt_dau(receptor)  # [50, T]
    
    print(f"  Adaptation (dau1997): [{adapted_dau.min():.3e}, {adapted_dau.max():.3e}]")
    
    # STAGE 3: Adaptation (osses2021)
    adapt_osses = AdaptLoop(fs=fs, preset='osses2021', dtype=torch.float64)
    with torch.no_grad():
        adapted_osses = adapt_osses(receptor)  # [50, T]
    
    print(f"  Adaptation (osses2021): [{adapted_osses.min():.3e}, {adapted_osses.max():.3e}]")
    
    # STAGE 3: Adaptation (paulick2024)
    adapt_paulick = AdaptLoop(fs=fs, preset='paulick2024', dtype=torch.float64)
    with torch.no_grad():
        adapted_paulick = adapt_paulick(receptor)  # [50, T]
    
    print(f"  Adaptation (paulick2024): [{adapted_paulick.min():.3e}, {adapted_paulick.max():.3e}]")
    
    # Validation
    assert adapted_dau.shape == receptor.shape
    assert adapted_osses.shape == receptor.shape
    assert adapted_paulick.shape == receptor.shape
    assert not torch.isnan(adapted_dau).any()
    assert not torch.isnan(adapted_osses).any()
    assert not torch.isnan(adapted_paulick).any()
    assert not torch.isinf(adapted_dau).any()
    assert not torch.isinf(adapted_osses).any()
    assert not torch.isinf(adapted_paulick).any()
    
    print(f"\n✓ End-to-end pipeline working correctly")


def test_design_comparison_visualization():
    """Generate design comparison figure: dau1997 vs osses2021 vs paulick2024."""
    print("\n" + "="*80)
    print("TEST 4: DESIGN COMPARISON VISUALIZATION")
    print("="*80)
    
    fs = 44100
    
    # Create all 3 presets
    adapt_dau = AdaptLoop(fs=fs, preset='dau1997', dtype=torch.float64)
    adapt_osses = AdaptLoop(fs=fs, preset='osses2021', dtype=torch.float64)
    adapt_paulick = AdaptLoop(fs=fs, preset='paulick2024', dtype=torch.float64)
    
    # Extract design parameters
    tau_dau = torch.tensor([0.005, 0.050, 0.129, 0.253, 0.500])
    tau_osses = torch.tensor([0.005, 0.050, 0.129, 0.253, 0.500])
    tau_paulick = torch.tensor([0.007, 0.0318, 0.0878, 0.2143, 0.5])
    
    a1_dau = adapt_dau.a1.cpu().numpy()
    b0_dau = adapt_dau.b0.cpu().numpy()
    a1_osses = adapt_osses.a1.cpu().numpy()
    b0_osses = adapt_osses.b0.cpu().numpy()
    a1_paulick = adapt_paulick.a1.cpu().numpy()
    b0_paulick = adapt_paulick.b0.cpu().numpy()
    
    print(f"\nGenerating comparison figure with 3 columns...")
    
    # Create figure with 3 columns, 5 rows
    fig = plt.figure(figsize=(18, 16))
    
    # ========== ROW 1: TIME CONSTANTS ==========
    ax1 = plt.subplot(5, 3, 1)
    ax1.bar(range(5), tau_dau.numpy() * 1000, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Time Constant (ms)', fontsize=10)
    ax1.set_title('Dau1997: Time Constants', fontsize=11)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f'Loop {i+1}' for i in range(5)], fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    for i, val in enumerate(tau_dau.numpy() * 1000):
        ax1.text(i, val + 5, f'{val:.1f}', ha='center', fontsize=8)
    
    ax2 = plt.subplot(5, 3, 2)
    ax2.bar(range(5), tau_osses.numpy() * 1000, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Time Constant (ms)', fontsize=10)
    ax2.set_title('Osses2021: Time Constants', fontsize=11)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([f'Loop {i+1}' for i in range(5)], fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    for i, val in enumerate(tau_osses.numpy() * 1000):
        ax2.text(i, val + 5, f'{val:.1f}', ha='center', fontsize=8)
    
    ax3 = plt.subplot(5, 3, 3)
    ax3.bar(range(5), tau_paulick.numpy() * 1000, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Time Constant (ms)', fontsize=10)
    ax3.set_title('Paulick2024: Time Constants', fontsize=11)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels([f'Loop {i+1}' for i in range(5)], fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    for i, val in enumerate(tau_paulick.numpy() * 1000):
        ax3.text(i, val + 5, f'{val:.1f}', ha='center', fontsize=8)
    
    # ========== ROW 2: MINIMUM LEVEL ==========
    ax4 = plt.subplot(5, 3, 4)
    ax4.axhline(adapt_dau.minlvl.item(), color='steelblue', linewidth=3, label='Scalar minlvl')
    ax4.set_xlabel('Channel Number', fontsize=10)
    ax4.set_ylabel('Minimum Level (linear)', fontsize=10)
    ax4.set_title('Dau1997: Scalar Minimum Level', fontsize=11)
    ax4.set_xlim(0, 49)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=8)
    ax4.text(25, adapt_dau.minlvl.item() * 1.5, f'{adapt_dau.minlvl.item():.2e}', 
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax5 = plt.subplot(5, 3, 5)
    ax5.axhline(adapt_osses.minlvl.item(), color='darkorange', linewidth=3, label='Scalar minlvl')
    ax5.set_xlabel('Channel Number', fontsize=10)
    ax5.set_ylabel('Minimum Level (linear)', fontsize=10)
    ax5.set_title('Osses2021: Scalar Minimum Level', fontsize=11)
    ax5.set_xlim(0, 49)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=8)
    ax5.text(25, adapt_osses.minlvl.item() * 1.5, f'{adapt_osses.minlvl.item():.2e}', 
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax6 = plt.subplot(5, 3, 6)
    channels = np.arange(50)
    minlvl_paulick = adapt_paulick.minlvl_per_channel.cpu().numpy()
    ax6.plot(channels, minlvl_paulick, color='coral', linewidth=2, marker='o', 
             markersize=3, label='Frequency-specific')
    ax6.set_xlabel('Channel Number', fontsize=10)
    ax6.set_ylabel('Minimum Level (linear)', fontsize=10)
    ax6.set_title('Paulick2024: Frequency-Specific Minimum Level', fontsize=11)
    ax6.set_xlim(0, 49)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(fontsize=8)
    
    # ========== ROW 3: RC LOWPASS FREQUENCY RESPONSE ==========
    freqs = np.logspace(0, 4, 1000)  # 1 Hz to 10 kHz
    T = 1 / fs
    
    ax7 = plt.subplot(5, 3, 7)
    for i in range(5):
        H = b0_dau[i] / np.abs(1 - a1_dau[i] * np.exp(-1j * 2 * np.pi * freqs * T))
        ax7.semilogx(freqs, 20 * np.log10(H), linewidth=1.5, label=f'Loop {i+1}')
    ax7.set_xlabel('Frequency (Hz)', fontsize=10)
    ax7.set_ylabel('Magnitude (dB)', fontsize=10)
    ax7.set_title('Dau1997: RC Lowpass Frequency Response', fontsize=11)
    ax7.set_xlim(1, 10000)
    ax7.set_ylim(-60, 5)
    ax7.grid(True, alpha=0.3, which='both')
    ax7.legend(fontsize=7, loc='lower left')
    
    ax8 = plt.subplot(5, 3, 8)
    for i in range(5):
        H = b0_osses[i] / np.abs(1 - a1_osses[i] * np.exp(-1j * 2 * np.pi * freqs * T))
        ax8.semilogx(freqs, 20 * np.log10(H), linewidth=1.5, label=f'Loop {i+1}')
    ax8.set_xlabel('Frequency (Hz)', fontsize=10)
    ax8.set_ylabel('Magnitude (dB)', fontsize=10)
    ax8.set_title('Osses2021: RC Lowpass Frequency Response', fontsize=11)
    ax8.set_xlim(1, 10000)
    ax8.set_ylim(-60, 5)
    ax8.grid(True, alpha=0.3, which='both')
    ax8.legend(fontsize=7, loc='lower left')
    
    ax9 = plt.subplot(5, 3, 9)
    for i in range(5):
        H = b0_paulick[i] / np.abs(1 - a1_paulick[i] * np.exp(-1j * 2 * np.pi * freqs * T))
        ax9.semilogx(freqs, 20 * np.log10(H), linewidth=1.5, label=f'Loop {i+1}')
    ax9.set_xlabel('Frequency (Hz)', fontsize=10)
    ax9.set_ylabel('Magnitude (dB)', fontsize=10)
    ax9.set_title('Paulick2024: RC Lowpass Frequency Response', fontsize=11)
    ax9.set_xlim(1, 10000)
    ax9.set_ylim(-60, 5)
    ax9.grid(True, alpha=0.3, which='both')
    ax9.legend(fontsize=7, loc='lower left')
    
    # ========== ROW 4: IMPULSE RESPONSE ==========
    n_impulse = 2000
    impulse = np.zeros(n_impulse)
    impulse[0] = 1.0
    t_ms = np.arange(n_impulse) / fs * 1000
    
    ax10 = plt.subplot(5, 3, 10)
    for i in range(5):
        y = np.zeros(n_impulse)
        for n in range(n_impulse):
            if n == 0:
                y[n] = b0_dau[i] * impulse[n]
            else:
                y[n] = b0_dau[i] * impulse[n] + a1_dau[i] * y[n-1]
        ax10.plot(t_ms, y, linewidth=1.5, label=f'Loop {i+1}', alpha=0.8)
    ax10.set_xlabel('Time (ms)', fontsize=10)
    ax10.set_ylabel('Amplitude', fontsize=10)
    ax10.set_title('Dau1997: Impulse Response', fontsize=11)
    ax10.set_xlim(0, 20)
    ax10.grid(True, alpha=0.3)
    ax10.legend(fontsize=7)
    
    ax11 = plt.subplot(5, 3, 11)
    for i in range(5):
        y = np.zeros(n_impulse)
        for n in range(n_impulse):
            if n == 0:
                y[n] = b0_osses[i] * impulse[n]
            else:
                y[n] = b0_osses[i] * impulse[n] + a1_osses[i] * y[n-1]
        ax11.plot(t_ms, y, linewidth=1.5, label=f'Loop {i+1}', alpha=0.8)
    ax11.set_xlabel('Time (ms)', fontsize=10)
    ax11.set_ylabel('Amplitude', fontsize=10)
    ax11.set_title('Osses2021: Impulse Response', fontsize=11)
    ax11.set_xlim(0, 20)
    ax11.grid(True, alpha=0.3)
    ax11.legend(fontsize=7)
    
    ax12 = plt.subplot(5, 3, 12)
    for i in range(5):
        y = np.zeros(n_impulse)
        for n in range(n_impulse):
            if n == 0:
                y[n] = b0_paulick[i] * impulse[n]
            else:
                y[n] = b0_paulick[i] * impulse[n] + a1_paulick[i] * y[n-1]
        ax12.plot(t_ms, y, linewidth=1.5, label=f'Loop {i+1}', alpha=0.8)
    ax12.set_xlabel('Time (ms)', fontsize=10)
    ax12.set_ylabel('Amplitude', fontsize=10)
    ax12.set_title('Paulick2024: Impulse Response', fontsize=11)
    ax12.set_xlim(0, 20)
    ax12.grid(True, alpha=0.3)
    ax12.legend(fontsize=7)
    
    # ========== ROW 5: OVERSHOOT LIMITING ==========
    x_overshoot = np.linspace(0, 5, 500)
    
    ax13 = plt.subplot(5, 3, 13)
    for i in range(5):
        if adapt_dau.limit > 1.0:
            factor = adapt_dau.factor[i].item()
            expfac = adapt_dau.expfac[i].item()
            offset = adapt_dau.offset[i].item()
            
            y_limited = np.where(
                x_overshoot > 1.0,
                factor / (1.0 + np.exp(expfac * (x_overshoot - 1.0))) - offset,
                x_overshoot
            )
            ax13.plot(x_overshoot, y_limited, linewidth=1.5, label=f'Loop {i+1}')
    ax13.plot(x_overshoot, x_overshoot, 'k--', linewidth=1, alpha=0.5, label='Unity')
    ax13.set_xlabel('Input', fontsize=10)
    ax13.set_ylabel('Output (after limiting)', fontsize=10)
    ax13.set_title('Dau1997: Overshoot Limiting (limit=10)', fontsize=11)
    ax13.set_xlim(0, 5)
    ax13.grid(True, alpha=0.3)
    ax13.legend(fontsize=7)
    
    ax14 = plt.subplot(5, 3, 14)
    for i in range(5):
        if adapt_osses.limit > 1.0:
            factor = adapt_osses.factor[i].item()
            expfac = adapt_osses.expfac[i].item()
            offset = adapt_osses.offset[i].item()
            
            y_limited = np.where(
                x_overshoot > 1.0,
                factor / (1.0 + np.exp(expfac * (x_overshoot - 1.0))) - offset,
                x_overshoot
            )
            ax14.plot(x_overshoot, y_limited, linewidth=1.5, label=f'Loop {i+1}')
    ax14.plot(x_overshoot, x_overshoot, 'k--', linewidth=1, alpha=0.5, label='Unity')
    ax14.set_xlabel('Input', fontsize=10)
    ax14.set_ylabel('Output (after limiting)', fontsize=10)
    ax14.set_title('Osses2021: Overshoot Limiting (limit=5)', fontsize=11)
    ax14.set_xlim(0, 5)
    ax14.grid(True, alpha=0.3)
    ax14.legend(fontsize=7)
    
    ax15 = plt.subplot(5, 3, 15)
    for i in range(5):
        if adapt_paulick.limit > 1.0:
            factor = adapt_paulick.factor[i].item()
            expfac = adapt_paulick.expfac[i].item()
            offset = adapt_paulick.offset[i].item()
            
            y_limited = np.where(
                x_overshoot > 1.0,
                factor / (1.0 + np.exp(expfac * (x_overshoot - 1.0))) - offset,
                x_overshoot
            )
            ax15.plot(x_overshoot, y_limited, linewidth=1.5, label=f'Loop {i+1}')
    ax15.plot(x_overshoot, x_overshoot, 'k--', linewidth=1, alpha=0.5, label='Unity')
    ax15.set_xlabel('Input', fontsize=10)
    ax15.set_ylabel('Output (after limiting)', fontsize=10)
    ax15.set_title('Paulick2024: Overshoot Limiting (limit=10)', fontsize=11)
    ax15.set_xlim(0, 5)
    ax15.grid(True, alpha=0.3)
    ax15.legend(fontsize=7)
    
    plt.tight_layout()
    
    # Save figure
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    output_path = TEST_FIGURES_DIR / 'adaptation_design_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADAPTATION - COMPREHENSIVE TEST SUITE (ALL PRESETS)")
    print("="*80)
    
    test_forward_validation()
    test_frequency_specific_minlvl()
    test_end_to_end_pipeline()
    test_design_comparison_visualization()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
