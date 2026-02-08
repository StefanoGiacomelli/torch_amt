"""
Compression Comparison - Test Suite

Contents:
1. test_compression_comparison: Compares BrokenStick vs PowerLaw compression algorithms
   with swept amplitude envelope signal, showing transfer curves with varying exponents

Structure:
- Time-domain waveform comparison with amplitude sweep
- Transfer curve visualization for both algorithms
- Varying exponent analysis (BrokenStick: 0.2-0.5, PowerLaw: 0.3-0.6)

Figures generated:
- compression_comparison.png: 2x2 grid showing waveforms and transfer curves
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_amt.common import BrokenStickCompression, PowerCompression


def test_compression_comparison():
    """Compare BrokenStick and PowerLaw compression with direct transfer curve mapping."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("COMPRESSION ALGORITHM COMPARISON TEST")
    print("="*80)
    
    # Compression parameters
    dboffset = 120.0        # dB SPL reference (default for Threshold of Pain)
    knee_db = 80.0          # Knee point: 80 [dB ref]
    exponent_broken = 0.3   # BrokenStick exponent
    exponent_power = 0.4    # PowerLaw exponent
    
    print("\nCompression Parameters:")
    print(f"  dboffset: {dboffset} dB SPL (reference level)")
    print(f"  knee_db: {knee_db} dB SPL")
    print(f"  → knee_linear = 10^(({knee_db} - {dboffset})/20) = {10**((knee_db - dboffset)/20):.6e}")
    print(f"  → knee in dB SPL: {knee_db} dB")
    print(f"  BrokenStick exponent: {exponent_broken}")
    print(f"  PowerLaw exponent: {exponent_power}")
    
    # =========================================================================
    # PART 1: Time-domain signal test
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: Time-Domain Signal Test")
    print("="*80)
    
    # Parameters for time-domain test
    fs = 44100
    duration = 0.25 # seconds
    fc = 100        # Carrier frequency (Hz)
    
    # Amplitude sweep in dB SPL: 40 dB SPL to 90 dB SPL (realistic range)
    amp_start_db_spl = 100.0  # Loud sound level
    amp_min_db_spl = 60.0     # Speech level
    
    # Convert to dB re: dboffset
    amp_start_db = amp_start_db_spl - dboffset
    amp_min_db = amp_min_db_spl - dboffset
    
    print(f"\nTime-Domain Signal Parameters:")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Carrier frequency: {fc} Hz")
    print(f"  dboffset for audio: {dboffset} dB SPL")
    print(f"  Amplitude sweep: {amp_start_db_spl} dB SPL to {amp_min_db_spl} dB SPL")
    print(f"  → In dB SPL: {amp_start_db:.1f} to {amp_min_db:.1f} dB SPL")
    
    # Generate time vector and swept signal
    n_samples = int(fs * duration)
    t = torch.linspace(0, duration, n_samples)
    carrier = torch.sin(2 * torch.pi * fc * t)
    
    # Linear sweep in dB
    amp_db = torch.linspace(amp_start_db, amp_min_db, n_samples)
    amp_linear = 10 ** (amp_db / 20.0)
    signal = amp_linear * carrier
    print(f"  Input amplitude range: {amp_linear.min().item():.6f} to {amp_linear.max().item():.6f}")
    
    # Apply compressions with audio dboffset
    broken_stick_audio = BrokenStickCompression(knee_db=knee_db, exponent=exponent_broken, dboffset=dboffset)
    power_law_audio = PowerCompression(knee_db=knee_db, exponent=exponent_power, dboffset=dboffset)
    
    with torch.no_grad():
        signal_broken = broken_stick_audio(signal.unsqueeze(-1)).squeeze()
        signal_power = power_law_audio(signal.unsqueeze(-1)).squeeze()
    
    print(f"  BrokenStick output range: {signal_broken.min().item():.6f} to {signal_broken.max().item():.6f}")
    print(f"  PowerLaw output range: {signal_power.min().item():.6f} to {signal_power.max().item():.6f}")
    
    # =========================================================================
    # PART 2: Transfer curves with varying exponents
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: Transfer Curves with Varying Exponents")
    print("="*80)
    
    # Create input signal sweep in dB
    input_db_range = torch.linspace(-100, 10, 1000)
    input_linear = 10 ** (input_db_range / 20.0)
    
    # Test with different exponents
    exponents_broken = [0.2, 0.3, 0.4, 0.5]
    exponents_power = [0.3, 0.4, 0.5, 0.6]
    
    print(f"\nTesting BrokenStick with exponents: {exponents_broken}")
    print(f"Testing PowerLaw with exponents: {exponents_power}")
    
    outputs_broken_var = {}
    outputs_power_var = {}
    
    for n in exponents_broken:
        comp = BrokenStickCompression(knee_db=knee_db, exponent=n, dboffset=dboffset)
        with torch.no_grad():
            out = comp(input_linear.unsqueeze(-1)).squeeze()
        out_db = 20 * torch.log10(torch.abs(out) + 1e-12)
        outputs_broken_var[n] = out_db.numpy()
    
    for n in exponents_power:
        comp = PowerCompression(knee_db=knee_db, exponent=n, dboffset=dboffset)
        with torch.no_grad():
            out = comp(input_linear.unsqueeze(-1)).squeeze()
        out_db = 20 * torch.log10(torch.abs(out) + 1e-12)
        outputs_power_var[n] = out_db.numpy()
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*80)
    print("Creating Visualization...")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Compression Algorithms Comparison', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Time-domain waveforms (zoomed)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(t.numpy(), signal.numpy(), 'b-', linewidth=1.5, label='Input Signal', alpha=0.7)
    ax1.plot(t.numpy(), signal_broken.numpy(), 'r-', linewidth=1.2, label='BrokenStick Output', alpha=0.8)
    ax1.plot(t.numpy(), signal_power.numpy(), 'g-', linewidth=1.0, label='PowerLaw Output', alpha=0.8)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveforms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 3: BrokenStick Transfer Curves with varying n
    ax3 = fig.add_subplot(gs[1, 0])
    input_db_np = input_db_range.numpy()
    
    ax3.plot(input_db_np, input_db_np, 'k--', linewidth=0.8, alpha=0.3, label='Unity gain')
    ax3.axvline(knee_db, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(exponents_broken)))
    for n, color in zip(exponents_broken, colors):
        label = f'n={n}' + (' (default)' if n == exponent_broken else '')
        linewidth = 2.5 if n == exponent_broken else 1.8
        linestyle = '--' if n == exponent_broken else '-'
        ax3.plot(input_db_np, outputs_broken_var[n], color=color, linewidth=linewidth, linestyle=linestyle, label=label, alpha=0.9)
    
    ax3.set_xlabel(f'Input Level [dB]', fontsize=10)
    ax3.set_ylabel(f'Output Level [dB]', fontsize=10)
    ax3.set_title(f'BrokenStick Transfer Curves (knee={knee_db} dB, ref: {dboffset} dB)', fontsize=11)
    ax3.set_xlim([-100, 10])
    ax3.set_ylim([-100, 10])
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add secondary axes for dB SPL values
    ax3_top = ax3.twiny()
    ax3_right = ax3.twinx()
    ax3_top.set_xlim([ax3.get_xlim()[0] + dboffset, ax3.get_xlim()[1] + dboffset])
    ax3_right.set_ylim([ax3.get_ylim()[0] + dboffset, ax3.get_ylim()[1] + dboffset])
    ax3_top.set_xlabel(f'Input Level [dB SPL]', fontsize=9, color='gray')
    ax3_right.set_ylabel(f'Output Level [dB SPL]', fontsize=9, color='gray')
    ax3_top.tick_params(axis='x', labelsize=8, colors='gray')
    ax3_right.tick_params(axis='y', labelsize=8, colors='gray')
    
    # Plot 4: PowerLaw Transfer Curves with varying n
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(input_db_np, input_db_np, 'k--', linewidth=0.8, alpha=0.3, label='Unity gain')
    ax4.axvline(knee_db, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(exponents_power)))
    for n, color in zip(exponents_power, colors):
        label = f'n={n}' + (' (default)' if n == exponent_power else '')
        linewidth = 2.5 if n == exponent_power else 1.8
        linestyle = '--' if n == exponent_power else '-'
        ax4.plot(input_db_np, outputs_power_var[n], color=color, linewidth=linewidth, linestyle=linestyle, label=label, alpha=0.9)
    
    ax4.set_xlabel('Input Level [dB]', fontsize=10)
    ax4.set_ylabel('Output Level [dB]', fontsize=10)
    ax4.set_title(f'PowerLaw Transfer Curves (knee={knee_db} dB, ref: {dboffset} dB)', fontsize=11)
    ax4.set_xlim([-100, 10])
    ax4.set_ylim([-100, 10])
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Add secondary axes for dB SPL values
    ax4_top = ax4.twiny()
    ax4_right = ax4.twinx()
    ax4_top.set_xlim([ax4.get_xlim()[0] + dboffset, ax4.get_xlim()[1] + dboffset])
    ax4_right.set_ylim([ax4.get_ylim()[0] + dboffset, ax4.get_ylim()[1] + dboffset])
    ax4_top.set_xlabel(f'Input Level [dB SPL]', fontsize=9, color='gray')
    ax4_right.set_ylabel(f'Output Level [dB SPL]', fontsize=9, color='gray')
    ax4_top.tick_params(axis='x', labelsize=8, colors='gray')
    ax4_right.tick_params(axis='y', labelsize=8, colors='gray')
    
    # Save figure
    output_path = TEST_FIGURES_DIR / 'compression_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    print("="*80)
    
    plt.close()


if __name__ == '__main__':
    test_compression_comparison()
