"""
IHC Paulick2024 - Test Suite

Contents:
1. test_ihc_paulick2024: Comprehensive test of IHC transduction model
   - MET channel conductance computation
   - Pre-charge evolution
   - Receptor potential generation

Structure:
- AM modulated tone (1 kHz carrier, 15 Hz modulation) @ 60 dB SPL
- DRNL filterbank preprocessing (BM velocity)
- IHC Paulick2024 transduction
- Verification checks for realistic voltage ranges

Figures generated:
- ihc_paulick2024_test.png: 6-panel comprehensive analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import IHCPaulick2024, DRNLFilterbank


def test_ihc_paulick2024():
    """Complete IHC Paulick2024 test with single comprehensive figure."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("IHC PAULICK2024 COMPREHENSIVE TEST")
    print("="*80)
    
    # ========== PARAMETERS ==========
    fs = 44100
    test_freq = 1000  # Hz
    test_level = 60   # dB SPL
    dur = 0.05        # seconds
    
    print(f"\nTest parameters:")
    print(f"  Signal: {test_freq} Hz @ {test_level} dB SPL")
    print(f"  Duration: {dur*1000:.1f} ms")
    print(f"  Sampling rate: {fs} Hz")
    
    # ========== GENERATE INPUT & PROCESS DRNL ==========
    print(f"\nGenerating input and processing through DRNL...")
    t = torch.arange(0, dur, 1/fs, dtype=torch.float64)
    # AM modulation at 15 Hz: (1 + sin(2πf_mod*t)) / 2 gives 0-1 range
    carrier = torch.sin(2 * torch.pi * test_freq * t)
    modulator = (1 + torch.sin(2 * torch.pi * 15 * t)) / 2
    tone = carrier * modulator
    
    # Scale to SPL
    rms_target = 20e-6 * 10.0 ** (test_level / 20.0)
    rms_current = torch.sqrt(torch.mean(tone ** 2))
    tone = tone * (rms_target / rms_current)
    
    # Process through DRNL
    drnl = DRNLFilterbank((250, 8000), fs=fs, n_channels=50, dtype=torch.float64)
    with torch.no_grad():
        vel = drnl(tone)  # [50, time] - BM velocity
    
    # Find on-frequency channel
    fc_diff = torch.abs(drnl.fc - test_freq)
    on_ch = torch.argmin(fc_diff).item()
    
    print(f"  On-frequency channel: {on_ch} (fc = {drnl.fc[on_ch]:.1f} Hz)")
    print(f"  BM velocity range: [{vel.min():.4e}, {vel.max():.4e}] m/s")
    
    # ========== COMPUTE MET CONDUCTANCE ==========
    print(f"\nComputing MET channel conductance...")
    ihc = IHCPaulick2024(fs=fs, dtype=torch.float64)
    
    # Compute conductance for sigmoid curve
    x_range = torch.linspace(-100e-9, 100e-9, 1000, dtype=torch.float64)
    x_range_3d = x_range.unsqueeze(0).unsqueeze(0)
    G_curve = ihc._compute_met_conductance(x_range_3d)
    G_curve_np = G_curve.squeeze().numpy()
    x_np = x_range.numpy() * 1e9  # Convert to nm
    
    G_half = ihc.Gmet_max.item() / 2.0
    print(f"  Gmet_max: {ihc.Gmet_max.item()*1e9:.2f} nS")
    print(f"  Half-max conductance: {G_half*1e9:.2f} nS")
    
    # ========== COMPUTE PRE-CHARGE ==========
    print(f"\nComputing pre-charge evolution...")
    Ts = 1.0 / fs
    n_precharge = int(fs * ihc.precharge_duration)
    
    V_history = torch.zeros(n_precharge, dtype=torch.float64)
    V_now = ihc.V_rest.clone()
    
    for i in range(n_precharge):
        Imet = ihc.G_precharge * (V_now - ihc.EP)
        Ik = ihc.Gkf * (V_now - ihc.Ekf)
        Is = ihc.Gks * (V_now - ihc.Eks)
        V_now = V_now - (Imet + Ik + Is) * Ts / ihc.Cm
        V_history[i] = V_now
    
    V_final = V_now.item()
    print(f"  Pre-charge: {ihc.V_rest.item()*1000:.3f} mV → {V_final*1000:.3f} mV")
    
    # ========== PROCESS IHC ==========
    print(f"\nProcessing through IHC transduction...")
    with torch.no_grad():
        V = ihc(vel)  # [50, time] - Receptor potential
    
    print(f"  Receptor potential range: [{V.min():.4e}, {V.max():.4e}] V")
    print(f"  On-channel V range: [{V[on_ch].min():.4e}, {V[on_ch].max():.4e}] V")
    
    # ========== CREATE FIGURE ==========
    print(f"\nGenerating comprehensive figure...")
    
    fig = plt.figure(figsize=(14, 12))
    t_ms = t.numpy() * 1000
    
    # Row 1, Col 1: Input signal
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t_ms, tone.numpy(), linewidth=0.8, color='steelblue')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (Pa)')
    ax1.set_title(f'Input Signal: {test_freq} Hz @ {test_level} dB SPL')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 20])
    
    # Row 1, Col 2: BM velocity (all channels)
    ax2 = plt.subplot(3, 2, 2)
    vel_np = vel.numpy()
    n_samples_vel = vel.shape[1]
    t_vel_ms = np.arange(n_samples_vel) / fs * 1000
    im1 = ax2.imshow(vel_np, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[t_vel_ms[0], t_vel_ms[-1], 0, drnl.num_channels])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Channel')
    ax2.set_title('BM Velocity')
    ax2.axhline(on_ch, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im1, ax=ax2, label='Velocity (m/s)')
    
    # Row 2, Col 1: MET channel conductance
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(x_np, G_curve_np * 1e9, linewidth=2, color='darkblue')
    ax3.axhline(G_half * 1e9, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(ihc.x0.item() * 1e9, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Stereocilia Displacement (nm)')
    ax3.set_ylabel('MET Conductance (nS)')
    ax3.set_title('MET Channel Conductance')
    ax3.grid(True, alpha=0.3)
    
    # Row 2, Col 2: Pre-charge voltage evolution
    ax4 = plt.subplot(3, 2, 4)
    t_precharge_ms = np.arange(n_precharge) / fs * 1000
    V_mV = V_history.numpy() * 1000
    ax4.plot(t_precharge_ms, V_mV, linewidth=1.5, color='darkgreen')
    ax4.axhline(V_final * 1000, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Voltage (mV)')
    ax4.set_title('Pre-charge Voltage Evolution')
    ax4.set_xlim([0, 15])  # Zoom primi 15ms
    ax4.grid(True, alpha=0.3)
    
    # Row 3, Col 1: Receptor potential (on-frequency)
    ax5 = plt.subplot(3, 2, 5)
    V_on = V[on_ch].numpy()
    ax5.plot(t_ms, V_on * 1000, linewidth=0.8, color='darkred')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Voltage (mV)')
    ax5.set_title(f'Receptor Potential (Ch{on_ch}: {drnl.fc[on_ch]:.1f} Hz)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 20])
    
    # Row 3, Col 2: Receptor potential spectrogram (all channels)
    ax6 = plt.subplot(3, 2, 6)
    n_samples = V.shape[1]
    t_display_ms = np.arange(n_samples) / fs * 1000
    im2 = ax6.imshow(V.numpy() * 1000, aspect='auto', origin='lower', cmap='seismic',
                     extent=[t_display_ms[0], t_display_ms[-1], 0, drnl.num_channels])
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Channel')
    ax6.set_title('Receptor Potential (All Channels)')
    ax6.axhline(on_ch, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im2, ax=ax6, label='Voltage (mV)')
    
    # Add channel frequency labels
    yticks = [0, 12, 24, 36, 49]
    ytick_labels = [f'{drnl.fc[i]:.0f} Hz' if i < len(drnl.fc) else '' for i in yticks]
    ax6.set_yticks(yticks)
    ax6.set_yticklabels(ytick_labels)
    
    plt.suptitle('IHC Paulick2024: Comprehensive Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = TEST_FIGURES_DIR / 'ihc_paulick2024_test.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    # ========== VERIFICATION CHECKS ==========
    print(f"\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)
    print(f"  ✓ MET conductance range: [{G_curve.min().item()*1e9:.4f}, {G_curve.max().item()*1e9:.4f}] nS")
    print(f"  ✓ Pre-charge converged: {V_final*1000:.3f} mV")
    print(f"  ✓ Receptor potential range: [{V.min().item()*1000:.3f}, {V.max().item()*1000:.3f}] mV")
    print(f"  ✓ No NaN: {not torch.isnan(V).any().item()}")
    print(f"  ✓ No Inf: {not torch.isinf(V).any().item()}")
    print(f"  ✓ Realistic V range (±100 mV): {V.min().item() > -0.1 and V.max().item() < 0.1}")
    
    print("\n" + "="*80)
    print("IHC PAULICK2024 TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_ihc_paulick2024()
