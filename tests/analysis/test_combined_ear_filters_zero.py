"""
Zero-Phase vs Minimum-Phase Comparison - Test Suite

Contents:
1. test_zero_vs_minimum_phase_comparison: Complete comparison between filtfilt (zero-phase) and minimum-phase filtering

Structure:
- Impulse response comparison (headphone + middle ear)
- Frequency response comparison (magnitude and phase)
- Processed logarithmic sweep comparison
- Comparative spectrograms
- Quantitative analysis of phase variance and magnitude matching

Figures generated:
- zero_vs_minimum_phase_comparison.png: 6-row comprehensive comparison with complete pipeline
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import HeadphoneFilter, MiddleEarFilter


def test_zero_vs_minimum_phase_comparison():
    """Test zero-phase (filtfilt) vs minimum-phase filtering for ear filters."""
    
    # Create test_figures directory
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100
    print("="*80)
    print("ZERO-PHASE VS MINIMUM-PHASE COMPARISON")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Sampling rate: {fs} Hz")

    # Create filters (both minimum and zero phase)
    hpf_min = HeadphoneFilter(fs=fs, order=512, phase_type='minimum')
    hpf_zero = HeadphoneFilter(fs=fs, order=512, phase_type='zero')

    mef_min = MiddleEarFilter(fs=fs, filter_type='lopezpoveda2001', order=512,
                           phase_type='minimum', normalize_gain=True)
    mef_zero = MiddleEarFilter(fs=fs, filter_type='lopezpoveda2001', order=512,
                            phase_type='zero', normalize_gain=True)

    print(f"\nFilters:")
    print(f"  Headphone (minimum): {hpf_min}")
    print(f"  Headphone (zero):    {hpf_zero}")
    print(f"  Middle Ear (minimum): {mef_min}")
    print(f"  Middle Ear (zero):    {mef_zero}")

    # ============================================================================
    # TEST 1: Impulse Response Comparison
    # ============================================================================
    print(f"\n[1] Impulse Response Comparison")
    impulse_duration = 0.5  # seconds
    impulse = torch.zeros(int(fs * impulse_duration), dtype=torch.float32)
    impulse[0] = 1.0
    impulse = impulse.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        # Headphone filters
        impulse_hp_min = hpf_min(impulse).squeeze(0).numpy()
        impulse_hp_zero = hpf_zero(impulse).squeeze(0).numpy()
        
        # Middle ear filters
        impulse_me_min = mef_min(impulse).squeeze(0).numpy()
        impulse_me_zero = mef_zero(impulse).squeeze(0).numpy()

    print(f"  Impulse responses computed")

    # ============================================================================
    # TEST 2: Frequency Response Comparison
    # ============================================================================
    print(f"\n[2] Frequency Response Comparison")

    # Headphone
    freqs_hp, H_hp_min = hpf_min.get_frequency_response(nfft=8192)
    _, H_hp_zero = hpf_zero.get_frequency_response(nfft=8192)

    freqs_np = freqs_hp.cpu().numpy()
    H_hp_min_mag = 20 * torch.log10(torch.abs(H_hp_min) + 1e-10).cpu().numpy()
    H_hp_zero_mag = 20 * torch.log10(torch.abs(H_hp_zero) + 1e-10).cpu().numpy()
    H_hp_min_phase = torch.angle(H_hp_min).cpu().numpy()
    H_hp_zero_phase = torch.angle(H_hp_zero).cpu().numpy()

    # Middle ear
    freqs_me, H_me_min = mef_min.get_frequency_response(nfft=8192)
    _, H_me_zero = mef_zero.get_frequency_response(nfft=8192)

    H_me_min_mag = 20 * torch.log10(torch.abs(H_me_min) + 1e-10).cpu().numpy()
    H_me_zero_mag = 20 * torch.log10(torch.abs(H_me_zero) + 1e-10).cpu().numpy()
    H_me_min_phase = torch.angle(H_me_min).cpu().numpy()
    H_me_zero_phase = torch.angle(H_me_zero).cpu().numpy()

    print(f"  Headphone (minimum) phase range: {H_hp_min_phase.min():.3f} to {H_hp_min_phase.max():.3f} rad")
    print(f"  Headphone (zero) phase range:    {H_hp_zero_phase.min():.3f} to {H_hp_zero_phase.max():.3f} rad")
    print(f"  Middle ear (minimum) phase range: {H_me_min_phase.min():.3f} to {H_me_min_phase.max():.3f} rad")
    print(f"  Middle ear (zero) phase range:    {H_me_zero_phase.min():.3f} to {H_me_zero_phase.max():.3f} rad")

    # ============================================================================
    # TEST 3: Logarithmic Sweep Comparison
    # ============================================================================
    print(f"\n[3] Logarithmic Sweep Comparison")
    sweep_duration = 5.0  # seconds
    t = np.linspace(0, sweep_duration, int(fs * sweep_duration), endpoint=False)

    # Logarithmic sweep from 20 Hz to fs/2
    f0, f1 = 20, fs / 2
    k = (f1 - f0) / sweep_duration
    sweep = np.sin(2 * np.pi * (f0 * t + (k / 2) * t**2))

    sweep_tensor = torch.tensor(sweep, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # Headphone
        sweep_hp_min = hpf_min(sweep_tensor).squeeze(0).numpy()
        sweep_hp_zero = hpf_zero(sweep_tensor).squeeze(0).numpy()
        
        # Middle ear
        sweep_me_min = mef_min(sweep_tensor).squeeze(0).numpy()
        sweep_me_zero = mef_zero(sweep_tensor).squeeze(0).numpy()
        
        # Full pipeline: headphone + middle ear (zero-phase)
        sweep_pipeline_zero = mef_zero(hpf_zero(sweep_tensor)).squeeze(0).numpy()

    print(f"  Sweep processed through all filters")

    # ============================================================================
    # PLOTTING
    # ============================================================================
    print(f"\nGenerating plots...")

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('Zero-Phase vs Minimum-Phase Filtering Comparison', 
             fontsize=16, fontweight='bold', y=0.98)

    # Create 6x2 grid (last row spans both columns)
    gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3)

    # --- Row 1: Headphone Impulse Responses ---
    ax1 = fig.add_subplot(gs[0, 0])
    t_impulse = np.arange(1000) / fs * 1000  # first 1000 samples in ms
    ax1.plot(t_impulse, impulse_hp_min[:1000], 'b-', linewidth=1, label='Minimum-phase', alpha=0.8)
    ax1.plot(t_impulse, impulse_hp_zero[:1000], 'r--', linewidth=1, label='Zero-phase', alpha=0.8)
    ax1.set_title('1a. Headphone - Impulse Response (first 1000 samples)', fontsize=11)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_impulse, impulse_me_min[:1000], 'b-', linewidth=1, label='Minimum-phase', alpha=0.8)
    ax2.plot(t_impulse, impulse_me_zero[:1000], 'r--', linewidth=1, label='Zero-phase', alpha=0.8)
    ax2.set_title('1b. Middle Ear - Impulse Response (first 1000 samples)', fontsize=11)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- Row 2: Frequency Response Magnitude ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogx(freqs_np, H_hp_min_mag, 'b-', linewidth=1.5, label='Minimum-phase', alpha=0.8)
    ax3.semilogx(freqs_np, H_hp_zero_mag, 'r--', linewidth=1.5, label='Zero-phase', alpha=0.8)
    ax3.set_title('2a. Headphone - Frequency Response (Magnitude)', fontsize=11)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Magnitude [dB]')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([20, fs/2])
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(freqs_np, H_me_min_mag, 'b-', linewidth=1.5, label='Minimum-phase', alpha=0.8)
    ax4.semilogx(freqs_np, H_me_zero_mag, 'r--', linewidth=1.5, label='Zero-phase', alpha=0.8)
    ax4.set_title('2b. Middle Ear - Frequency Response (Magnitude)', fontsize=11)
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Magnitude [dB]')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim([20, fs/2])
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.legend()

    # --- Row 3: Phase Response Comparison ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogx(freqs_np, H_hp_min_phase, 'b-', linewidth=1.5, label='Minimum-phase', alpha=0.8)
    ax5.semilogx(freqs_np, H_hp_zero_phase, 'r--', linewidth=1.5, label='Zero-phase', alpha=0.8)
    ax5.set_title('3a. Headphone - Phase Response', fontsize=11)
    ax5.set_xlabel('Frequency [Hz]')
    ax5.set_ylabel('Phase [rad]')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.set_xlim([20, fs/2])
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.semilogx(freqs_np, H_me_min_phase, 'b-', linewidth=1.5, label='Minimum-phase', alpha=0.8)
    ax6.semilogx(freqs_np, H_me_zero_phase, 'r--', linewidth=1.5, label='Zero-phase', alpha=0.8)
    ax6.set_title('3b. Middle Ear - Phase Response', fontsize=11)
    ax6.set_xlabel('Frequency [Hz]')
    ax6.set_ylabel('Phase [rad]')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.set_xlim([20, fs/2])
    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax6.legend()

    # --- Row 4: Sweep Time Domain Comparison ---
    plot_samples = 2000
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(t[:plot_samples], sweep[:plot_samples], 'gray', linewidth=0.5, 
         label='Input', alpha=0.5)
    ax7.plot(t[:plot_samples], sweep_hp_min[:plot_samples], 'b-', linewidth=0.8, 
         label='Minimum-phase', alpha=0.8)
    ax7.plot(t[:plot_samples], sweep_hp_zero[:plot_samples], 'r--', linewidth=0.8, 
         label='Zero-phase', alpha=0.8)
    ax7.set_title('4a. Headphone - Sweep Time Domain (0.00-0.05s)', fontsize=11)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Amplitude')
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(t[:plot_samples], sweep[:plot_samples], 'gray', linewidth=0.5, 
         label='Input', alpha=0.5)
    ax8.plot(t[:plot_samples], sweep_me_min[:plot_samples], 'b-', linewidth=0.8, 
         label='Minimum-phase', alpha=0.8)
    ax8.plot(t[:plot_samples], sweep_me_zero[:plot_samples], 'r--', linewidth=0.8, 
         label='Zero-phase', alpha=0.8)
    ax8.set_title('4b. Middle Ear - Sweep Time Domain (0.00-0.05s)', fontsize=11)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Amplitude')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # --- Row 5: Phase Shift Visualization (Zoom) ---
    ax9 = fig.add_subplot(gs[4, 0])
    zoom_samples = int(0.05 * fs)  # 0.00-0.05s
    ax9.plot(t[:zoom_samples], sweep[:zoom_samples], 'gray', linewidth=1.5, 
         label='Input', alpha=0.7, marker='o', markersize=2, markevery=10)
    ax9.plot(t[:zoom_samples], sweep_hp_min[:zoom_samples], 'b-', linewidth=1.5, 
         label='Minimum-phase', marker='s', markersize=2, markevery=10, alpha=0.8)
    ax9.plot(t[:zoom_samples], sweep_hp_zero[:zoom_samples], 'r--', linewidth=1.5, 
         label='Zero-phase', marker='^', markersize=2, markevery=10, alpha=0.8)

    # Add connecting lines for minimum-phase (blue)
    for i in range(0, zoom_samples, 10):
        ax9.plot([t[i], t[i]], [sweep[i], sweep_hp_min[i]], 
                 'blue', linestyle='--', linewidth=0.5, alpha=0.4)

    # Add connecting lines for zero-phase (red, should be minimal)
    for i in range(0, zoom_samples, 10):
        ax9.plot([t[i], t[i]], [sweep[i], sweep_hp_zero[i]], 
                 'red', linestyle='--', linewidth=0.5, alpha=0.4)

    ax9.set_title('5a. Headphone - Phase Shift Visualization', fontsize=11)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Amplitude')
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    ax10 = fig.add_subplot(gs[4, 1])
    ax10.plot(t[:zoom_samples], sweep[:zoom_samples], 'gray', linewidth=1.5, 
          label='Input', alpha=0.7, marker='o', markersize=2, markevery=10)
    ax10.plot(t[:zoom_samples], sweep_me_min[:zoom_samples], 'b-', linewidth=1.5, 
          label='Minimum-phase', marker='s', markersize=2, markevery=10, alpha=0.8)
    ax10.plot(t[:zoom_samples], sweep_me_zero[:zoom_samples], 'r--', linewidth=1.5, 
          label='Zero-phase', marker='^', markersize=2, markevery=10, alpha=0.8)

    # Add connecting lines
    for i in range(0, zoom_samples, 10):
        ax10.plot([t[i], t[i]], [sweep[i], sweep_me_min[i]], 
                  'blue', linestyle='--', linewidth=0.5, alpha=0.4)

    for i in range(0, zoom_samples, 10):
        ax10.plot([t[i], t[i]], [sweep[i], sweep_me_zero[i]], 
                  'red', linestyle='--', linewidth=0.5, alpha=0.4)

    ax10.set_title('5b. Middle Ear - Phase Shift Visualization', fontsize=11)
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('Amplitude')
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    # --- Row 6: Full Pipeline (Headphone + Middle Ear Zero-Phase) ---
    ax11 = fig.add_subplot(gs[5, :])
    ax11.plot(t[:plot_samples], sweep[:plot_samples], 'gray', linewidth=1.5, 
          label='Input', alpha=0.7)
    ax11.plot(t[:plot_samples], sweep_pipeline_zero[:plot_samples], 'green', linewidth=1.5, 
          label='Zero-phase Pipeline (Headphone + Middle Ear)', alpha=0.9)
    ax11.set_title('6. Full Pipeline - Input vs Zero-Phase Output (0.00-0.05s)', fontsize=12)
    ax11.set_xlabel('Time [s]')
    ax11.set_ylabel('Amplitude')
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = TEST_FIGURES_DIR / 'zero_vs_minimum_phase_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    # ============================================================================
    # QUANTITATIVE COMPARISON
    # ============================================================================
    print(f"\n" + "="*80)
    print("QUANTITATIVE PHASE COMPARISON")
    print("="*80)

    # Compute phase variance
    hp_min_phase_var = np.var(H_hp_min_phase)
    hp_zero_phase_var = np.var(H_hp_zero_phase)
    me_min_phase_var = np.var(H_me_min_phase)
    me_zero_phase_var = np.var(H_me_zero_phase)

    print(f"\nPhase Variance:")
    print(f"  Headphone (minimum): {hp_min_phase_var:.6f} rad²")
    print(f"  Headphone (zero):    {hp_zero_phase_var:.6f} rad² ({hp_zero_phase_var/hp_min_phase_var*100:.2f}% of minimum)")
    print(f"  Middle Ear (minimum): {me_min_phase_var:.6f} rad²")
    print(f"  Middle Ear (zero):    {me_zero_phase_var:.6f} rad² ({me_zero_phase_var/me_min_phase_var*100:.2f}% of minimum)")

    # Compute magnitude matching
    hp_mag_diff = np.mean(np.abs(H_hp_min_mag - H_hp_zero_mag))
    me_mag_diff = np.mean(np.abs(H_me_min_mag - H_me_zero_mag))

    print(f"\nMagnitude Response Matching:")
    print(f"  Headphone mean difference: {hp_mag_diff:.4f} dB")
    print(f"  Middle Ear mean difference: {me_mag_diff:.4f} dB")

    print("="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_zero_vs_minimum_phase_comparison()
