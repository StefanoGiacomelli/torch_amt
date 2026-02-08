"""
Middle Ear Filter - Test Suite

Contents:
1. test_middleear_filter_analysis: Complete middle ear filter analysis with both lopezpoveda2001 and jepsen2008 variants

Structure:
- Impulse response analysis (both variants)
- Frequency response comparison (magnitude and phase)
- Logarithmic sweep processing (20 Hz to fs/2)
- Spectrogram visualization for both variants
- Phase shift visualization

Figures generated:
- middleear_filter_analysis.png: 5-row comprehensive analysis comparing two implementations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import MiddleEarFilter


def test_middleear_filter_analysis():
    """Test middle ear filter with impulse response, frequency response and sweep for both variants."""
    
    # Create test_figures directory
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100
    print("="*80)
    print("MIDDLE EAR FILTER TEST")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Sampling rate: {fs} Hz")

    # Create filters (both variants)
    mef_lopez = MiddleEarFilter(fs=fs, filter_type='lopezpoveda2001', order=512, 
                             phase_type='minimum', normalize_gain=True)
    mef_jepsen = MiddleEarFilter(fs=fs, filter_type='jepsen2008', order=512,
                              phase_type='minimum', normalize_gain=True)

    print(f"\nMiddleEarFilter configurations:")
    print(f"  Lopez-Poveda 2001: {mef_lopez}")
    print(f"  Jepsen 2008: {mef_jepsen}")

    # ============================================================================
    # TEST 1: Impulse Response
    # ============================================================================
    print(f"\n[1] Impulse Response Test")
    impulse_duration = 0.5  # seconds
    impulse = torch.zeros(int(fs * impulse_duration), dtype=torch.float32)
    impulse[0] = 1.0
    impulse = impulse.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        impulse_response_lopez = mef_lopez(impulse).squeeze(0).numpy()
        impulse_response_jepsen = mef_jepsen(impulse).squeeze(0).numpy()

    print(f"  Impulse input: {impulse.shape}")
    print(f"  Lopez-Poveda response: {impulse_response_lopez.shape}")
    print(f"  Jepsen response: {impulse_response_jepsen.shape}")

    # ============================================================================
    # TEST 2: Frequency Response
    # ============================================================================
    print(f"\n[2] Frequency Response Analysis")
    freqs_lopez, H_lopez = mef_lopez.get_frequency_response(nfft=8192)
    freqs_jepsen, H_jepsen = mef_jepsen.get_frequency_response(nfft=8192)

    freqs_np = freqs_lopez.cpu().numpy()
    H_lopez_mag_db = 20 * torch.log10(torch.abs(H_lopez) + 1e-10).cpu().numpy()
    H_lopez_phase = torch.angle(H_lopez).cpu().numpy()

    H_jepsen_mag_db = 20 * torch.log10(torch.abs(H_jepsen) + 1e-10).cpu().numpy()
    H_jepsen_phase = torch.angle(H_jepsen).cpu().numpy()

    print(f"  Lopez-Poveda magnitude range: {H_lopez_mag_db.min():.2f} to {H_lopez_mag_db.max():.2f} dB")
    print(f"  Jepsen magnitude range: {H_jepsen_mag_db.min():.2f} to {H_jepsen_mag_db.max():.2f} dB")

    # ============================================================================
    # TEST 3: Logarithmic Frequency Sweep
    # ============================================================================
    print(f"\n[3] Logarithmic Sweep Test")
    sweep_duration = 5.0  # seconds
    t = np.linspace(0, sweep_duration, int(fs * sweep_duration), endpoint=False)

    # Logarithmic sweep from 20 Hz to fs/2
    f0, f1 = 20, fs / 2
    k = (f1 - f0) / sweep_duration
    sweep = np.sin(2 * np.pi * (f0 * t + (k / 2) * t**2))

    sweep_tensor = torch.tensor(sweep, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        filtered_sweep_lopez = mef_lopez(sweep_tensor).squeeze(0).numpy()
        filtered_sweep_jepsen = mef_jepsen(sweep_tensor).squeeze(0).numpy()

    print(f"  Sweep: {f0} Hz → {f1:.0f} Hz over {sweep_duration} s")
    print(f"  Input shape: {sweep_tensor.shape}")
    print(f"  Lopez-Poveda output: {filtered_sweep_lopez.shape}")
    print(f"  Jepsen output: {filtered_sweep_jepsen.shape}")

    # ============================================================================
    # PLOTTING
    # ============================================================================
    print(f"\nGenerating plots...")

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('Middle Ear Filter Analysis - Lopez-Poveda 2001 & Jepsen 2008', 
             fontsize=16, fontweight='bold', y=0.98)

    # Create 5x2 grid (added row for zoom plots)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

    # --- Row 1: Impulse Responses ---
    ax1 = fig.add_subplot(gs[0, 0])
    t_impulse = np.arange(len(impulse_response_lopez)) / fs * 1000  # ms
    ax1.plot(t_impulse[:1000], impulse_response_lopez[:1000], 'b-', linewidth=1, label='Lopez-Poveda 2001')
    ax1.set_title('1a. Impulse Response - Lopez-Poveda (first 1000 samples)', fontsize=11)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_impulse[:1000], impulse_response_jepsen[:1000], 'r-', linewidth=1, label='Jepsen 2008')
    ax2.set_title('1b. Impulse Response - Jepsen (first 1000 samples)', fontsize=11)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # --- Row 2: Frequency Response Magnitude ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogx(freqs_np, H_lopez_mag_db, 'b-', linewidth=1.5, label='Lopez-Poveda 2001')
    ax3.semilogx(freqs_np, H_jepsen_mag_db, 'r--', linewidth=1.5, label='Jepsen 2008', alpha=0.7)
    ax3.set_title('2a. Frequency Response (Magnitude) - Both Variants', fontsize=11)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Magnitude [dB]')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([20, fs/2])
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(freqs_np, H_lopez_phase, 'b-', linewidth=1.5, label='Lopez-Poveda 2001')
    ax4.semilogx(freqs_np, H_jepsen_phase, 'r--', linewidth=1.5, label='Jepsen 2008', alpha=0.7)
    ax4.set_title('2b. Phase Response - Both Variants', fontsize=11)
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Phase [rad]')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim([20, fs/2])
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.legend()

    # --- Row 3: Sweep Time Domain ---
    plot_samples = int(0.05 * fs)  # 0.00-0.05s
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t[:plot_samples], sweep[:plot_samples], 'k-', linewidth=0.5, 
         label='Input', alpha=0.5)
    ax5.plot(t[:plot_samples], filtered_sweep_lopez[:plot_samples], 'b-', 
         linewidth=0.8, label='Lopez-Poveda')
    ax5.set_title('3a. Sweep - Lopez-Poveda Time Domain (0.00-0.05s)', fontsize=11)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Amplitude')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t[:plot_samples], sweep[:plot_samples], 'k-', linewidth=0.5, 
         label='Input', alpha=0.5)
    ax6.plot(t[:plot_samples], filtered_sweep_jepsen[:plot_samples], 'r-', 
         linewidth=0.8, label='Jepsen')
    ax6.set_title('3b. Sweep - Jepsen Time Domain (0.00-0.05s)', fontsize=11)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Amplitude')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # --- Row 4: Spectrograms ---
    from scipy import signal as scipy_signal

    ax7 = fig.add_subplot(gs[3, 0])
    f_spec, t_spec, Sxx_lopez = scipy_signal.spectrogram(
    filtered_sweep_lopez, fs=fs, window='hann',
    nperseg=512, noverlap=256, scaling='spectrum'
    )
    Sxx_lopez_db = 10 * np.log10(Sxx_lopez + 1e-10)
    im1 = ax7.pcolormesh(t_spec, f_spec, Sxx_lopez_db, shading='gouraud', cmap='viridis')
    ax7.set_title('4a. Lopez-Poveda Filtered Sweep - Spectrogram', fontsize=11)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Frequency [Hz]')
    ax7.set_ylim([0, fs/2])
    plt.colorbar(im1, ax=ax7, label='Power [dB]')

    ax8 = fig.add_subplot(gs[3, 1])
    f_spec, t_spec, Sxx_jepsen = scipy_signal.spectrogram(
    filtered_sweep_jepsen, fs=fs, window='hann',
    nperseg=512, noverlap=256, scaling='spectrum'
    )
    Sxx_jepsen_db = 10 * np.log10(Sxx_jepsen + 1e-10)
    im2 = ax8.pcolormesh(t_spec, f_spec, Sxx_jepsen_db, shading='gouraud', cmap='viridis')
    ax8.set_title('4b. Jepsen Filtered Sweep - Spectrogram', fontsize=11)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Frequency [Hz]')
    ax8.set_ylim([0, fs/2])
    plt.colorbar(im2, ax=ax8, label='Power [dB]')

    # --- Row 5: Zoom on Phase Shift (first 3 periods) ---
    ax9 = fig.add_subplot(gs[4, 0])
    zoom_samples = int(0.05 * fs)  # 0.00-0.05s
    ax9.plot(t[:zoom_samples], sweep[:zoom_samples], 'gray', linewidth=1.5, 
         label='Input', alpha=0.7, marker='o', markersize=2, markevery=10)
    ax9.plot(t[:zoom_samples], filtered_sweep_lopez[:zoom_samples], 'b-', linewidth=1.5, 
         label='Lopez-Poveda', marker='s', markersize=2, markevery=10)

    # Add vertical lines every 10 samples
    for i in range(0, zoom_samples, 10):
        ax9.plot([t[i], t[i]], [sweep[i], filtered_sweep_lopez[i]], 
                 'cyan', linestyle='--', linewidth=0.5, alpha=0.6)

    ax9.set_title('5a. Phase Shift - Lopez-Poveda', fontsize=11)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Amplitude')
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    ax10 = fig.add_subplot(gs[4, 1])
    ax10.plot(t[:zoom_samples], sweep[:zoom_samples], 'gray', linewidth=1.5, 
          label='Input', alpha=0.7, marker='o', markersize=2, markevery=10)
    ax10.plot(t[:zoom_samples], filtered_sweep_jepsen[:zoom_samples], 'r-', linewidth=1.5, 
          label='Jepsen', marker='s', markersize=2, markevery=10)

    # Add vertical lines every 10 samples
    for i in range(0, zoom_samples, 10):
        ax10.plot([t[i], t[i]], [sweep[i], filtered_sweep_jepsen[i]], 
                  'lightcoral', linestyle='--', linewidth=0.5, alpha=0.6)

    ax10.set_title('5b. Phase Shift - Jepsen', fontsize=11)
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('Amplitude')
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = TEST_FIGURES_DIR / 'middleear_filter_analysis.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    print("="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_middleear_filter_analysis()
