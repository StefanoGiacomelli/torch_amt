"""
Headphone Filter - Test Suite

Contents:
1. test_headphone_filter_analysis: Complete headphone filter analysis with impulse, frequency response, and sweep tests

Structure:
- Impulse response analysis
- Frequency response (magnitude and phase)
- Logarithmic sweep processing (20 Hz to fs/2)
- Spectrogram visualization
- Phase shift visualization

Figures generated:
- headphone_filter_analysis.png: 6-panel comprehensive analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import HeadphoneFilter


def test_headphone_filter_analysis():
    """Test headphone filter with impulse response, frequency response and sweep."""
    
    # Create test_figures directory
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100
    print("="*80)
    print("HEADPHONE FILTER TEST")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Sampling rate: {fs} Hz")

    # Create filter
    hpf = HeadphoneFilter(fs=fs, order=512, phase_type='minimum')
    print(f"\nHeadphoneFilter configuration:")
    print(f"  {hpf}")

    # ============================================================================
    # TEST 1: Impulse Response
    # ============================================================================
    print(f"\n[1] Impulse Response Test")
    impulse_duration = 0.5  # seconds
    impulse = torch.zeros(int(fs * impulse_duration), dtype=torch.float32)
    impulse[0] = 1.0
    impulse = impulse.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        impulse_response = hpf(impulse).squeeze(0).numpy()

    print(f"  Impulse input: {impulse.shape}")
    print(f"  Impulse response: {impulse_response.shape}")

    # ============================================================================
    # TEST 2: Frequency Response
    # ============================================================================
    print(f"\n[2] Frequency Response Analysis")
    freqs, H = hpf.get_frequency_response(nfft=8192)
    freqs_np = freqs.cpu().numpy()
    H_magnitude_db = 20 * torch.log10(torch.abs(H) + 1e-10).cpu().numpy()
    H_phase = torch.angle(H).cpu().numpy()

    print(f"  Frequency vector: {freqs_np.shape}")
    print(f"  Response range: {H_magnitude_db.min():.2f} to {H_magnitude_db.max():.2f} dB")

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
        filtered_sweep = hpf(sweep_tensor).squeeze(0).numpy()

    print(f"  Sweep: {f0} Hz → {f1:.0f} Hz over {sweep_duration} s")
    print(f"  Input shape: {sweep_tensor.shape}")
    print(f"  Output shape: {filtered_sweep.shape}")

    # ============================================================================
    # PLOTTING
    # ============================================================================
    print(f"\nGenerating plots...")

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Headphone Filter Analysis (Pralong & Carlile 1996)', 
             fontsize=16, fontweight='bold', y=0.98)

    # Create 4x2 grid
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # --- Row 1: Impulse Response ---
    ax1 = fig.add_subplot(gs[0, :])
    t_impulse = np.arange(len(impulse_response)) / fs * 1000  # ms
    ax1.plot(t_impulse[:1000], impulse_response[:1000], 'b-', linewidth=1)
    ax1.set_title('1. Impulse Response (first 1000 samples)', fontsize=12)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # --- Row 2: Frequency Response ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogx(freqs_np, H_magnitude_db, 'b-', linewidth=1.5)
    ax2.set_title('2. Frequency Response (Magnitude)', fontsize=11)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude [dB]')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([20, fs/2])
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='0 dB')
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogx(freqs_np, H_phase, 'g-', linewidth=1.5)
    ax3.set_title('3. Phase Response', fontsize=11)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Phase [rad]')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([20, fs/2])
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='0 rad')
    ax3.legend()

    # --- Row 3: Sweep Test ---
    ax4 = fig.add_subplot(gs[2, 0])
    plot_samples = int(0.05 * fs)  # 0.00-0.05s
    ax4.plot(t[:plot_samples], sweep[:plot_samples], 'b-', linewidth=0.5, label='Input', alpha=0.7)
    ax4.plot(t[:plot_samples], filtered_sweep[:plot_samples], 'r-', linewidth=0.5, label='Filtered')
    ax4.set_title('4. Logarithmic Sweep - Time Domain (0.00-0.05s)', fontsize=11)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 1])
    # Spectrogram of filtered sweep
    from scipy import signal as scipy_signal
    f_spec, t_spec, Sxx = scipy_signal.spectrogram(
    filtered_sweep, fs=fs, window='hann', 
    nperseg=512, noverlap=256, scaling='spectrum'
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax5.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='viridis')
    ax5.set_title('5. Filtered Sweep - Spectrogram', fontsize=11)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Frequency [Hz]')
    ax5.set_ylim([0, fs/2])
    plt.colorbar(im, ax=ax5, label='Power [dB]')

    # --- Row 5: Zoom on Phase Shift (first 3 periods) ---
    ax6 = fig.add_subplot(gs[3, :])
    # First 5 cycles of the logarithmic sweep (~0.08s = 1280 samples @ 16kHz)
    zoom_samples = int(0.05 * fs)  # 0.00-0.05s
    ax6.plot(t[:zoom_samples], sweep[:zoom_samples], 'gray', linewidth=1.5, 
         label='Input', alpha=0.7, marker='o', markersize=2, markevery=10)
    ax6.plot(t[:zoom_samples], filtered_sweep[:zoom_samples], 'b-', linewidth=1.5, 
         label='Filtered', marker='s', markersize=2, markevery=10)

    # Add vertical lines every 10 samples to show phase shift
    for i in range(0, zoom_samples, 10):
        ax6.plot([t[i], t[i]], [sweep[i], filtered_sweep[i]], 
                 'yellow', linestyle='--', linewidth=0.5, alpha=0.6)

    ax6.set_title('6. Phase Shift Visualization', fontsize=11)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Amplitude')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'headphone_filter_analysis.png'
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    print("="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_headphone_filter_analysis()
