"""
Combined Ear Filters - Test Suite

Contents:
1. test_combined_ear_filters_analysis: Complete pipeline testing of headphone + middle ear filtering

Structure:
- Impulse response analysis (combined vs individual filters)
- Frequency response comparison (magnitude and phase)
- Logarithmic sweep processing through complete pipeline
- Spectrogram visualization of input vs output

Figures generated:
- combined_ear_filters_analysis.png: 5-row comprehensive analysis (impulse, frequency, sweep, spectrogram, phase shift)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import HeadphoneFilter, MiddleEarFilter


def test_combined_ear_filters_analysis():
    """Test combined effect of headphone + middle ear filtering."""
    
    # Create test_figures directory
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100
    print("="*80)
    print("COMBINED EAR FILTERS TEST")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Sampling rate: {fs} Hz")

    # Create filters
    hpf = HeadphoneFilter(fs=fs, order=512, phase_type='minimum')
    mef_lopez = MiddleEarFilter(fs=fs, filter_type='lopezpoveda2001', order=512,
                                 phase_type='minimum', normalize_gain=True)

    print(f"\nFilter Pipeline:")
    print(f"  [1] Headphone Filter: {hpf}")
    print(f"  [2] Middle Ear Filter: {mef_lopez}")

    # ============================================================================
    # TEST 1: Impulse Response (Combined)
    # ============================================================================
    print(f"\n[1] Combined Impulse Response")
    impulse_duration = 0.5  # seconds
    impulse = torch.zeros(int(fs * impulse_duration), dtype=torch.float32)
    impulse[0] = 1.0
    impulse = impulse.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        # Individual responses
        impulse_hp = hpf(impulse).squeeze(0).numpy()
        impulse_me = mef_lopez(impulse).squeeze(0).numpy()
        
        # Combined response
        impulse_combined = mef_lopez(hpf(impulse)).squeeze(0).numpy()

    print(f"  Impulse input: {impulse.shape}")
    print(f"  Headphone only: {impulse_hp.shape}")
    print(f"  Middle ear only: {impulse_me.shape}")
    print(f"  Combined: {impulse_combined.shape}")

    # ============================================================================
    # TEST 2: Combined Frequency Response
    # ============================================================================
    print(f"\n[2] Combined Frequency Response")
    freqs_hp, H_hp = hpf.get_frequency_response(nfft=8192)
    freqs_me, H_me = mef_lopez.get_frequency_response(nfft=8192)

    # Combined response (multiply in frequency domain)
    H_combined = H_hp * H_me

    freqs_np = freqs_hp.cpu().numpy()
    H_hp_db = 20 * torch.log10(torch.abs(H_hp) + 1e-10).cpu().numpy()
    H_me_db = 20 * torch.log10(torch.abs(H_me) + 1e-10).cpu().numpy()
    H_combined_db = 20 * torch.log10(torch.abs(H_combined) + 1e-10).cpu().numpy()

    print(f"  Headphone magnitude range: {H_hp_db.min():.2f} to {H_hp_db.max():.2f} dB")
    print(f"  Middle ear magnitude range: {H_me_db.min():.2f} to {H_me_db.max():.2f} dB")
    print(f"  Combined magnitude range: {H_combined_db.min():.2f} to {H_combined_db.max():.2f} dB")

    # ============================================================================
    # TEST 3: Logarithmic Sweep (Combined Pipeline)
    # ============================================================================
    print(f"\n[3] Combined Pipeline - Logarithmic Sweep")
    sweep_duration = 5.0  # seconds
    t = np.linspace(0, sweep_duration, int(fs * sweep_duration), endpoint=False)

    # Logarithmic sweep from 20 Hz to fs/2
    f0, f1 = 20, fs / 2
    k = (f1 - f0) / sweep_duration
    sweep = np.sin(2 * np.pi * (f0 * t + (k / 2) * t**2))

    sweep_tensor = torch.tensor(sweep, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # Stage by stage
        after_hp = hpf(sweep_tensor)
        after_me = mef_lopez(after_hp)
        
        sweep_hp = after_hp.squeeze(0).numpy()
        sweep_me = mef_lopez(sweep_tensor).squeeze(0).numpy()
        sweep_combined = after_me.squeeze(0).numpy()

    print(f"  Sweep: {f0} Hz → {f1:.0f} Hz over {sweep_duration} s")
    print(f"  Input → Headphone → Middle Ear")
    print(f"  {sweep_tensor.shape} → {after_hp.shape} → {after_me.shape}")

    # ============================================================================
    # PLOTTING
    # ============================================================================
    print(f"\nGenerating plots...")

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('Combined Ear Filters Analysis (Headphone + Middle Ear)', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Create 5x2 grid (added row for zoom)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

    # --- Row 1: Impulse Responses Comparison ---
    ax1 = fig.add_subplot(gs[0, :])
    t_impulse = np.arange(1000) / fs * 1000  # first 1000 samples in ms
    ax1.plot(t_impulse, impulse_hp[:1000], 'b-', linewidth=1, label='Headphone only', alpha=0.7)
    ax1.plot(t_impulse, impulse_me[:1000], 'g-', linewidth=1, label='Middle ear only', alpha=0.7)
    ax1.plot(t_impulse, impulse_combined[:1000], 'r-', linewidth=1.5, label='Combined', alpha=0.9)
    ax1.set_title('1. Impulse Responses Comparison (first 1000 samples)', fontsize=12)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.legend()

    # --- Row 2: Frequency Response Magnitude ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogx(freqs_np, H_hp_db, 'b-', linewidth=1.5, label='Headphone', alpha=0.7)
    ax2.semilogx(freqs_np, H_me_db, 'g-', linewidth=1.5, label='Middle ear', alpha=0.7)
    ax2.semilogx(freqs_np, H_combined_db, 'r-', linewidth=2, label='Combined')
    ax2.set_title('2a. Frequency Response (Magnitude)', fontsize=11)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude [dB]')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([20, fs/2])
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend()
# --- Row 2: Frequency Response Phase ---
    H_hp_phase = torch.angle(H_hp).cpu().numpy()
    H_me_phase = torch.angle(H_me).cpu().numpy()
    H_combined_phase = torch.angle(H_combined).cpu().numpy()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogx(freqs_np, H_hp_phase, 'b-', linewidth=1.5, label='Headphone', alpha=0.7)
    ax3.semilogx(freqs_np, H_me_phase, 'g-', linewidth=1.5, label='Middle ear', alpha=0.7)
    ax3.semilogx(freqs_np, H_combined_phase, 'r-', linewidth=2, label='Combined')
    ax3.set_title('2b. Phase Response', fontsize=11)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Phase [rad]')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([20, fs/2])
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend()

# --- Row 3: Sweep Time Domain ---
    plot_samples = 2000
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t[:plot_samples], sweep[:plot_samples], 'k-', linewidth=0.5, 
         label='Input', alpha=0.5)
    ax4.plot(t[:plot_samples], sweep_hp[:plot_samples], 'b-', linewidth=0.7, 
         label='Post Headphone', alpha=0.7)
    ax4.plot(t[:plot_samples], sweep_combined[:plot_samples], 'r-', linewidth=0.9, 
         label='Combined Output')
    ax4.set_title('3a. Pipeline Sweep - Time Domain (0.00-0.05s)', fontsize=11)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 1])
# Normalized comparison
    sweep_norm = sweep / (np.abs(sweep).max() + 1e-10)
    sweep_hp_norm = sweep_hp / (np.abs(sweep_hp).max() + 1e-10)
    sweep_combined_norm = sweep_combined / (np.abs(sweep_combined).max() + 1e-10)

    ax5.plot(t[:plot_samples], sweep_norm[:plot_samples], 'k-', linewidth=0.5, 
         label='Input (norm)', alpha=0.5)
    ax5.plot(t[:plot_samples], sweep_hp_norm[:plot_samples], 'b-', linewidth=0.7, 
         label='Post Headphone (norm)', alpha=0.7)
    ax5.plot(t[:plot_samples], sweep_combined_norm[:plot_samples], 'r-', linewidth=0.9, 
         label='Combined (norm)')
    ax5.set_title('3b. Pipeline Sweep - Normalized (0.00-0.05s)', fontsize=11)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Normalized Amplitude')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

# --- Row 4: Spectrograms ---
    from scipy import signal as scipy_signal

    ax6 = fig.add_subplot(gs[3, 0])
    f_spec, t_spec, Sxx_input = scipy_signal.spectrogram(
    sweep, fs=fs, window='hann',
    nperseg=512, noverlap=256, scaling='spectrum'
    )
    Sxx_input_db = 10 * np.log10(Sxx_input + 1e-10)
    im1 = ax6.pcolormesh(t_spec, f_spec, Sxx_input_db, shading='gouraud', cmap='viridis')
    ax6.set_title('4a. Input Sweep - Spectrogram', fontsize=11)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Frequency [Hz]')
    ax6.set_ylim([0, fs/2])
    plt.colorbar(im1, ax=ax6, label='Power [dB]')

    ax7 = fig.add_subplot(gs[3, 1])
    f_spec, t_spec, Sxx_combined = scipy_signal.spectrogram(
    sweep_combined, fs=fs, window='hann',
    nperseg=512, noverlap=256, scaling='spectrum'
    )
    Sxx_combined_db = 10 * np.log10(Sxx_combined + 1e-10)
    im2 = ax7.pcolormesh(t_spec, f_spec, Sxx_combined_db, shading='gouraud', cmap='viridis')
    ax7.set_title('4b. Combined Output - Spectrogram', fontsize=11)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Frequency [Hz]')
    ax7.set_ylim([0, fs/2])
    plt.colorbar(im2, ax=ax7, label='Power [dB]')

# --- Row 5: Zoom on Phase Shift (first 3 periods) ---
    ax8 = fig.add_subplot(gs[4, :])
    zoom_samples = int(0.08 * fs)  # First 5 cycles (~0.08s)
    ax8.plot(t[:zoom_samples], sweep[:zoom_samples], 'gray', linewidth=1.5, 
         label='Input', alpha=0.7, marker='o', markersize=2, markevery=10)
    ax8.plot(t[:zoom_samples], sweep_hp[:zoom_samples], 'b-', linewidth=1.5, 
         label='Post Headphone', marker='s', markersize=2, markevery=10, alpha=0.7)
    ax8.plot(t[:zoom_samples], sweep_combined[:zoom_samples], 'r-', linewidth=1.5, 
         label='Combined', marker='^', markersize=2, markevery=10)

    # Add cyan lines from input to post headphone
    for i in range(0, zoom_samples, 10):
        ax8.plot([t[i], t[i]], [sweep[i], sweep_hp[i]], 
                 'cyan', linestyle='--', linewidth=0.5, alpha=0.6)

    # Add red faded lines from input to combined
    for i in range(0, zoom_samples, 10):
        ax8.plot([t[i], t[i]], [sweep[i], sweep_combined[i]], 
                 'red', linestyle='--', linewidth=0.5, alpha=0.3)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Amplitude')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = TEST_FIGURES_DIR / 'combined_ear_filters_analysis.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    print("="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_combined_ear_filters_analysis()
