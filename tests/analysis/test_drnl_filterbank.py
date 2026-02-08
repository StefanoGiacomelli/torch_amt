"""
DRNL Filterbank - Test Suite

Contents:
1. test_drnl_linear_path: Tests and visualizes linear path (gain → gammatone → lowpass)
2. test_drnl_nonlinear_path: Tests and visualizes nonlinear path (gammatone → compression → lowpass)
3. test_drnl_input_output: Compares input/output response at different SPL levels

Structure:
- Dual-resonance nonlinear (DRNL) filterbank implementation
- Stage-by-stage signal processing visualization
- Filter frequency response analysis
- Level-dependent compression behavior

Figures generated:
- drnl_linear_path.png: Linear path analysis (9-panel grid)
- drnl_nonlinear_path.png: Nonlinear path analysis (11-panel grid)
- drnl_input_output.png: Input/output comparison (7-panel grid)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from torch_amt.common import DRNLFilterbank


def test_drnl_linear_path():
    """Test and visualize the linear path of DRNL filterbank."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100  # Hz
    test_freq = 1000  # Hz
    test_level = 60  # dB SPL
    dur = 0.05  # seconds
    
    print("="*80)
    print("DRNL LINEAR PATH TEST")
    print("="*80)
    print(f"\nTest signal: {test_freq} Hz tone @ {test_level} dB SPL")
    print(f"Duration: {dur*1000:.1f} ms")
    print(f"Sampling rate: {fs} Hz")
    
    # Create DRNL filterbank
    drnl = DRNLFilterbank((250, 8000), fs=fs, n_channels=50, dtype=torch.float64)
    
    # Find on-frequency channel
    fc_diff = torch.abs(drnl.fc - test_freq)
    on_ch = torch.argmin(fc_diff).item()
    print(f"\nOn-frequency channel: {on_ch} (fc = {drnl.fc[on_ch]:.1f} Hz)")
    
    # Generate input signal
    t = torch.arange(0, dur, 1/fs, dtype=torch.float64)
    n_samples = len(t)
    tone = torch.sin(2 * torch.pi * test_freq * t)
    
    # Scale to SPL
    rms_target = 20e-6 * 10.0 ** (test_level / 20.0)
    rms_current = torch.sqrt(torch.mean(tone ** 2))
    tone = tone * (rms_target / rms_current)
    
    # Convert to numpy for stage-by-stage processing
    x = tone.numpy()
    
    # ========== Process LINEAR PATH stage by stage ==========
    print(f"\nProcessing linear path...")
    
    # Stage 1: Apply gain
    g_val = drnl.g[on_ch].item()
    y_gain = x * g_val
    print(f"  Stage 1 - Gain: g = {g_val:.2e}")
    print(f"    Input range: [{x.min():.4e}, {x.max():.4e}]")
    print(f"    After gain: [{y_gain.min():.4e}, {y_gain.max():.4e}]")
    
    # Stage 2: Gammatone filtering
    GT_lin_b, GT_lin_a = drnl.GT_lin_coeffs[on_ch]
    y_gt = scipy_signal.lfilter(GT_lin_b, GT_lin_a, y_gain)
    y_gt = np.real(y_gt)
    print(f"  Stage 2 - Gammatone BP filter (n={drnl.n_gt_lin}):")
    print(f"    CF = {drnl.CF_lin[on_ch]:.1f} Hz")
    print(f"    BW_norm = {drnl.BW_lin_norm[on_ch]:.4f}")
    print(f"    After GT: [{y_gt.min():.4e}, {y_gt.max():.4e}]")
    
    # Stage 3: Lowpass filtering (cascaded)
    y_lp = y_gt.copy()
    LP_lin_b, LP_lin_a = drnl.LP_lin_coeffs[on_ch]
    for i in range(drnl.n_lp_lin):
        y_lp = scipy_signal.lfilter(LP_lin_b, LP_lin_a, y_lp)
    print(f"  Stage 3 - Lowpass filter (n={drnl.n_lp_lin} cascaded):")
    print(f"    Cutoff = {drnl.LP_lin_cutoff[on_ch]:.1f} Hz")
    print(f"    After LP: [{y_lp.min():.4e}, {y_lp.max():.4e}]")
    
    y_lin_final = y_lp
    
    # Compute impulse responses for filters
    print(f"\nComputing filter characteristics...")
    impulse_len = 4096
    impulse = np.zeros(impulse_len)
    impulse[0] = 1.0
    
    # GT impulse response (all channels)
    gt_ir_all = []
    for ch in range(drnl.num_channels):
        GT_b, GT_a = drnl.GT_lin_coeffs[ch]
        ir = scipy_signal.lfilter(GT_b, GT_a, impulse)
        gt_ir_all.append(np.real(ir))
    gt_ir_all = np.array(gt_ir_all)  # [channels, time]
    
    # LP impulse response (on-channel only)
    lp_ir = impulse.copy()
    for i in range(drnl.n_lp_lin):
        lp_ir = scipy_signal.lfilter(LP_lin_b, LP_lin_a, lp_ir)
    
    # Frequency responses
    nfft = 8192
    gt_fr_all = np.fft.rfft(gt_ir_all, n=nfft, axis=1)
    gt_fr_mag = 20 * np.log10(np.abs(gt_fr_all) + 1e-10)
    
    lp_fr = np.fft.rfft(lp_ir, n=nfft)
    lp_fr_mag = 20 * np.log10(np.abs(lp_fr) + 1e-10)
    
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    
    # ========== PLOT ==========
    print(f"\nGenerating figure...")
    
    fig = plt.figure(figsize=(14, 12))
    t_ms = t.numpy() * 1000
    
    # Row 1: Input waveform + spectrogram, with overlaid gain output
    ax1 = plt.subplot(5, 2, 1)
    ax1.plot(t_ms, x, label='Input', color='steelblue', linewidth=1)
    ax1.plot(t_ms, y_gain / g_val, label='Post gain (scaled)', color='orange', alpha=0.7, linewidth=0.8)
    ax1.axhline(x.max(), color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axhline(x.min(), color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (Pa)')
    ax1.set_title(f'Input: {test_freq} Hz @ {test_level} dB SPL')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 10])
    
    # Row 1, col 2: Input spectrogram
    ax2 = plt.subplot(5, 2, 2)
    f, t_spec, Sxx = scipy_signal.spectrogram(x, fs=fs, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax2.pcolormesh(t_spec*1000, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Input Spectrogram')
    ax2.set_ylim([0, 3000])
    plt.colorbar(im, ax=ax2, label='Power (dB)')
    
    # Row 2: GT impulse responses (all channels)
    ax3 = plt.subplot(5, 2, 3)
    t_ir = np.arange(impulse_len) / fs * 1000
    t_max = 10  # ms
    t_mask = t_ir <= t_max
    colors = plt.cm.viridis(np.linspace(0, 1, drnl.num_channels))
    # Plot all channels thin
    for ch in range(drnl.num_channels):
        ax3.plot(t_ir[t_mask], gt_ir_all[ch, t_mask], alpha=0.2, linewidth=0.3, color=colors[ch])
    # Highlight ONLY on-frequency channel
    ax3.plot(t_ir[t_mask], gt_ir_all[on_ch, t_mask], alpha=1.0, linewidth=2.0,
            color=colors[on_ch], label=f'{drnl.fc[on_ch]:.0f} Hz (on-freq)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Gammatone Impulse Responses')
    ax3.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.grid(True, alpha=0.3)
    
    # Row 2, col 2: GT frequency responses
    ax4 = plt.subplot(5, 2, 4)
    # Plot all channels thin
    for ch in range(drnl.num_channels):
        ax4.plot(freqs, gt_fr_mag[ch, :], alpha=0.2, linewidth=0.3, color=colors[ch])
    # Highlight ONLY on-frequency channel
    ax4.plot(freqs, gt_fr_mag[on_ch, :], alpha=1.0, linewidth=2.0, color=colors[on_ch])
    ax4.axvline(test_freq, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.set_title('Gammatone Frequency Responses')
    ax4.set_xlim([100, 5000])
    ax4.set_xscale('log')
    ax4.set_ylim([-80, 10])
    ax4.grid(True, alpha=0.3, which='both')
    
    # Row 3: LP impulse response
    ax5 = plt.subplot(5, 2, 5)
    ax5.plot(t_ir[t_mask], lp_ir[t_mask], color='darkgreen', linewidth=1)
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title(f'Lowpass Impulse Response (n={drnl.n_lp_lin} cascaded)')
    ax5.grid(True, alpha=0.3)
    
    # Row 3, col 2: LP frequency response
    ax6 = plt.subplot(5, 2, 6)
    ax6.plot(freqs, lp_fr_mag, color='darkgreen', linewidth=1.5)
    ax6.axvline(drnl.LP_lin_cutoff[on_ch].item(), color='red', linestyle='--', 
               linewidth=1, label=f'Cutoff: {drnl.LP_lin_cutoff[on_ch]:.0f} Hz')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude (dB)')
    ax6.set_title('Lowpass Frequency Response')
    ax6.set_xlim([100, 5000])
    ax6.set_xscale('log')
    ax6.set_ylim([-80, 10])
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, which='both')
    
    # Row 4: Output waveform (stages comparison)
    ax7 = plt.subplot(5, 2, 7)
    ax7.plot(t_ms, y_gain / g_val, label='Post gain', alpha=0.5, linewidth=0.8)
    ax7.plot(t_ms, y_gt / g_val, label='Post GT', alpha=0.7, linewidth=0.8)
    ax7.plot(t_ms, y_lin_final / g_val, label='Post LP (final)', linewidth=1)
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Amplitude (normalized)')
    ax7.set_title('Linear Path: Stage-by-stage output')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 10])
    
    # Row 4, col 2: Final output spectrum
    ax8 = plt.subplot(5, 2, 8)
    f_out, Pxx = scipy_signal.welch(y_lin_final, fs=fs, nperseg=1024)
    Pxx_db = 10 * np.log10(Pxx + 1e-10)
    ax8.plot(f_out, Pxx_db, color='darkblue', linewidth=1.5)
    ax8.axvline(test_freq, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax8.set_xlabel('Frequency (Hz)')
    ax8.set_ylabel('Power (dB)')
    ax8.set_title('Linear Path Output: Power Spectrum')
    ax8.set_xlim([0, 3000])
    ax8.grid(True, alpha=0.3)
    
    # Row 5: Output spectrogram (span 2 columns)
    ax9 = plt.subplot(5, 1, 5)
    f, t_spec, Sxx = scipy_signal.spectrogram(y_lin_final, fs=fs, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax9.pcolormesh(t_spec*1000, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax9.set_xlabel('Time (ms)')
    ax9.set_ylabel('Frequency (Hz)')
    ax9.set_title(f'Linear Path Output Spectrogram (Ch{on_ch}: {drnl.fc[on_ch]:.1f} Hz)')
    ax9.set_ylim([0, 3000])
    plt.colorbar(im, ax=ax9, label='Power (dB)')
    
    plt.suptitle('DRNL Filterbank: Linear Path Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = TEST_FIGURES_DIR / 'drnl_linear_path.png'
    plt.savefig(output_file, dpi=600, format='png', bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    print("\n" + "="*80)
    print("LINEAR PATH TEST COMPLETE")
    print("="*80)


def test_drnl_nonlinear_path():
    """Test and visualize the nonlinear path of DRNL filterbank with all stages."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100  # Hz
    test_freq = 1000  # Hz
    test_level = 60  # dB SPL
    dur = 0.05  # seconds
    impulse_len = 4096
    
    print("="*80)
    print("DRNL NONLINEAR PATH TEST")
    print("="*80)
    print(f"\nTest signal: {test_freq} Hz tone @ {test_level} dB SPL")
    print(f"Duration: {dur*1000:.1f} ms")
    print(f"Sampling rate: {fs} Hz")
    
    # Create DRNL filterbank
    drnl = DRNLFilterbank((250, 8000), fs=fs, n_channels=50, dtype=torch.float64)
    
    # Find on-frequency channel
    fc_diff = torch.abs(drnl.fc - test_freq)
    on_ch = torch.argmin(fc_diff).item()
    print(f"\nOn-frequency channel: {on_ch} (fc = {drnl.fc[on_ch]:.1f} Hz)")
    
    # Generate input signal
    t = torch.arange(0, dur, 1/fs, dtype=torch.float64)
    n_samples = len(t)
    tone = torch.sin(2 * torch.pi * test_freq * t)
    
    # Scale to SPL
    rms_target = 20e-6 * 10.0 ** (test_level / 20.0)
    rms_current = torch.sqrt(torch.mean(tone ** 2))
    tone = tone * (rms_target / rms_current)
    
    # Convert to numpy for stage-by-stage processing
    x = tone.numpy()
    
    # ========== Process NONLINEAR PATH stage by stage ==========
    print(f"\nProcessing nonlinear path...")
    
    # Stage 1: First Gammatone filtering
    GT_nlin_b, GT_nlin_a = drnl.GT_nlin_coeffs[on_ch]
    y_gt1 = scipy_signal.lfilter(GT_nlin_b, GT_nlin_a, x)
    y_gt1 = np.real(y_gt1)
    print(f"  Stage 1 - Gammatone BP filter (n={drnl.n_gt_nlin}):")
    print(f"    CF = {drnl.CF_nlin[on_ch]:.1f} Hz")
    print(f"    BW_norm = {drnl.BW_nlin_norm[on_ch]:.4f}")
    print(f"    After GT1: [{y_gt1.min():.4e}, {y_gt1.max():.4e}]")
    
    # Stage 2: Broken-stick nonlinearity
    a_val = drnl.a[on_ch].item()
    b_val = drnl.b[on_ch].item()
    c_val = drnl.c[on_ch].item()
    
    y_abs = np.abs(y_gt1)
    y_decide = np.vstack([a_val * y_abs, b_val * (y_abs ** c_val)])
    y_nonlin = np.sign(y_gt1) * np.min(y_decide, axis=0)
    print(f"  Stage 2 - Broken-stick nonlinearity:")
    print(f"    a = {a_val:.4f}, b = {b_val:.4f}, c = {c_val:.4f}")
    print(f"    After nonlinearity: [{y_nonlin.min():.4e}, {y_nonlin.max():.4e}]")
    
    # Stage 3: Second Gammatone filtering
    y_gt2 = scipy_signal.lfilter(GT_nlin_b, GT_nlin_a, y_nonlin)
    y_gt2 = np.real(y_gt2)
    print(f"  Stage 3 - Gammatone BP filter (same params)")
    print(f"    After GT2: [{y_gt2.min():.4e}, {y_gt2.max():.4e}]")
    
    # Stage 4: Lowpass filtering
    y_lp = y_gt2.copy()
    LP_nlin_b, LP_nlin_a = drnl.LP_nlin_coeffs[on_ch]
    for i in range(drnl.n_lp_nlin):
        y_lp = scipy_signal.lfilter(LP_nlin_b, LP_nlin_a, y_lp)
    print(f"  Stage 4 - Lowpass filter (n={drnl.n_lp_nlin}):")
    print(f"    Cutoff = {drnl.LP_nlin_cutoff[on_ch]:.1f} Hz")
    print(f"    After LP: [{y_lp.min():.4e}, {y_lp.max():.4e}]")
    
    y_nlin_final = y_lp
    
    # Compute broken-stick transfer function with appropriate range
    print(f"\nComputing broken-stick transfer characteristic...")
    # Use range that shows both linear and compressed regions
    # Crossover at: a*|x| = b*|x|^c => |x| = (b/a)^(1/(1-c))
    x_cross = (b_val / a_val) ** (1 / (1 - c_val))
    print(f"  Crossover point: {x_cross:.4e}")
    x_range = np.linspace(-10*x_cross, 10*x_cross, 2000)
    y_linear = a_val * np.abs(x_range)
    y_compressed = b_val * (np.abs(x_range) ** c_val)
    y_transfer = np.sign(x_range) * np.minimum(y_linear, y_compressed)
    
    # Compute GT2 impulse/frequency responses
    gt2_ir_all = []
    for ch in range(drnl.num_channels):
        GT_b, GT_a = drnl.GT_nlin_coeffs[ch]
        impulse = np.zeros(impulse_len)
        impulse[0] = 1.0
        ir = scipy_signal.lfilter(GT_b, GT_a, impulse)
        gt2_ir_all.append(np.real(ir))
    gt2_ir_all = np.array(gt2_ir_all)
    
    nfft = 8192
    gt2_fr_all = np.fft.rfft(gt2_ir_all, n=nfft, axis=1)
    gt2_fr_mag = 20 * np.log10(np.abs(gt2_fr_all) + 1e-10)
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    colors = plt.cm.viridis(np.linspace(0, 1, drnl.num_channels))
    
    # ========== PLOT ==========
    print(f"\nGenerating figure...")
    
    fig = plt.figure(figsize=(14, 15))
    t_ms = t.numpy() * 1000
    t_ir = np.arange(impulse_len) / fs * 1000
    t_max_ir = 10  # ms
    t_mask = t_ir <= t_max_ir
    
    # Row 1: Input waveform + spectrogram
    ax1 = plt.subplot(6, 2, 1)
    ax1.plot(t_ms, x, label='Input', color='steelblue', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (Pa)')
    ax1.set_title(f'Input: {test_freq} Hz @ {test_level} dB SPL')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 10])
    
    ax2 = plt.subplot(6, 2, 2)
    f, t_spec, Sxx = scipy_signal.spectrogram(x, fs=fs, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax2.pcolormesh(t_spec*1000, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Input Spectrogram')
    ax2.set_ylim([0, 3000])
    plt.colorbar(im, ax=ax2, label='Power (dB)')
    
    # Row 2: GT Stage 1 impulse responses (all channels)
    ax3 = plt.subplot(6, 2, 3)
    # Plot all channels thin
    for ch in range(drnl.num_channels):
        ax3.plot(t_ir[t_mask], gt2_ir_all[ch, t_mask], alpha=0.2, linewidth=0.3, color=colors[ch])
    # Highlight ONLY on-frequency channel
    ax3.plot(t_ir[t_mask], gt2_ir_all[on_ch, t_mask], alpha=1.0, linewidth=2.0,
            color=colors[on_ch], label=f'{drnl.fc[on_ch]:.0f} Hz (on-freq)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Gammatone Stage 1 Impulse Responses')
    ax3.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.grid(True, alpha=0.3)
    
    # Row 2, col 2: GT Stage 1 frequency responses
    ax4 = plt.subplot(6, 2, 4)
    # Plot all channels thin
    for ch in range(drnl.num_channels):
        ax4.plot(freqs, gt2_fr_mag[ch, :], alpha=0.2, linewidth=0.3, color=colors[ch])
    # Highlight ONLY on-frequency channel
    ax4.plot(freqs, gt2_fr_mag[on_ch, :], alpha=1.0, linewidth=2.0, color=colors[on_ch])
    ax4.axvline(drnl.CF_nlin[on_ch].item(), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.set_title('Gammatone Stage 1 Frequency Responses')
    ax4.set_xlim([100, 5000])
    ax4.set_xscale('log')
    ax4.set_ylim([-80, 10])
    ax4.grid(True, alpha=0.3, which='both')
    
    # Row 3: Broken-stick nonlinearity
    ax5 = plt.subplot(6, 2, 5)
    ax5.plot(x_range, y_transfer, color='darkred', linewidth=2.5, label='Broken-stick', zorder=3)
    ax5.plot(x_range, y_linear, '--', color='blue', alpha=0.6, linewidth=1.5, label=f'Linear (a={a_val:.1f})', zorder=1)
    ax5.plot(x_range, y_compressed, '--', color='green', alpha=0.6, linewidth=1.5, label=f'Compressed (c={c_val:.2f})', zorder=2)
    ax5.axvline(x_cross, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax5.axvline(-x_cross, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Input (Pa)')
    ax5.set_ylabel('Output (Pa)')
    ax5.set_title('Broken-stick Nonlinearity Transfer Function')
    ax5.set_xlim([-2e-6, 2e-6])
    ax5.set_ylim([-0.02, 0.02])
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Row 3, col 2: Post nonlinearity waveform
    ax6 = plt.subplot(6, 2, 6)
    ax6.plot(t_ms, y_gt1, label='Pre nonlin', alpha=0.5, linewidth=0.8)
    ax6.plot(t_ms, y_nonlin, label='Post nonlin', linewidth=1)
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Amplitude')
    ax6.set_title('Broken-stick Nonlinearity Effect')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 10])
    
    # Row 4: GT Stage 2 impulse/frequency responses
    ax7 = plt.subplot(6, 2, 7)
    # Plot all channels thin
    for ch in range(drnl.num_channels):
        ax7.plot(t_ir[t_mask], gt2_ir_all[ch, t_mask], alpha=0.2, linewidth=0.3, color=colors[ch])
    # Highlight ONLY on-frequency channel
    ax7.plot(t_ir[t_mask], gt2_ir_all[on_ch, t_mask], alpha=1.0, linewidth=2.0,
            color=colors[on_ch], label=f'{drnl.fc[on_ch]:.0f} Hz (on-freq)')
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Amplitude')
    ax7.set_title('Gammatone Stage 2 Impulse Responses')
    ax7.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(6, 2, 8)
    # Plot all channels thin
    for ch in range(drnl.num_channels):
        ax8.plot(freqs, gt2_fr_mag[ch, :], alpha=0.2, linewidth=0.3, color=colors[ch])
    # Highlight ONLY on-frequency channel
    ax8.plot(freqs, gt2_fr_mag[on_ch, :], alpha=1.0, linewidth=2.0, color=colors[on_ch])
    ax8.axvline(drnl.CF_nlin[on_ch].item(), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax8.set_xlabel('Frequency (Hz)')
    ax8.set_ylabel('Magnitude (dB)')
    ax8.set_title('Gammatone Stage 2 Frequency Responses')
    ax8.set_xlim([100, 5000])
    ax8.set_xscale('log')
    ax8.set_ylim([-80, 10])
    ax8.grid(True, alpha=0.3, which='both')
    
    # Row 5: LP filter + final output
    impulse_lp = np.zeros(impulse_len)
    impulse_lp[0] = 1.0
    lp_ir = impulse_lp.copy()
    for i in range(drnl.n_lp_nlin):
        lp_ir = scipy_signal.lfilter(LP_nlin_b, LP_nlin_a, lp_ir)
    lp_fr = np.fft.rfft(lp_ir, n=nfft)
    lp_fr_mag = 20 * np.log10(np.abs(lp_fr) + 1e-10)
    
    ax9 = plt.subplot(6, 2, 9)
    ax9.plot(t_ir[t_mask], lp_ir[t_mask], color='darkgreen', linewidth=1)
    ax9.set_xlabel('Time (ms)')
    ax9.set_ylabel('Amplitude')
    ax9.set_title(f'Lowpass Impulse Response (n={drnl.n_lp_nlin})')
    ax9.grid(True, alpha=0.3)
    
    ax10 = plt.subplot(6, 2, 10)
    ax10.plot(freqs, lp_fr_mag, color='darkgreen', linewidth=1.5)
    ax10.axvline(drnl.LP_nlin_cutoff[on_ch].item(), color='red', linestyle='--', 
               linewidth=1, label=f'Cutoff: {drnl.LP_nlin_cutoff[on_ch]:.0f} Hz')
    ax10.set_xlabel('Frequency (Hz)')
    ax10.set_ylabel('Magnitude (dB)')
    ax10.set_title(f'Lowpass Frequency Response (n={drnl.n_lp_nlin})')
    ax10.set_xlim([100, 5000])
    ax10.set_xscale('log')
    ax10.set_ylim([-80, 10])
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3, which='both')
    
    # Row 6: Output spectrogram (span 2 columns)
    ax11 = plt.subplot(6, 1, 6)
    f, t_spec, Sxx = scipy_signal.spectrogram(y_nlin_final, fs=fs, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax11.pcolormesh(t_spec*1000, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax11.set_xlabel('Time (ms)')
    ax11.set_ylabel('Frequency (Hz)')
    ax11.set_title(f'Nonlinear Path Output Spectrogram (Ch{on_ch}: {drnl.fc[on_ch]:.1f} Hz)')
    ax11.set_ylim([0, 3000])
    plt.colorbar(im, ax=ax11, label='Power (dB)')
    
    plt.suptitle('DRNL Filterbank: Nonlinear Path Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = TEST_FIGURES_DIR / 'drnl_nonlinear_path.png'
    plt.savefig(output_file, dpi=600, format='png', bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    print("\n" + "="*80)
    print("NONLINEAR PATH TEST COMPLETE")
    print("="*80)


def test_drnl_input_output():
    """Test and visualize input vs output comparison."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100  # Hz
    test_freq = 1000  # Hz
    test_level = 60  # dB SPL
    dur = 0.05  # seconds
    
    print("="*80)
    print("DRNL INPUT/OUTPUT COMPARISON TEST")
    print("="*80)
    print(f"\nTest signal: {test_freq} Hz tone @ {test_level} dB SPL")
    print(f"Duration: {dur*1000:.1f} ms")
    print(f"Sampling rate: {fs} Hz")
    
    # Create DRNL filterbank
    drnl = DRNLFilterbank((250, 8000), fs=fs, n_channels=50, dtype=torch.float64)
    
    # Generate input signal
    t = torch.arange(0, dur, 1/fs, dtype=torch.float64)
    tone = torch.sin(2 * torch.pi * test_freq * t)
    
    # Scale to SPL
    rms_target = 20e-6 * 10.0 ** (test_level / 20.0)
    rms_current = torch.sqrt(torch.mean(tone ** 2))
    tone = tone * (rms_target / rms_current)
    
    # Process through DRNL
    print(f"\nProcessing through full DRNL filterbank...")
    with torch.no_grad():
        output = drnl(tone)  # [num_channels, time]
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4e}, {output.max():.4e}]")
    
    # Find on-frequency channel
    fc_diff = torch.abs(drnl.fc - test_freq)
    on_ch = torch.argmin(fc_diff).item()
    print(f"  On-frequency channel: {on_ch} (fc = {drnl.fc[on_ch]:.1f} Hz)")
    
    # Convert to numpy
    x = tone.numpy()
    y_all = output.numpy()
    t_ms = t.numpy() * 1000
    
    # ========== PLOT ==========
    print(f"\nGenerating figure...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # Row 1: Input time domain
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(t_ms, x, color='steelblue', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (Pa)')
    ax1.set_title(f'Input: {test_freq} Hz @ {test_level} dB SPL')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 10])
    
    # Row 1, col 2: Input spectrogram
    ax2 = plt.subplot(4, 2, 2)
    f, t_spec, Sxx = scipy_signal.spectrogram(x, fs=fs, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im1 = ax2.pcolormesh(t_spec*1000, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Input Spectrogram')
    ax2.set_ylim([0, 3000])
    plt.colorbar(im1, ax=ax2, label='Power (dB)')
    
    # Row 2: Output time domain (on-frequency channel)
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(t_ms, y_all[on_ch, :], color='darkred', linewidth=1)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('BM Velocity')
    ax3.set_title(f'Output: On-frequency Ch{on_ch} ({drnl.fc[on_ch]:.1f} Hz)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 10])
    
    # Row 2, col 2: Output spectrogram (on-frequency)
    ax4 = plt.subplot(4, 2, 4)
    f, t_spec, Sxx_out = scipy_signal.spectrogram(y_all[on_ch, :], fs=fs, nperseg=512, noverlap=256)
    Sxx_out_db = 10 * np.log10(Sxx_out + 1e-10)
    im2 = ax4.pcolormesh(t_spec*1000, f, Sxx_out_db, shading='gouraud', cmap='viridis')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Output Spectrogram (on-freq)')
    ax4.set_ylim([0, 3000])
    plt.colorbar(im2, ax=ax4, label='Power (dB)')
    
    # Row 3: All channels output (heatmap)
    ax5 = plt.subplot(4, 2, 5)
    t_max = 10  # ms
    t_mask = t_ms <= t_max
    im3 = ax5.imshow(y_all[:, t_mask], aspect='auto', origin='lower', cmap='seismic',
                    extent=[0, t_max, 0, drnl.num_channels])
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Channel')
    ax5.set_title('Output: All channels (heatmap)')
    plt.colorbar(im3, ax=ax5, label='BM Velocity')
    
    # Mark on-frequency channel
    ax5.axhline(on_ch, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    ax5.text(t_max*0.98, on_ch, f' Ch{on_ch}', color='yellow', fontsize=8, va='center', ha='right')
    
    # Add fc labels
    yticks = [0, 12, 24, 36, 49]
    ytick_labels = [f'{drnl.fc[i]:.0f} Hz' if i < len(drnl.fc) else '' for i in yticks]
    ax5.set_yticks(yticks)
    ax5.set_yticklabels(ytick_labels)
    
    # Row 3, col 2: RMS output vs channel
    ax6 = plt.subplot(4, 2, 6)
    output_rms = np.sqrt(np.mean(y_all**2, axis=1))
    ax6.plot(drnl.fc.numpy(), 20*np.log10(output_rms + 1e-10), 'o-', 
            markersize=4, linewidth=1, color='darkgreen')
    ax6.axvline(test_freq, color='red', linestyle='--', linewidth=1, alpha=0.5, 
               label=f'Signal: {test_freq} Hz')
    ax6.set_xlabel('Channel CF (Hz)')
    ax6.set_ylabel('RMS Output (dB)')
    ax6.set_title('Output RMS vs Channel CF')
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend()
    
    # Row 4: Comparison (span 2 columns)
    ax7 = plt.subplot(4, 1, 4)
    
    # Normalize for comparison
    x_norm = x / np.max(np.abs(x))
    y_norm = y_all[on_ch, :] / np.max(np.abs(y_all[on_ch, :]))
    
    ax7.plot(t_ms, x_norm, label='Input (normalized)', alpha=0.6, linewidth=1)
    ax7.plot(t_ms, y_norm, label=f'Output Ch{on_ch} (normalized)', linewidth=1)
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Normalized Amplitude')
    ax7.set_title('Input vs Output Comparison (normalized)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 10])
    
    plt.suptitle('DRNL Filterbank: Input/Output Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = TEST_FIGURES_DIR / 'drnl_input_output.png'
    plt.savefig(output_file, dpi=600, format='png', bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    print("\n" + "="*80)
    print("INPUT/OUTPUT TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_drnl_linear_path()
    print("\n")
    test_drnl_nonlinear_path()
    print("\n")
    test_drnl_input_output()
