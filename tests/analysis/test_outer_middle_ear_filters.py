"""
Outer/Middle Ear Filters - Test Suite

Contents:
1. test_outer_middle_ear_comparison: Compares three filter implementations

Structure:
- Osses2021 approach: Headphone + Middle ear filters
- Glasberg2002 free field: OuterMiddleEarFilter with ANSI S3.4-2007
- Glasberg2002 diffuse field: OuterMiddleEarFilter with diffuse field data

Figures generated:
- outer_middle_ear_comparison.png: Impulse response, frequency response, phase response
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_amt.common import HeadphoneFilter, MiddleEarFilter, OuterMiddleEarFilter


def test_outer_middle_ear_comparison():
    """Compare different outer/middle ear filter implementations."""
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Outer/Middle Ear Filter Comparison")
    print("=" * 80)
    
    fs = 32000  # Glasberg2002 requires 32 kHz
    
    # Initialize filters
    print("\nInitializing filters...")
    
    # Osses2021 approach (Headphone + Middle ear)
    headphone_filter = HeadphoneFilter(fs=fs, phase_type='zero')
    middleear_filter = MiddleEarFilter(fs=fs, phase_type='zero')
    print("  ✓ Osses2021: HeadphoneFilter + MiddleEarFilter (Jepsen 2008)")
    
    # Glasberg2002 approach - Free field
    glasberg_free_1997 = OuterMiddleEarFilter(
        fs=fs, 
        compensation_type='tfOuterMiddle1997',
        field_type='free'
    )
    print("  ✓ Glasberg2002: OuterMiddleEarFilter (1997, Free field)")
    
    # Glasberg2002 approach - Diffuse field
    glasberg_diffuse_1997 = OuterMiddleEarFilter(
        fs=fs,
        compensation_type='tfOuterMiddle1997',
        field_type='diffuse'
    )
    print("  ✓ Glasberg2002: OuterMiddleEarFilter (1997, Diffuse field)")
    
    # Glasberg2002 approach - 2007 version
    glasberg_free_2007 = OuterMiddleEarFilter(
        fs=fs,
        compensation_type='tfOuterMiddle2007',
        field_type='free'
    )
    print("  ✓ Glasberg2002: OuterMiddleEarFilter (2007, Free field)")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Outer/Middle Ear Filter Implementations Comparison', 
                 fontsize=16, fontweight='bold')
    
    # ========================================================================
    # Row 1: Impulse Responses
    # ========================================================================
    print("\nComputing impulse responses...")
    
    # Generate impulse
    impulse_len = 2048
    impulse = torch.zeros(1, impulse_len)  # [batch, samples]
    impulse[0, impulse_len // 2] = 1.0
    
    # Osses2021: Headphone + Middle ear (needs [batch, channels, samples])
    impulse_3d = impulse.unsqueeze(1)  # [batch, 1, samples]
    h_headphone_3d = headphone_filter(impulse_3d.clone())
    h_osses_3d = middleear_filter(h_headphone_3d)
    h_osses = h_osses_3d.squeeze()  # Back to [samples]
    
    # Glasberg filters (work with [batch, samples])
    h_glass_free_1997 = glasberg_free_1997(impulse.clone()).squeeze()
    h_glass_diffuse_1997 = glasberg_diffuse_1997(impulse.clone()).squeeze()
    h_glass_free_2007 = glasberg_free_2007(impulse.clone()).squeeze()
    
    # Time vector (ms)
    t_ms = torch.arange(impulse_len) / fs * 1000
    
    # Single centered plot: Zoomed impulse response (30-34 ms)
    ax = fig.add_subplot(gs[0, :])
    zoom_start_ms = 30
    zoom_end_ms = 34
    zoom_mask = (t_ms >= zoom_start_ms) & (t_ms <= zoom_end_ms)
    ax.plot(t_ms[zoom_mask].numpy(), h_osses[zoom_mask].numpy(), 'b-', linewidth=2, label='Osses2021 (Headphone+MiddleEar)', alpha=0.8)
    ax.plot(t_ms[zoom_mask].numpy(), h_glass_free_1997[zoom_mask].numpy(), 'r-', linewidth=2, label='Glasberg2002 Free field (1997)', alpha=0.8)
    ax.plot(t_ms[zoom_mask].numpy(), h_glass_diffuse_1997[zoom_mask].numpy(), 'g-', linewidth=2, label='Glasberg2002 Diffuse field (1997)', alpha=0.8)
    ax.plot(t_ms[zoom_mask].numpy(), h_glass_free_2007[zoom_mask].numpy(), 'm--', linewidth=2, label='Glasberg2002 Free field (2007)', alpha=0.6)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Impulse Response (Zoomed: {zoom_start_ms}-{zoom_end_ms} ms)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Row 2: Frequency Response (Magnitude)
    # ========================================================================
    print("Computing frequency responses...")
    
    nfft = 16384
    
    # Osses2021: Get frequency response from cascaded impulse response
    # This is more accurate than trying to combine individual filter responses
    H_osses = torch.fft.rfft(h_osses, n=nfft)
    resp_osses_db = 20 * torch.log10(torch.abs(H_osses) + 1e-12)
    freqs_osses = torch.linspace(0, fs / 2, nfft // 2 + 1)
    
    # Glasberg filters
    freqs_glass, resp_glass_free_1997_db = glasberg_free_1997.get_frequency_response(nfft=nfft)
    _, resp_glass_diffuse_1997_db = glasberg_diffuse_1997.get_frequency_response(nfft=nfft)
    _, resp_glass_free_2007_db = glasberg_free_2007.get_frequency_response(nfft=nfft)
    
    # Full width: Frequency Response
    ax = fig.add_subplot(gs[1, :])
    ax.semilogx(freqs_osses.numpy(), resp_osses_db.numpy(), 'b-', linewidth=2, label='Osses2021')
    ax.semilogx(freqs_glass.numpy(), resp_glass_free_1997_db.numpy(), 'r-', linewidth=2, label='Free field 1997')
    ax.semilogx(freqs_glass.numpy(), resp_glass_diffuse_1997_db.numpy(), 'g-', linewidth=2, label='Diffuse field 1997')
    ax.semilogx(freqs_glass.numpy(), resp_glass_free_2007_db.numpy(), 'm--', linewidth=2, label='Free field 2007', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Frequency Response')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([20, fs/2])
    ax.set_ylim([-50, 20])
    
    # ========================================================================
    # Row 3: Group Delay
    # ========================================================================
    print("Computing group delay...")
    
    # Compute phase from impulse response FFT
    def get_phase_response(h, nfft):
        H = torch.fft.rfft(h, n=nfft)
        phase = torch.angle(H)
        return phase
    
    phase_osses = get_phase_response(h_osses, nfft)
    phase_free_1997 = get_phase_response(h_glass_free_1997, nfft)
    phase_diffuse_1997 = get_phase_response(h_glass_diffuse_1997, nfft)
    phase_free_2007 = get_phase_response(h_glass_free_2007, nfft)
    
    # Group delay
    ax = fig.add_subplot(gs[2, :])
    
    def compute_group_delay(phase, freqs):
        # Group delay = -d(phase)/d(omega) = -d(phase)/d(2*pi*f)
        # Approximate with finite differences
        # Use numpy for unwrap (torch doesn't have it)
        phase_unwrapped = torch.from_numpy(np.unwrap(phase.numpy()))
        dphase = torch.diff(phase_unwrapped)
        dfreq = torch.diff(freqs)
        group_delay = -dphase / (2 * np.pi * dfreq)
        # Convert to milliseconds
        group_delay_ms = group_delay * 1000
        return group_delay_ms
    
    gd_osses = compute_group_delay(phase_osses, freqs_osses)
    gd_free_1997 = compute_group_delay(phase_free_1997, freqs_glass)
    gd_diffuse_1997 = compute_group_delay(phase_diffuse_1997, freqs_glass)
    gd_free_2007 = compute_group_delay(phase_free_2007, freqs_glass)
    
    # Use center frequencies for group delay plot
    freqs_gd = (freqs_osses[:-1] + freqs_osses[1:]) / 2
    
    ax.semilogx(freqs_gd.numpy(), gd_osses.numpy(), 'b-', linewidth=1.5, label='Osses2021', alpha=0.8)
    ax.semilogx(freqs_gd.numpy(), gd_free_1997.numpy(), 'r-', linewidth=1.5, label='Free field 1997', alpha=0.8)
    ax.semilogx(freqs_gd.numpy(), gd_diffuse_1997.numpy(), 'g-', linewidth=1.5, label='Diffuse field 1997', alpha=0.8)
    ax.semilogx(freqs_gd.numpy(), gd_free_2007.numpy(), 'm--', linewidth=1.5, label='Free field 2007', alpha=0.6)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (ms)')
    ax.set_title('Group Delay')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([20, fs/2])
    ax.set_ylim([0, 5])  # Reasonable range for group delay
    
    # ========================================================================
    # Row 4: Transfer Function Comparison (Glasberg Data)
    # ========================================================================
    print("Plotting Glasberg transfer functions...")
    
    # Left: Free vs Diffuse
    ax = fig.add_subplot(gs[3, 0])
    
    # Get original transfer function data
    fvec_free, tf_free = glasberg_free_1997.get_transfer_function()
    fvec_diff, tf_diff = glasberg_diffuse_1997.get_transfer_function()
    
    ax.semilogx(fvec_free.numpy(), tf_free.numpy(), 'r-', linewidth=2, label='Free field')
    ax.semilogx(fvec_diff.numpy(), tf_diff.numpy(), 'g-', linewidth=2, label='Diffuse field')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Transfer Function (dB)')
    ax.set_title('Glasberg Transfer Functions (Free vs Diffuse)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([20, 16000])
    
    # Right: Difference (Free - Diffuse)
    ax = fig.add_subplot(gs[3, 1])
    tf_diff_comparison = tf_free - tf_diff
    
    ax.semilogx(fvec_free.numpy(), tf_diff_comparison.numpy(), 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Difference (dB)')
    ax.set_title('Free Field - Diffuse Field')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([20, 16000])
    
    plt.savefig(TEST_FIGURES_DIR / 'outer_middle_ear_comparison.png', dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: outer_middle_ear_comparison.png")
    
    # ========================================================================
    # Quantitative Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("Quantitative Analysis")
    print("=" * 80)
    
    # Frequency response differences at key frequencies
    test_freqs_hz = [100, 500, 1000, 2000, 4000, 8000]
    print(f"\nFrequency response at key frequencies (dB):")
    print(f"{'Freq (Hz)':<12} {'Osses2021':<12} {'Free field 1997':<17} {'Diffuse field 1997':<19} {'Free field 2007':<17}")
    print("-" * 80)
    
    for f_test in test_freqs_hz:
        # Find closest frequency index
        idx = torch.argmin(torch.abs(freqs_glass - f_test))
        print(f"{f_test:<12} {resp_osses_db[idx].item():<12.2f} "
              f"{resp_glass_free_1997_db[idx].item():<17.2f} "
              f"{resp_glass_diffuse_1997_db[idx].item():<19.2f} "
              f"{resp_glass_free_2007_db[idx].item():<17.2f}")
    
    # Overall statistics
    print(f"\nOverall frequency response statistics (20-16000 Hz):")
    print(f"  Osses2021: mean = {resp_osses_db.mean():.2f} dB, std = {resp_osses_db.std():.2f} dB")
    print(f"  Free field 1997: mean = {resp_glass_free_1997_db.mean():.2f} dB, std = {resp_glass_free_1997_db.std():.2f} dB")
    print(f"  Diffuse field 1997: mean = {resp_glass_diffuse_1997_db.mean():.2f} dB, std = {resp_glass_diffuse_1997_db.std():.2f} dB")
    print(f"  Free field 2007: mean = {resp_glass_free_2007_db.mean():.2f} dB, std = {resp_glass_free_2007_db.std():.2f} dB")

if __name__ == '__main__':
    test_outer_middle_ear_comparison()
