"""
Gammatone Filterbank Implementation Comparison - Test Suite

Contents:
1. test_gammatone_implementations: Compare SOS vs Polynomial Gammatone implementations

Structure:
- Frequency response accuracy comparison
- Peak location error analysis
- Statistical summary (mean/max/std errors)
- Top 5 worst channels visualization

Figures generated:
- gammatone_poly_vs_sos.png: 4-panel comparison (errors, improvement, frequency/impulse responses)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import GammatoneFilterbank


def test_gammatone_implementations():
    """Compare polynomial vs SOS cascade implementations."""
    
    # Create test_figures directory
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100
    flow = 20
    fhigh = 20000
    
    print("="*80)
    print("GAMMATONE FILTERBANK: POLYNOMIAL vs SOS COMPARISON")
    print("="*80)
    
    # Create both filterbanks using the implementation parameter
    print("\nCreating filterbanks...")
    fb_poly = GammatoneFilterbank((flow, fhigh), fs=fs, implementation='poly')
    fb_sos = GammatoneFilterbank((flow, fhigh), fs=fs, implementation='sos')
    
    print(f"  Polynomial implementation: {fb_poly.num_channels} channels")
    print(f"  SOS implementation:        {fb_sos.num_channels} channels")
    
    # Generate impulse
    impulse_len = 32768
    impulse = torch.zeros(impulse_len)
    impulse[0] = 1.0
    
    print(f"\nGenerating impulse response (length={impulse_len})...")
    
    # Apply both filterbanks
    with torch.no_grad():
        response_poly = fb_poly(impulse)
        response_sos = fb_sos(impulse)
    
    print(f"  Polynomial response shape: {response_poly.shape}")
    print(f"  SOS response shape:        {response_sos.shape}")
    
    # FFT analysis
    NFFT = 32768
    freqs = np.fft.rfftfreq(NFFT, 1/fs)
    freq_resolution = fs / NFFT
    
    print(f"\nFFT analysis (NFFT={NFFT}, resolution={freq_resolution:.3f} Hz)...")
    
    # Analyze peaks for both implementations
    def find_peak(response, fc_theoretical, freqs, freq_resolution):
        """Find peak frequency with parabolic interpolation."""
        # FFT
        freq_resp = np.fft.rfft(response.numpy(), n=NFFT)
        mag = np.abs(freq_resp)
        
        # Search in narrow range
        search_range_hz = max(fc_theoretical * 0.3, 200.0)
        f_min = max(0, fc_theoretical - search_range_hz)
        f_max = min(fs/2, fc_theoretical + search_range_hz)
        
        idx_min = int(f_min / freq_resolution)
        idx_max = int(f_max / freq_resolution)
        idx_max = min(idx_max, len(freqs) - 1)
        
        # Find peak
        peak_idx_local = np.argmax(mag[idx_min:idx_max+1])
        peak_idx = idx_min + peak_idx_local
        
        # Parabolic interpolation
        if peak_idx > 0 and peak_idx < len(freqs) - 1:
            y_prev = mag[peak_idx - 1]
            y_peak = mag[peak_idx]
            y_next = mag[peak_idx + 1]
            
            # Smith's method
            denom = 2 * (2 * y_peak - y_prev - y_next)
            if abs(denom) > 1e-10:
                delta = (y_next - y_prev) / denom
            else:
                delta = 0.0
            
            peak_freq = freqs[peak_idx] + delta * freq_resolution
        else:
            peak_freq = freqs[peak_idx]
        
        return peak_freq, mag[peak_idx]
    
    # Compare peak locations
    print(f"\n{'Channel':>7} {'fc (Hz)':>10} {'Poly (Hz)':>12} {'Err %':>8} {'SOS (Hz)':>12} {'Err %':>8} {'Improvement':>12}")
    print("-"*80)
    
    errors_poly = []
    errors_sos = []
    problematic_poly = 0
    problematic_sos = 0
    
    for ch in range(fb_poly.num_channels):
        fc_theoretical = fb_poly.fc[ch].item()
        
        # Find peaks
        peak_poly, mag_poly = find_peak(response_poly[ch], fc_theoretical, freqs, freq_resolution)
        peak_sos, mag_sos = find_peak(response_sos[ch], fc_theoretical, freqs, freq_resolution)
        
        # Calculate errors
        error_poly = abs(peak_poly - fc_theoretical)
        error_sos = abs(peak_sos - fc_theoretical)
        error_pct_poly = (error_poly / fc_theoretical) * 100
        error_pct_sos = (error_sos / fc_theoretical) * 100
        
        errors_poly.append(error_pct_poly)
        errors_sos.append(error_pct_sos)
        
        if error_pct_poly > 10.0:
            problematic_poly += 1
        if error_pct_sos > 10.0:
            problematic_sos += 1
        
        # Show improvement
        improvement = error_pct_poly - error_pct_sos
        improvement_str = f"{improvement:+.1f}%" if abs(improvement) > 0.1 else "~"
        
        # Print ALL channels
        print(f"{ch:7d} {fc_theoretical:10.1f} {peak_poly:12.1f} {error_pct_poly:7.1f}% "
              f"{peak_sos:12.1f} {error_pct_sos:7.1f}% {improvement_str:>12}")
    
    # Summary statistics
    errors_poly = np.array(errors_poly)
    errors_sos = np.array(errors_sos)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Metric':<30} {'Polynomial':>20} {'SOS':>20}")
    print("-"*80)
    print(f"{'Mean absolute error':<30} {errors_poly.mean():19.2f}% {errors_sos.mean():19.2f}%")
    print(f"{'Max absolute error':<30} {errors_poly.max():19.2f}% {errors_sos.max():19.2f}%")
    print(f"{'Std absolute error':<30} {errors_poly.std():19.2f}% {errors_sos.std():19.2f}%")
    print(f"{'Channels with >10% error':<30} {problematic_poly:20d} {problematic_sos:20d}")
    print(f"{'Channels with >5% error':<30} {sum(errors_poly > 5.0):20d} {sum(errors_sos > 5.0):20d}")
    
    # Improvement
    print("\n" + "="*80)
    if errors_sos.mean() < errors_poly.mean():
        improvement = ((errors_poly.mean() - errors_sos.mean()) / errors_poly.mean()) * 100
        print(f"✓ SOS implementation is BETTER: {improvement:.1f}% average error reduction")
    else:
        print(f"✗ SOS implementation is NOT better")
    print("="*80)
    
    # Plot comparison for problematic channels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gammatone: Polynomial vs SOS Implementation', fontsize=14, fontweight='bold')
    
    # Plot 1: Error comparison
    axes[0, 0].plot(errors_poly, 'b-o', label='Polynomial', markersize=3, alpha=0.7)
    axes[0, 0].plot(errors_sos, 'r-x', label='SOS', markersize=3, alpha=0.7)
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Error [%]')
    axes[0, 0].set_title('Center Frequencies Error')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Error difference
    error_diff = errors_poly - errors_sos
    axes[0, 1].bar(range(len(error_diff)), error_diff, color=['g' if x > 0 else 'r' for x in error_diff])
    axes[0, 1].axhline(0, color='k', linestyle='-', linewidth=0.8)
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Error Difference [%]\n(Positive = SOS better)')
    axes[0, 1].set_title('Improvement with SOS')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Frequency response comparison for TOP 5 worst polynomial channels
    worst_channels = np.argsort(errors_poly)[-5:][::-1]  # Top 5 worst
    
    # Define colors for channels
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']  # Red, Orange, Green, Purple, Brown
    
    for i, ch in enumerate(worst_channels):
        fc_worst = fb_poly.fc[ch].item()
        
        freq_resp_poly = np.fft.rfft(response_poly[ch].numpy(), n=NFFT)
        freq_resp_sos = np.fft.rfft(response_sos[ch].numpy(), n=NFFT)
        mag_poly_db = 20 * np.log10(np.abs(freq_resp_poly) + 1e-10)
        mag_sos_db = 20 * np.log10(np.abs(freq_resp_sos) + 1e-10)
        
        # Zoom to ±500 Hz around fc
        zoom_range = 500
        zoom_idx_low = max(0, int((fc_worst - zoom_range) / freq_resolution))
        zoom_idx_high = min(len(freqs), int((fc_worst + zoom_range) / freq_resolution))
        
        # Plot with color per channel, style per implementation
        axes[1, 0].plot(freqs[zoom_idx_low:zoom_idx_high], mag_poly_db[zoom_idx_low:zoom_idx_high], 
                        color=colors[i], linestyle='-', linewidth=1.5, alpha=0.7,
                        label=f'Ch {ch} Poly ({fc_worst:.0f} Hz)')
        axes[1, 0].plot(freqs[zoom_idx_low:zoom_idx_high], mag_sos_db[zoom_idx_low:zoom_idx_high], 
                        color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'Ch {ch} SOS')
        
        # Mark theoretical fc in gray
        axes[1, 0].axvline(fc_worst, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    axes[1, 0].set_xlabel('Frequency [Hz]')
    axes[1, 0].set_ylabel('Magnitude [dB]')
    axes[1, 0].set_title('Frequency Response - Top 5 Worst Channels (by error)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7, ncol=2)
    
    # Plot 4: Impulse response comparison for TOP 5
    t_ms = np.arange(2000) / fs * 1000
    
    for i, ch in enumerate(worst_channels):
        fc_ch = fb_poly.fc[ch].item()
        
        # Plot with same color scheme
        axes[1, 1].plot(t_ms, response_poly[ch, :2000].numpy(), 
                       color=colors[i], linestyle='-', alpha=0.7,
                       label=f'Ch {ch} Poly ({fc_ch:.0f} Hz)')
        axes[1, 1].plot(t_ms, response_sos[ch, :2000].numpy(), 
                       color=colors[i], linestyle='--', alpha=0.7,
                       label=f'Ch {ch} SOS')
    
    axes[1, 1].set_xlabel('Time [ms]')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Impulse Response - Top 5 Worst Channels (by error)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7, ncol=2)
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'gammatone_poly_vs_sos.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")


if __name__ == "__main__":
    test_gammatone_implementations()
