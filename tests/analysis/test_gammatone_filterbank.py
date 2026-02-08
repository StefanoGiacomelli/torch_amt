"""
Gammatone Filterbank - Test Suite

Contents:
1. test_gammatone_filterbank: Tests Gammatone filterbank with both implementations
   - Polynomial implementation ('poly')
   - Second-order sections implementation ('sos')
   - Impulse response analysis
   - Frequency response analysis
   - Center frequency accuracy verification

Structure:
- 20-20000 Hz frequency range
- 4th order Gammatone filters
- Impulse and frequency response characterization
- Theoretical vs measured fc comparison

Figures generated (per implementation):
- gammatone_ir_{implementation}.png: Impulse response (time-domain + spectrogram)
- gammatone_frequency_response_{implementation}.png: Frequency response (magnitude + heatmap)
- gammatone_fc_analysis_{implementation}.png: Center frequency analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.common import GammatoneFilterbank


def _run_gammatone_filterbank_test(implementation='poly'):
    """Test Gammatone filterbank with specified implementation.
    
    Parameters
    ----------
    implementation : str
        Either 'poly' or 'sos' for polynomial or second-order sections implementation.
    """
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 44100  # Hz
    flow = 20   # Hz
    fhigh = 20000  # Hz
    
    print("="*80)
    print(f"GAMMATONE FILTERBANK TEST - Implementation: {implementation.upper()}")
    print("="*80)
    print(f"\nParameters:")
    print(f"  fs = {fs} Hz")
    print(f"  flow = {flow} Hz")
    print(f"  fhigh = {fhigh} Hz")
    print(f"  implementation = {implementation}")
    
    # Create filterbank
    filterbank = GammatoneFilterbank(fc=(flow, fhigh), fs=fs, n=4, 
                                     implementation=implementation, dtype=torch.float32)
    
    print(f"\nFilterbank:")
    print(f"  Number of channels: {filterbank.num_channels}")
    print(f"  fc range: {filterbank.fc[0]:.2f} - {filterbank.fc[-1]:.2f} Hz")
    print(f"        fc: {filterbank.fc.numpy()}")
    
    # Generate impulse
    # For low frequency channels (20 Hz), we need longer impulse response
    # Period at 20 Hz = 1/20 = 0.05s = 2205 samples @ 44.1kHz
    # Use 32768 samples (0.743s) to capture many cycles even for lowest frequency
    impulse_len = 32768  # samples
    impulse = torch.zeros(impulse_len, dtype=torch.float32)
    impulse[0] = 1.0  # Dirac delta
    
    print(f"\nInput: impulse (delta) of {impulse_len} samples ({impulse_len/fs*1000:.1f} ms)")
    
    # Compute impulse response
    with torch.no_grad():
        impulse_response = filterbank(impulse)  # [num_channels, time]
    
    print(f"\nImpulse response:")
    print(f"  Shape: {impulse_response.shape}")
    print(f"  Range: [{impulse_response.min():.2e}, {impulse_response.max():.2e}]")
    
    # Compute frequency response (FFT)
    # Zero-pad for better frequency resolution
    # NFFT = 32768 gives resolution of 44100/32768 ≈ 1.35 Hz (vs 10.77 Hz with 4096)
    nfft = 32768
    impulse_response_padded = torch.nn.functional.pad(impulse_response, (0, nfft - impulse_len))    # Causal padding
    freq_response = torch.fft.rfft(impulse_response_padded, dim=1)
    freq_response_mag = torch.abs(freq_response)
    freq_response_db = 20 * torch.log10(freq_response_mag + 1e-10)
    
    # Frequency axis
    freqs = torch.fft.rfftfreq(nfft, 1/fs).numpy()
    freq_resolution = fs / nfft
    
    print(f"\nFrequency response:")
    print(f"  NFFT: {nfft}")
    print(f"  Frequency resolution: {freq_resolution:.3f} Hz")
    print(f"  Freq range: 0 - {freqs[-1]:.0f} Hz")
    print(f"  Magnitude range: [{freq_response_mag.min():.2e}, {freq_response_mag.max():.2e}]")

    # =============================================================================
    # PLOT 1: Impulse response
    # =============================================================================
    print(f"\nGenerating impulse response plot...")

    fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
    fig1.suptitle('Gammatone Filterbank - Impulse Response', fontsize=14, fontweight='bold')
    
    # Convert to numpy
    ir_np = impulse_response.numpy()
    t = np.arange(impulse_len) / fs * 1000  # ms
    
    # Plot 1a: All channels overlapped (first 3ms)
    t_max_ms = 3
    t_max_samples = int(t_max_ms * fs / 1000)
    for ch in range(filterbank.num_channels):
        fc_hz = filterbank.fc[ch].item()
        label = f'Ch{ch}: {fc_hz:.0f} Hz'
        axes1[0].plot(t[:t_max_samples], ir_np[ch, :t_max_samples], alpha=0.6, linewidth=0.8, label=label)
    
    axes1[0].set_title(f'Impulse Response (first {t_max_ms} ms)')
    axes1[0].set_xlabel('Time [ms]')
    axes1[0].set_ylabel('Amplitude')
    axes1[0].grid(True, alpha=0.3)
    axes1[0].set_xlim([0, t_max_ms])
    axes1[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, ncol=2)
    
    # Plot 1b: Heatmap (all channels, first 3ms)
    t_max_ms_heatmap = 3
    t_max_samples_heatmap = int(t_max_ms_heatmap * fs / 1000)
    extent_time = [0, t_max_ms_heatmap, 0, filterbank.num_channels]
    im1b = axes1[1].imshow(ir_np[:, :t_max_samples_heatmap], 
                           aspect='auto', origin='lower', 
                           extent=extent_time, cmap='seismic')
    axes1[1].set_title('Impulse Response (Heatmap, first 3 ms)')
    axes1[1].set_xlabel('Time [ms]')
    axes1[1].set_ylabel('Filterbank Channel')
    
    # Add fc labels for some channels
    yticks = np.linspace(0, filterbank.num_channels-1, 5, dtype=int)
    ytick_labels = [f'{filterbank.fc[i]:.0f} Hz' if i < len(filterbank.fc) else '' for i in yticks]
    axes1[1].set_yticks(yticks)
    axes1[1].set_yticklabels(ytick_labels)
    
    plt.colorbar(im1b, ax=axes1[1], label='Amplitude')
    
    plt.tight_layout()
    output1 = TEST_FIGURES_DIR / f'gammatone_ir_{implementation}.png'
    plt.savefig(output1, dpi=600, format='png', bbox_inches='tight')
    print(f"  Plot saved: {output1}")

    # =============================================================================
    # PLOT 2: Frequency response
    # =============================================================================
    print(f"\nGenerating frequency response plot...")
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    fig2.suptitle('Gammatone Filterbank - Frequency Response', fontsize=14, fontweight='bold')
    
    # Convert to numpy
    fr_db_np = freq_response_db.numpy()
    
    # Plot 2a: All channels overlapped
    for ch in range(filterbank.num_channels):
        fc_hz = filterbank.fc[ch].item()
        label = f'Ch{ch}: {fc_hz:.0f} Hz'
        axes2[0].plot(freqs, fr_db_np[ch, :], alpha=0.6, linewidth=1.0, label=label)
    
    axes2[0].set_title(f'Frequency Response (Magnitude)')
    axes2[0].set_xlabel('Frequency [Hz]')
    axes2[0].set_ylabel('Magnitude [dB]')
    axes2[0].set_xlim([0, min(fhigh * 1.2, fs/2)])
    axes2[0].set_ylim([-60, 10])
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, ncol=2)
    
    # Add vertical lines at fc
    for ch in np.linspace(0, filterbank.num_channels-1, 5, dtype=int):
        if ch < len(filterbank.fc):
            fc_hz = filterbank.fc[ch].item()
            axes2[0].axvline(fc_hz, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Plot 2b: Frequency heatmap
    # Limit to band of interest
    freq_max_hz = min(fhigh * 1.2, fs/2)
    freq_max_idx = min(np.searchsorted(freqs, freq_max_hz), len(freqs) - 1)
    extent_freq = [0, freqs[freq_max_idx], 0, filterbank.num_channels]
    
    im2b = axes2[1].imshow(fr_db_np[:, :freq_max_idx+1], 
                           aspect='auto', origin='lower', extent=extent_freq, 
                           cmap='viridis', vmin=-60, vmax=0)
    axes2[1].set_title('Frequency Response (Magnitude Heatmap)')
    axes2[1].set_xlabel('Frequency [Hz]')
    axes2[1].set_ylabel('Filterbank Channel')
    
    # Add fc labels for some channels
    axes2[1].set_yticks(yticks)
    axes2[1].set_yticklabels(ytick_labels)
    
    plt.colorbar(im2b, ax=axes2[1], label='Magnitude [dB]')
    
    plt.tight_layout()
    output2 = TEST_FIGURES_DIR / f'gammatone_frequency_response_{implementation}.png'
    plt.savefig(output2, dpi=600, format='png', bbox_inches='tight')
    print(f"  Plot saved: {output2}")

    # =============================================================================
    # PLOT 3: Comparison of theoretical vs measured fc
    # =============================================================================
    print(f"\nAnalyzing frequency response peaks...")
    
    def parabolic_interpolation(y_prev, y_peak, y_next):
        """
        Parabolic interpolation to find sub-bin peak location.
        
        Returns offset from peak_idx (in bins), can be fractional.
        Reference: Smith, J.O. "Spectral Audio Signal Processing", 
        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        """
        denom = 2 * (2*y_peak - y_prev - y_next)
        if abs(denom) < 1e-10:
            return 0.0
        delta = (y_next - y_prev) / denom
        return delta

    # Find peak of each channel with sub-bin accuracy
    peak_freqs = []
    peak_mags_db = []
    problematic_channels = []
    
    for ch in range(filterbank.num_channels):
        fc_theoretical = filterbank.fc[ch].item()
        
        # Search for peak in a NARROW range around theoretical fc
        # Use ±30% of fc or ±200 Hz, whichever is larger
        # This prevents finding harmonics or other spurious peaks
        search_range_hz = max(fc_theoretical * 0.3, 200.0)
        f_min = max(0, fc_theoretical - search_range_hz)
        f_max = min(fs/2, fc_theoretical + search_range_hz)
        
        # Convert to bin indices
        idx_min = int(f_min / freq_resolution)
        idx_max = int(f_max / freq_resolution)
        idx_max = min(idx_max, len(freqs) - 1)
        
        # Find peak within search range using LINEAR magnitude (not dB)
        # This is more robust for low-magnitude signals
        search_mags = freq_response_mag[ch, idx_min:idx_max+1]
        peak_idx_local = torch.argmax(search_mags).item()
        peak_idx = idx_min + peak_idx_local
        
        # Check for problematic channels
        # Use RELATIVE threshold: peak should be at least 0.1% of max response
        peak_mag = freq_response_mag[ch, peak_idx].item()
        max_mag_channel = torch.max(freq_response_mag[ch, :]).item()
        
        # Flag as problematic if: peak at DC, OR peak is less than 0.1% of channel's max
        is_problematic = (peak_idx == 0) or (peak_mag < 0.001 * max_mag_channel)
        
        if is_problematic:
            problematic_channels.append((ch, fc_theoretical, peak_idx, peak_mag, max_mag_channel))
        
        # Parabolic interpolation for sub-bin accuracy
        if peak_idx > 0 and peak_idx < len(freqs) - 1 and not is_problematic:
            # Use magnitude (not dB) for interpolation
            y_prev = freq_response_mag[ch, peak_idx - 1].item()
            y_peak = freq_response_mag[ch, peak_idx].item()
            y_next = freq_response_mag[ch, peak_idx + 1].item()
            
            delta = parabolic_interpolation(y_prev, y_peak, y_next)
            
            # Refined frequency
            peak_freq = freqs[peak_idx] + delta * freq_resolution
        else:
            peak_freq = freqs[peak_idx]
        
        peak_mag_db = freq_response_db[ch, peak_idx].item()
        peak_freqs.append(peak_freq)
        peak_mags_db.append(peak_mag_db)
    
    if problematic_channels:
        print(f"\nWARNING: {len(problematic_channels)} channels with potential issues:")
        for ch, fc, pidx, pmag, maxmag in problematic_channels[:5]:
            print(f"  Ch {ch}: fc={fc:.1f} Hz, peak_idx={pidx} ({freqs[pidx]:.1f} Hz), peak_mag={pmag:.2e}, max_mag={maxmag:.2e}")

    # Convert to arrays
    peak_freqs = np.array(peak_freqs)
    peak_mags_db = np.array(peak_mags_db)
    fc_np = filterbank.fc.numpy()
    
    # Calculate errors for ALL channels
    errors_hz = peak_freqs - fc_np
    errors_percent = (errors_hz / fc_np) * 100
    
    print(f"\nfc error statistics (ALL {len(errors_hz)} channels):")
    print(f"  Mean error: {np.mean(np.abs(errors_hz)):.2f} Hz ({np.mean(np.abs(errors_percent)):.2f}%)")
    print(f"  Max error: {np.max(np.abs(errors_hz)):.2f} Hz ({np.max(np.abs(errors_percent)):.2f}%)")
    print(f"  Std error: {np.std(errors_hz):.2f} Hz")
    print(f"  Frequency resolution: {freq_resolution:.3f} Hz")
    
    print(f"\nALL channels (fc, measured, error):")
    for idx in range(len(fc_np)):
        print(f"  Ch {idx:2d}: fc={fc_np[idx]:8.1f} Hz, measured={peak_freqs[idx]:8.1f} Hz, error={errors_hz[idx]:7.1f} Hz ({errors_percent[idx]:6.1f}%)")
    
    print(f"\nMeasured peak magnitude statistics (dB):")
    print(f"  Mean magnitude: {peak_mags_db.mean():.2f} dB")
    print(f"  Std magnitude: {peak_mags_db.std():.2f} dB")

    # Comparison plot
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
    fig3.suptitle('Gammatone Filterbank - Center Frequency Verification', fontsize=14, fontweight='bold')
    
    # Plot 3a: theoretical vs measured fc
    axes3[0].plot(fc_np, 'b-o', label='Theoretical fc', markersize=4)
    axes3[0].plot(peak_freqs, 'r--x', label='Measured fc (FFT peaks)', markersize=4)
    axes3[0].set_xlabel('Filterbank Channel')
    axes3[0].set_ylabel('Frequency [Hz]')
    axes3[0].set_title('Center Frequencies: Theoretical vs Measured')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].legend()
    
    # Plot 3b: Percentage error
    axes3[1].plot(errors_percent, 'g-o', markersize=4, label='Error')
    axes3[1].axhline(0, color='k', linestyle='-', linewidth=0.8, label='Zero error')
    axes3[1].set_xlabel('Filterbank Channel')
    axes3[1].set_ylabel('Error [%]')
    axes3[1].set_title('Percentage error in fc (positive = peak > theoretical fc)')
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend()
    
    plt.tight_layout()
    output3 = TEST_FIGURES_DIR / f'gammatone_fc_analysis_{implementation}.png'
    plt.savefig(output3, dpi=600, format='png', bbox_inches='tight')
    print(f"  Plot saved: {output3}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print(f"  Generated plots:")
    print(f"    1. {output1}")
    print(f"    2. {output2}")
    print(f"    3. {output3}")
    print("="*80)


def test_gammatone_filterbank_poly():
    """Test Gammatone filterbank with polynomial implementation."""
    _run_gammatone_filterbank_test('poly')


def test_gammatone_filterbank_sos():
    """Test Gammatone filterbank with second-order sections implementation."""
    _run_gammatone_filterbank_test('sos')


if __name__ == '__main__':
    test_gammatone_filterbank_poly()
    test_gammatone_filterbank_sos()
