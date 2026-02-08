"""
Multi-Resolution FFT - Test Suite

Contents:
1. test_multi_resolution_fft: Verifies FFT with multiple window lengths

Structure:
- Tests frequency-dependent window selection
- Verifies time-frequency resolution tradeoff
- Uses chirp, pure tone, and tone burst signals

Figures generated:
- multi_resolution_fft.png: Complete analysis with multiple signals
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import MultiResolutionFFT


def generate_test_signals(fs=32000, duration=1.0):
    """
    Generate test signals for multi-resolution FFT analysis.
    
    Args:
        fs: Sampling rate in Hz
        duration: Signal duration in seconds
        
    Returns:
        chirp: Frequency sweep from 50 Hz to 15 kHz
        tone_1khz: Pure tone at 1 kHz
        tone_burst: Brief tone burst (good for testing temporal resolution)
        t: Time vector
    """
    t = torch.linspace(0, duration, int(fs * duration))
    
    # 1. Chirp (linear frequency sweep)
    f0 = 50  # Start frequency
    f1 = 15000  # End frequency
    # Linear chirp: f(t) = f0 + (f1 - f0) * t / T
    # Phase: phi(t) = 2*pi * (f0*t + (f1-f0)*t^2 / (2*T))
    chirp = torch.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    
    # 2. Pure tone at 1 kHz
    tone_1khz = torch.sin(2 * np.pi * 1000 * t)
    
    # 3. Tone burst (100 Hz, 20 ms duration, at t=0.5s)
    tone_burst = torch.zeros_like(t)
    burst_duration = 0.02  # 20 ms
    burst_start = 0.5
    burst_end = burst_start + burst_duration
    burst_mask = (t >= burst_start) & (t <= burst_end)
    
    # Apply Hann window to avoid clicks
    burst_samples = burst_mask.sum()
    hann_window = torch.hann_window(burst_samples)
    tone_burst[burst_mask] = hann_window * torch.sin(2 * np.pi * 100 * t[burst_mask])
    
    return chirp, tone_1khz, tone_burst, t


def test_multi_resolution_fft():
    """Test multi-resolution FFT with various signals."""
    
    print("=" * 80)
    print("Multi-Resolution FFT Test")
    print("=" * 80)
    
    # Parameters
    fs = 32000
    duration = 1.0
    
    # Generate test signals
    print("\nGenerating test signals...")
    chirp, tone_1khz, tone_burst, t = generate_test_signals(fs, duration)
    
    # Add batch dimension
    chirp_batch = chirp.unsqueeze(0)
    tone_batch = tone_1khz.unsqueeze(0)
    burst_batch = tone_burst.unsqueeze(0)
    
    # Initialize multi-resolution FFT
    print("Initializing MultiResolutionFFT...")
    multi_fft = MultiResolutionFFT(fs=fs, learnable=False)
    
    print(f"  Window lengths: {multi_fft.window_lengths}")
    print(f"  Frequency thresholds: {multi_fft.freq_thresholds.numpy()} Hz")
    
    # Compute multi-resolution FFT for all signals
    print("\nComputing multi-resolution FFT...")
    psd_chirp, freqs = multi_fft(chirp_batch)
    psd_tone, _ = multi_fft(tone_batch)
    psd_burst, _ = multi_fft(burst_batch)
    
    print(f"  PSD shape: {psd_chirp.shape}")
    print(f"  Frequency bins: {len(freqs)}")
    print(f"  Time frames: {psd_chirp.shape[1]}")
    
    # Also compute individual STFTs for comparison
    print("\nComputing individual window STFTs...")
    window_lengths_to_plot = [2048, 512, 64]  # Long, medium, short
    stfts_individual = {}
    
    for wlen in window_lengths_to_plot:
        stft = multi_fft._compute_stft(chirp_batch, wlen)
        psd = torch.abs(stft) ** 2 / wlen
        stfts_individual[wlen] = psd.squeeze(0).T.numpy()  # (freq, time)
    
    # Get window selection map
    print("\nComputing window selection map...")
    freqs_map, window_indices = multi_fft.get_window_selection_map()
    
    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # === Row 0: Input chirp spectrogram ===
    ax = fig.add_subplot(gs[0, :])
    
    # Compute STFT of chirp using longest window for full view
    wlen_full = 2048
    stft_full = multi_fft._compute_stft(chirp_batch, wlen_full)
    stft_magnitude = torch.abs(stft_full).squeeze(0).T.numpy()  # (freq, time)
    
    # Time and frequency vectors
    hop_length_full = multi_fft.hop_lengths[wlen_full]
    n_frames_full = stft_magnitude.shape[1]
    t_frames_full = np.arange(n_frames_full) * hop_length_full / fs
    n_freq_bins_full = stft_magnitude.shape[0]
    freqs_full = np.linspace(0, fs / 2, n_freq_bins_full)
    
    # Plot STFT magnitude (dB scale)
    stft_db = 20 * np.log10(stft_magnitude + 1e-12)
    
    im = ax.pcolormesh(
        t_frames_full, freqs_full / 1000, stft_db,
        shading='auto', cmap='viridis', vmin=stft_db.max() - 120, vmax=stft_db.max()
    )
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Input Signal STFT: Chirp 50 Hz → 15 kHz')
    ax.set_ylim([0, 15])
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # === Row 1: Spectrograms for different window lengths ===
    
    for idx, wlen in enumerate(window_lengths_to_plot):
        ax = fig.add_subplot(gs[1, idx])
        
        # Get time vector for this window
        hop_length = multi_fft.hop_lengths[wlen]
        n_frames = stfts_individual[wlen].shape[1]
        t_frames = np.arange(n_frames) * hop_length / fs
        
        # Get frequency vector for this window
        n_freq_bins = stfts_individual[wlen].shape[0]
        freqs_this_window = np.linspace(0, fs / 2, n_freq_bins)
        
        # Plot spectrogram (dB scale)
        psd_db = 10 * np.log10(stfts_individual[wlen] + 1e-12)
        
        im = ax.pcolormesh(
            t_frames, freqs_this_window / 1000, psd_db,
            shading='auto', cmap='viridis', vmin=psd_db.max() - 60, vmax=psd_db.max()
        )
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title(f'Window: {wlen} samples ({wlen/fs*1000:.1f} ms)')
        ax.set_ylim([0, 15])
        plt.colorbar(im, ax=ax, label='PSD (dB)')
    
    # === Row 2: Analysis plots ===
    
    # Left: Window selection map
    ax = fig.add_subplot(gs[2, 0])
    
    # Create color map for windows
    n_windows = len(multi_fft.window_lengths)
    colors = plt.cm.tab10(np.linspace(0, 1, n_windows))
    
    for i, wlen in enumerate(multi_fft.window_lengths):
        mask = window_indices == i
        if mask.sum() > 0:
            ax.fill_between(
                freqs_map[mask].numpy() / 1000,
                i, i + 0.8,
                color=colors[i],
                alpha=0.7,
                label=f'{wlen} samples'
            )
    
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Window Index')
    ax.set_title('Frequency-Dependent Window Selection')
    ax.set_xlim([0, 16])
    ax.set_ylim([0, n_windows])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines at thresholds
    for thresh in multi_fft.freq_thresholds:
        ax.axvline(thresh / 1000, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Middle: PSD comparison at specific frequencies
    ax = fig.add_subplot(gs[2, 1])
    
    # Extract PSD at specific times (middle of chirp)
    mid_frame = psd_chirp.shape[1] // 2
    psd_mid = psd_chirp[0, mid_frame, :].numpy()
    
    # Plot in dB
    psd_mid_db = 10 * np.log10(psd_mid + 1e-12)
    
    ax.semilogx(freqs.numpy(), psd_mid_db, 'b-', linewidth=2, label='Multi-resolution FFT')
    
    # Find and mark peak frequency
    peak_idx = np.argmax(psd_mid_db)
    peak_freq = freqs[peak_idx].numpy()
    peak_level = psd_mid_db[peak_idx]
    ax.axvline(peak_freq, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Peak: {peak_freq:.1f} Hz')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.set_title(f'PSD at t={mid_frame * multi_fft.hop_lengths[2048] / fs:.2f}s (Chirp)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([50, fs / 2])
    
    # Right: Temporal resolution test with tone burst
    ax = fig.add_subplot(gs[2, 2])
    
    # Average PSD over frequency range 50-150 Hz (where tone burst is)
    freq_mask = (freqs >= 50) & (freqs <= 150)
    psd_burst_avg = psd_burst[0, :, freq_mask].mean(dim=1).numpy()
    
    # Time vector for frames
    hop_length_ref = multi_fft.hop_lengths[2048]  # Use longest window hop
    t_frames = np.arange(len(psd_burst_avg)) * hop_length_ref / fs
    
    # Plot
    ax.plot(t_frames, 10 * np.log10(psd_burst_avg + 1e-12), 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average PSD 50-150 Hz (dB)')
    ax.set_title('Temporal Resolution: 100 Hz Tone Burst')
    ax.grid(True, alpha=0.3)
    
    # Mark the actual burst location
    ax.axvspan(0.5, 0.52, alpha=0.3, color='red', label='Input burst')
    ax.legend()
    
    plt.suptitle('Multi-Resolution FFT Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = Path(__file__).parent.parent.parent / 'test_figures' / 'multi_resolution_fft.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Quantitative analysis
    print("\n" + "=" * 80)
    print("Quantitative Analysis")
    print("=" * 80)
    
    print(f"\nMulti-resolution FFT configuration:")
    print(f"  Number of windows: {len(multi_fft.window_lengths)}")
    print(f"  Window lengths: {multi_fft.window_lengths} samples")
    print(f"  Time resolutions: {[f'{wlen/fs*1000:.1f} ms' for wlen in multi_fft.window_lengths]}")
    print(f"  Frequency resolutions: {[f'{fs/wlen:.2f} Hz' for wlen in multi_fft.window_lengths]}")
    
    print(f"\nFrequency-dependent window selection:")
    for i, (thresh_low, thresh_high, wlen) in enumerate(zip(
        [0] + multi_fft.freq_thresholds.tolist(),
        multi_fft.freq_thresholds.tolist() + [fs / 2],
        multi_fft.window_lengths
    )):
        print(f"  {thresh_low:6.0f} - {thresh_high:6.0f} Hz: {wlen:4d} samples "
              f"({wlen/fs*1000:5.1f} ms, Δf={fs/wlen:6.2f} Hz)")
    
    print(f"\nPSD statistics (chirp signal):")
    psd_chirp_np = psd_chirp.squeeze(0).numpy()
    print(f"  Mean PSD: {10 * np.log10(psd_chirp_np.mean() + 1e-12):.2f} dB")
    print(f"  Max PSD: {10 * np.log10(psd_chirp_np.max()):.2f} dB")
    print(f"  Min PSD: {10 * np.log10(psd_chirp_np.min() + 1e-12):.2f} dB")
    print(f"  Dynamic range: {10 * np.log10(psd_chirp_np.max() / (psd_chirp_np.min() + 1e-12)):.2f} dB")
    print("=" * 80)


if __name__ == "__main__":
    test_multi_resolution_fft()
