"""
Glasberg & Moore (2002) Pipeline - Test Suite

Contents:
1. test_glasberg2002_detailed: Stage-by-stage visualization of complete loudness pipeline
   - Pure tone 1 kHz @ 60 dB SPL
   - Amplitude-modulated tone (1 kHz carrier, 4 Hz modulation)
   - Multi-tone complex (250, 500, 1000, 2000, 4000 Hz)

Structure:
- Multi-resolution FFT → ERB integration → Excitation pattern → Specific loudness
- Short-term loudness (STL) and Long-term loudness (LTL) computation
- Each signal analyzed with waveform, spectrogram, excitation, and loudness patterns

Figures generated:
- glasberg2002_pure_tone.png: Pure tone analysis (4x3 grid)
- glasberg2002_am_tone.png: AM tone analysis (4x3 grid)
- glasberg2002_multi_tone.png: Multi-tone analysis (4x3 grid)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.models.glasberg2002 import Glasberg2002


def generate_signals(fs=32000, duration=2.0):
    """Generate 3 test signals."""
    t = torch.linspace(0, duration, int(fs * duration))
    
    # 1. Pure tone 1 kHz @ 60 dB SPL
    pure_tone = torch.sin(2 * np.pi * 1000 * t) * 10 ** ((60 - 100) / 20)
    
    # 2. AM tone: 1 kHz carrier, 4 Hz modulation, 60 dB average
    carrier = torch.sin(2 * np.pi * 1000 * t)
    modulation = 0.5 * (1 + torch.sin(2 * np.pi * 4 * t))  # 0 to 1
    am_tone = carrier * modulation * 10 ** ((60 - 100) / 20)
    
    # 3. Multi-tone complex @ 60 dB SPL
    freqs_multi = [250, 500, 1000, 2000, 4000]
    multi_tone = torch.zeros_like(t)
    for f in freqs_multi:
        multi_tone += torch.sin(2 * np.pi * f * t)
    multi_tone = multi_tone / len(freqs_multi)
    multi_tone = multi_tone * 10 ** ((60 - 100) / 20)
    
    return pure_tone, am_tone, multi_tone, t


def plot_signal_analysis(signal, signal_name, t, model, output_path):
    """Create detailed stage-by-stage analysis for one signal."""
    
    print(f"\nProcessing {signal_name}...")
    
    # Add batch dimension
    signal_batch = signal.unsqueeze(0)
    
    # Reset model state
    model.reset_state()
    
    # Process through pipeline with all intermediate outputs
    ltl_tensor, stages = model(signal_batch)
    
    # Extract results
    ltl = ltl_tensor.squeeze(0).numpy()
    stl = stages['stl'].squeeze(0).numpy()
    specific_loudness = stages['specific_loudness'].squeeze(0).numpy()
    excitation = stages['excitation'].squeeze(0).numpy()
    erb_excitation = stages['erb_excitation'].squeeze(0).numpy()
    psd = stages['psd'].squeeze(0).numpy()
    freqs_psd = stages['freqs'].numpy()
    erb_freqs = model.get_erb_frequencies().numpy()
    
    # Time vectors
    t_np = t.numpy()
    t_loudness = np.linspace(0, t_np[-1], len(ltl))
    t_psd = np.linspace(0, t_np[-1], psd.shape[0])
    
    # Print debug info
    print(f"  Signal shape: {signal.shape}")
    print(f"  PSD shape: {psd.shape}")
    print(f"  ERB excitation shape: {erb_excitation.shape}")
    print(f"  Specific loudness shape: {specific_loudness.shape}")
    print(f"  STL shape: {stl.shape}, range: [{stl.min():.3f}, {stl.max():.3f}] sone")
    print(f"  LTL shape: {ltl.shape}, range: [{ltl.min():.3f}, {ltl.max():.3f}] sone")
    print(f"  Mean STL: {stl.mean():.3f} sone, Mean LTL: {ltl.mean():.3f} sone")
    
    # Create figure with detailed layout
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # === Row 1: Input signal (time and frequency domain) ===
    
    # Time domain
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_np[:int(0.1*len(t_np))], signal.numpy()[:int(0.1*len(t_np))], 'b-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Input Waveform: {signal_name} (first 100 ms)')
    ax.grid(True, alpha=0.3)
    
    # Spectrogram
    ax = fig.add_subplot(gs[0, 1:])
    
    # Compute and plot spectrogram
    from scipy import signal as sp_signal
    f_spec, t_spec, Sxx = sp_signal.spectrogram(signal.numpy(), fs=32000, 
                                                  nperseg=2048, noverlap=1536)
    
    im = ax.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), 
                      shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spectrogram: {signal_name}')
    ax.set_ylim([0, 8000])
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    # === Row 2: ERB excitation pattern ===
    
    # ERB excitation over time
    ax = fig.add_subplot(gs[1, :2])
    im = ax.pcolormesh(t_psd, erb_freqs, erb_excitation.T, 
                      shading='gouraud', cmap='hot')
    ax.set_ylabel('ERB Center Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('ERB Excitation Pattern (dB SPL)')
    ax.set_yscale('log')
    ax.set_ylim([100, 10000])
    plt.colorbar(im, ax=ax, label='Excitation (dB SPL)')
    
    # Average ERB excitation
    ax = fig.add_subplot(gs[1, 2])
    erb_exc_avg = erb_excitation.mean(axis=0)
    ax.plot(erb_freqs, erb_exc_avg, 'r-', linewidth=2)
    ax.set_xlabel('ERB Center Frequency (Hz)')
    ax.set_ylabel('Excitation (dB SPL)')
    ax.set_title('Time-Averaged ERB Excitation')
    ax.set_xscale('log')
    ax.set_xlim([100, 10000])
    ax.grid(True, alpha=0.3)
    
    # === Row 3: Specific loudness pattern ===
    
    # Specific loudness over time
    ax = fig.add_subplot(gs[2, :2])
    # Clip very small values for better visualization
    specific_loudness_clipped = np.clip(specific_loudness, 1e-6, None)
    im = ax.pcolormesh(t_loudness, erb_freqs, specific_loudness_clipped.T, 
                      shading='gouraud', cmap='plasma')
    ax.set_ylabel('ERB Center Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Specific Loudness (sone/ERB)')
    ax.set_yscale('log')
    ax.set_ylim([100, 10000])
    plt.colorbar(im, ax=ax, label='Specific Loudness (sone/ERB)')
    
    # Average specific loudness
    ax = fig.add_subplot(gs[2, 2])
    spec_loud_avg = specific_loudness.mean(axis=0)
    ax.plot(erb_freqs, spec_loud_avg, 'purple', linewidth=2)
    ax.set_xlabel('ERB Center Frequency (Hz)')
    ax.set_ylabel('Specific Loudness (sone/ERB)')
    ax.set_title('Time-Averaged Specific Loudness')
    ax.set_xscale('log')
    ax.set_xlim([100, 10000])
    ax.grid(True, alpha=0.3)
    
    # === Row 4: Integrated loudness (STL and LTL) ===
    
    # STL vs LTL over time
    ax = fig.add_subplot(gs[3, :2])
    ax.plot(t_loudness, stl, 'b-', linewidth=1.5, alpha=0.7, label='STL (Short-Term)')
    ax.plot(t_loudness, ltl, 'r-', linewidth=2.5, label='LTL (Long-Term)')
    ax.axhline(stl.mean(), color='b', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(ltl.mean(), color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loudness (sone)')
    ax.set_title('Integrated Loudness: STL vs LTL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    
    stats_text = f"""
LOUDNESS STATISTICS

Short-Term Loudness (STL):
  Mean: {stl.mean():.3f} sone
  Std:  {stl.std():.3f} sone
  Min:  {stl.min():.3f} sone
  Max:  {stl.max():.3f} sone

Long-Term Loudness (LTL):
  Mean: {ltl.mean():.3f} sone
  Std:  {ltl.std():.3f} sone
  Min:  {ltl.min():.3f} sone
  Max:  {ltl.max():.3f} sone
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Glasberg & Moore (2002) Pipeline: {signal_name}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def test_glasberg2002_detailed():
    """Test with detailed stage-by-stage visualization."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Glasberg & Moore (2002) Detailed Pipeline Test")
    print("=" * 80)
    
    # Parameters
    fs = 32000
    duration = 2.0
    
    # Generate signals
    print("\nGenerating signals...")
    pure_tone, am_tone, multi_tone, t = generate_signals(fs, duration)
    
    # Initialize model
    print("\nInitializing model...")
    model = Glasberg2002(fs=fs, learnable=False, return_stages=True)
    print(f"  {model}")
    
    # Process each signal
    plot_signal_analysis(pure_tone, "Pure Tone 1 kHz @ 60 dB", t, model,
                        TEST_FIGURES_DIR / 'glasberg2002_pure_tone.png')
    
    plot_signal_analysis(am_tone, "AM Tone (1 kHz, 4 Hz mod)", t, model,
                        TEST_FIGURES_DIR / 'glasberg2002_am_tone.png')
    
    plot_signal_analysis(multi_tone, "Multi-Tone Complex", t, model,
                        TEST_FIGURES_DIR / 'glasberg2002_multi_tone.png')
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_glasberg2002_detailed()
