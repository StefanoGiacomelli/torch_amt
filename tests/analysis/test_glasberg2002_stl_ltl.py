"""
Glasberg & Moore (2002) STL/LTL - Test Suite

Contents:
1. test_glasberg2002_pipeline: Tests STL/LTL extraction and temporal filter properties
   - Short-Term Loudness (STL) spatial integration
   - Long-Term Loudness (LTL) temporal integration
   - Asymmetric attack/release filter (τ_attack=50ms, τ_release=200ms)

Structure:
- Step onset signal (tests attack response)
- Step offset signal (tests release response)
- Amplitude-modulated tone (tests STL tracking)
- Attack/release time constant analysis

Figures generated:
- glasberg2002_stl_ltl.png: 9-panel temporal filter analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.models.glasberg2002 import Glasberg2002


def generate_test_signals(fs=32000, duration=1.5):
    """
    Generate test signals for STL/LTL filter testing.
    
    Returns:
        step_up: Step function (onset) to test attack
        step_down: Step function (offset) to test release
        am_tone: Amplitude-modulated tone to show STL tracking
        t: Time vector
    """
    t = torch.linspace(0, duration, int(fs * duration))
    tone_1khz = torch.sin(2 * np.pi * 1000 * t)
    
    # 1. Step function: silence → 60 dB (tests attack)
    step_up = torch.zeros_like(t)
    step_start = int(0.3 * len(t))  # Start at 0.3s
    step_up[step_start:] = tone_1khz[step_start:] * 10 ** ((60 - 100) / 20)
    
    # 2. Step function: 60 dB → silence (tests release)
    step_down = torch.zeros_like(t)
    step_end = int(0.3 * len(t))  # End at 0.3s
    step_down[:step_end] = tone_1khz[:step_end] * 10 ** ((60 - 100) / 20)
    
    # 3. Amplitude-modulated tone (4 Hz modulation)
    carrier = tone_1khz
    modulation = 0.5 * (1 + torch.sin(2 * np.pi * 4 * t))  # 0 to 1
    am_tone = carrier * modulation * 10 ** ((60 - 100) / 20)
    
    return step_up, step_down, am_tone, t


def test_glasberg2002_pipeline():
    """Test STL/LTL extraction and temporal filter properties."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Glasberg & Moore (2002) STL/LTL Filter Test")
    print("=" * 80)
    
    # Parameters
    fs = 32000
    duration = 1.5
    
    # Generate test signals
    print("\nGenerating test signals...")
    step_up, step_down, am_tone, t = generate_test_signals(fs, duration)
    
    print(f"  Step onset: tests attack response (τ_attack = 50 ms)")
    print(f"  Step offset: tests release response (τ_release = 200 ms)")
    print(f"  AM tone: tests STL tracking (4 Hz modulation)")
    
    # Add batch dimension
    step_up_batch = step_up.unsqueeze(0)
    step_down_batch = step_down.unsqueeze(0)
    am_tone_batch = am_tone.unsqueeze(0)
    
    # Initialize model
    print("\nInitializing Glasberg2002 model...")
    model = Glasberg2002(fs=fs, learnable=False, return_stages=True)
    print(f"  {model}")
    
    # Get temporal integration parameters
    tau_attack, tau_release = model.loudness_integration.get_time_constants()
    print(f"\nTemporal integration parameters:")
    print(f"  τ_attack:  {tau_attack*1000:.1f} ms (fast response to onsets)")
    print(f"  τ_release: {tau_release*1000:.1f} ms (slow response to offsets)")
    print(f"  Asymmetry ratio: {tau_release/tau_attack:.1f}× (release is {tau_release/tau_attack:.1f}× slower)")
    
    # Process signals
    print("\nProcessing signals...")
    
    # Step up (attack)
    print("  Processing step onset (attack)...")
    model.reset_state()
    ltl_up_tensor, stages_up = model(step_up_batch)
    ltl_up = ltl_up_tensor.squeeze(0).numpy()
    stl_up = stages_up['stl'].squeeze(0).numpy()
    
    # Step down (release)
    print("  Processing step offset (release)...")
    model.reset_state()
    ltl_down_tensor, stages_down = model(step_down_batch)
    ltl_down = ltl_down_tensor.squeeze(0).numpy()
    stl_down = stages_down['stl'].squeeze(0).numpy()
    
    # AM tone
    print("  Processing AM tone...")
    model.reset_state()
    ltl_am_tensor, stages_am = model(am_tone_batch)
    ltl_am = ltl_am_tensor.squeeze(0).numpy()
    stl_am = stages_am['stl'].squeeze(0).numpy()
    
    # Time vector for loudness
    t_loudness = np.linspace(0, duration, len(ltl_up))
    
    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # === Top row: Input signal waveforms (spanning 2 columns) ===
    ax = fig.add_subplot(gs[0, :])
    
    # Plot only AM tone waveform
    t_np = t.numpy()
    
    # AM tone waveform
    ax.plot(t_np, am_tone.numpy(), 'b-', linewidth=1, alpha=0.8)
    
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_title('Input Signal Waveform: AM Tone (1 kHz carrier, 4 Hz modulation)', fontsize=11)
    ax.set_xlim([0, duration])
    ax.grid(True, alpha=0.3)
    
    # === Row 2: Input signal spectrograms (3 separate subplots) ===
    
    # Compute spectrograms for all 3 signals
    from scipy import signal as sp_signal
    
    # Step onset spectrogram
    ax1 = fig.add_subplot(gs[1, 0])
    f_spec, t_spec_up, Sxx_up = sp_signal.spectrogram(step_up.numpy(), fs=fs, 
                                                        nperseg=2048, noverlap=1536)
    im1 = ax1.pcolormesh(t_spec_up, f_spec, 10 * np.log10(Sxx_up + 1e-10),
                        shading='gouraud', cmap='viridis', vmin=-80, vmax=0)
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Step Onset Spectrogram (Attack test)', fontsize=10)
    ax1.set_ylim([0, 4000])
    ax1.axvline(0.3, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(0.3, 3500, 'Onset', ha='center', va='top', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Step offset spectrogram
    ax2 = fig.add_subplot(gs[1, 1])
    f_spec, t_spec_down, Sxx_down = sp_signal.spectrogram(step_down.numpy(), fs=fs,
                                                           nperseg=2048, noverlap=1536)
    im2 = ax2.pcolormesh(t_spec_down, f_spec, 10 * np.log10(Sxx_down + 1e-10),
                        shading='gouraud', cmap='viridis', vmin=-80, vmax=0)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Step Offset Spectrogram (Release test)', fontsize=10)
    ax2.set_ylim([0, 4000])
    ax2.axvline(0.3, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.text(0.3, 3500, 'Offset', ha='center', va='top', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # === Row 3, left: Step onset - Attack response ===
    ax = fig.add_subplot(gs[2, 0])
    
    ax.plot(t_loudness, stl_up, 'b-', linewidth=1.5, alpha=0.7, label='STL (instant response)')
    ax.plot(t_loudness, ltl_up, 'r-', linewidth=2.5, label=f'LTL (\u03c4_attack={tau_attack*1000:.0f} ms)')
    ax.axvline(0.3, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(0.3, ax.get_ylim()[1]*0.95, 'Onset', ha='center', va='top', fontsize=9)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loudness (sone)')
    ax.set_title('Step Onset: Fast Attack Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, duration])
    

    # === Row 3, right: Step offset - Release response ===
    ax = fig.add_subplot(gs[2, 1])
    
    ax.plot(t_loudness, stl_down, 'b-', linewidth=1.5, alpha=0.7, label='STL (instant response)')
    ax.plot(t_loudness, ltl_down, 'r-', linewidth=2.5, label=f'LTL (\u03c4_release={tau_release*1000:.0f} ms)')
    ax.axvline(0.3, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(0.3, ax.get_ylim()[1]*0.95, 'Offset', ha='center', va='top', fontsize=9)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loudness (sone)')
    ax.set_title('Step Offset: Slow Release Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, duration])
    

    # === Row 4, left: AM tone - STL tracking ===
    ax = fig.add_subplot(gs[3, 0])
    
    ax.plot(t_loudness, stl_am, 'b-', linewidth=1.5, alpha=0.7, label='STL (tracks modulation)')
    ax.plot(t_loudness, ltl_am, 'r-', linewidth=2.5, label='LTL (smoothed)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loudness (sone)')
    ax.set_title('AM Tone: STL Tracks Fast, LTL Smooths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, duration])
    
    # Add annotation
    modulation_stl = (stl_am.max() - stl_am.min()) / stl_am.mean() * 100
    modulation_ltl = (ltl_am.max() - ltl_am.min()) / ltl_am.mean() * 100
    ax.text(0.98, 0.95, f'STL modulation: {modulation_stl:.1f}%\nLTL modulation: {modulation_ltl:.1f}%\nSmoothing: {modulation_stl/modulation_ltl:.1f}\u00d7',
           transform=ax.transAxes, ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Row 4, right: Attack vs Release comparison ===
    ax = fig.add_subplot(gs[3, 1])
    
    # Find step onset/offset in loudness time
    step_idx = int(0.3 / duration * len(t_loudness))
    
    # Normalize attack and release portions
    # Attack: from step onset
    ltl_attack = ltl_up[step_idx:]
    ltl_attack_norm = (ltl_attack - ltl_attack[0]) / (ltl_attack.max() - ltl_attack[0])
    t_attack = np.linspace(0, len(ltl_attack) * (duration/len(t_loudness)), len(ltl_attack))
    
    # Release: from step offset
    ltl_release = ltl_down[step_idx:]
    ltl_release_norm = (ltl_release - ltl_release[-1]) / (ltl_release[0] - ltl_release[-1])
    t_release = np.linspace(0, len(ltl_release) * (duration/len(t_loudness)), len(ltl_release))
    
    ax.plot(t_attack, ltl_attack_norm, 'g-', linewidth=2.5, label=f'Attack (\u03c4={tau_attack*1000:.0f} ms)', alpha=0.8)
    ax.plot(t_release, ltl_release_norm, 'orange', linewidth=2.5, label=f'Release (\u03c4={tau_release*1000:.0f} ms)', alpha=0.8)
    
    # Mark time constants
    ax.axvline(tau_attack, color='g', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(tau_release, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(1 - 1/np.e, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(tau_attack, 0.63, '\u03c4_attack', ha='center', va='bottom', fontsize=8, color='g')
    ax.text(tau_release, 0.63, '\u03c4_release', ha='center', va='bottom', fontsize=8, color='orange')
    
    ax.set_xlabel('Time from transition (s)')
    ax.set_ylabel('Normalized LTL Response')
    ax.set_title('Attack vs Release Asymmetry')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.5])
    
    # Add annotation
    ax.text(0.98, 0.05, f'Asymmetry ratio: {tau_release/tau_attack:.1f}\u00d7\nRelease is {tau_release/tau_attack:.1f}\u00d7 slower than attack',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Glasberg & Moore (2002) STL/LTL Temporal Filter Analysis', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = TEST_FIGURES_DIR / 'glasberg2002_stl_ltl.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Quantitative analysis
    print("\n" + "=" * 80)
    print("Quantitative Analysis: Temporal Filter Properties")
    print("=" * 80)
    
    print(f"\nTemporal integration time constants:")
    print(f"  τ_attack: {tau_attack*1000:.1f} ms")
    print(f"  τ_release: {tau_release*1000:.1f} ms")
    print(f"  Asymmetry ratio: {tau_release/tau_attack:.1f}×")
    
    print(f"\nStep onset (attack response):")
    # Find 10-90% rise time
    idx_onset = int(0.3 / duration * len(ltl_up))
    ltl_after_onset = ltl_up[idx_onset:]
    if len(ltl_after_onset) > 0:
        ltl_10 = ltl_after_onset[0] + 0.1 * (ltl_after_onset.max() - ltl_after_onset[0])
        ltl_90 = ltl_after_onset[0] + 0.9 * (ltl_after_onset.max() - ltl_after_onset[0])
        idx_10 = np.argmax(ltl_after_onset > ltl_10)
        idx_90 = np.argmax(ltl_after_onset > ltl_90)
        rise_time = (idx_90 - idx_10) * (duration / len(t_loudness))
        print(f"  10-90% rise time: {rise_time*1000:.1f} ms")
        print(f"  Expected ~τ_attack: {tau_attack*1000:.1f} ms")
    
    print(f"\nStep offset (release response):")
    # Find 90-10% fall time
    ltl_after_offset = ltl_down[idx_onset:]
    if len(ltl_after_offset) > 0:
        ltl_90_down = ltl_after_offset[0] - 0.1 * (ltl_after_offset[0] - ltl_after_offset.min())
        ltl_10_down = ltl_after_offset[0] - 0.9 * (ltl_after_offset[0] - ltl_after_offset.min())
        idx_90_down = np.argmax(ltl_after_offset < ltl_90_down)
        idx_10_down = np.argmax(ltl_after_offset < ltl_10_down)
        if idx_10_down > idx_90_down:
            fall_time = (idx_10_down - idx_90_down) * (duration / len(t_loudness))
            print(f"  90-10% fall time: {fall_time*1000:.1f} ms")
            print(f"  Expected ~τ_release: {tau_release*1000:.1f} ms")
    
    print(f"\nAM tone (4 Hz modulation):")
    modulation_stl = (stl_am.max() - stl_am.min()) / stl_am.mean() * 100
    modulation_ltl = (ltl_am.max() - ltl_am.min()) / ltl_am.mean() * 100
    smoothing_factor = modulation_stl / modulation_ltl if modulation_ltl > 0 else 0
    print(f"  STL modulation depth: {modulation_stl:.1f}%")
    print(f"  LTL modulation depth: {modulation_ltl:.1f}%")
    print(f"  Smoothing factor: {smoothing_factor:.1f}×")
    print(f"  Mean STL: {stl_am.mean():.2f} sone")
    print(f"  Mean LTL: {ltl_am.mean():.2f} sone")
    
    print("=" * 80)


if __name__ == "__main__":
    test_glasberg2002_pipeline()
