"""
Comparison Test: Glasberg2002 vs Moore2016

This test compares the monaural Glasberg2002 model with the binaural Moore2016 model.

Key differences:
1. **Architecture**:
   - Glasberg2002: Monaural, single temporal integration stage
   - Moore2016: Binaural, dual temporal integration (STL + LTL), binaural inhibition

2. **Processing stages**:
   - Glasberg2002: Spectrum → ERB integration → Excitation → Specific loudness → Temporal integration
   - Moore2016: Spectrum (sparse) → Excitation → Specific loudness → STL AGC → Binaural inhibition → LTL AGC

3. **Expected behavior**:
   - For diotic stimuli: Moore2016 ≈ 2x Glasberg2002 (binaural summation)
   - For dichotic stimuli: Moore2016 shows binaural inhibition effects
   - Temporal dynamics differ due to dual AGC in Moore2016

Figures generated:
- comparison_diotic_tone.png: Side-by-side comparison for diotic stimulus
- comparison_dichotic_tone.png: Dichotic stimulus (Moore2016 only shows inhibition)
- comparison_am_tone.png: Temporal dynamics comparison
- comparison_summary.png: 4-panel summary of key differences
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from torch_amt.models.glasberg2002 import Glasberg2002
from torch_amt.models.moore2016 import Moore2016


TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'


def compare_diotic_tone():
    """Compare Glasberg2002 vs Moore2016 for diotic 1 kHz tone."""
    print("\n" + "="*80)
    print("COMPARISON 1: DIOTIC TONE (1 kHz, 60 dB SPL)")
    print("="*80)
    
    fs = 32000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    
    # 1 kHz tone at 60 dB SPL
    tone = np.sin(2 * np.pi * 1000 * t) * 0.02
    audio_mono = torch.from_numpy(tone).float().unsqueeze(0)  # (1, n_samples)
    audio_stereo = torch.stack([audio_mono[0], audio_mono[0]], dim=0).unsqueeze(0)  # (1, 2, n_samples)
    
    # Glasberg2002
    glasberg = Glasberg2002(fs=32000)
    results_g = glasberg(audio_mono, return_intermediate=True)
    sLoud_g = results_g['stl']
    lLoud_g = results_g['ltl']
    
    # Moore2016
    moore = Moore2016(fs=32000)
    sLoud_m, lLoud_m, mLoud_m = moore(audio_stereo)
    
    ratio_stl = sLoud_m.mean() / sLoud_g.mean()
    ratio_ltl = lLoud_m.mean() / lLoud_g.mean()
    
    print(f"\nGlasberg2002 (monaural):")
    print(f"  STL mean: {sLoud_g.mean():.3f} sones")
    print(f"  LTL mean: {lLoud_g.mean():.3f} sones")
    
    print(f"\nMoore2016 (binaural):")
    print(f"  STL mean: {sLoud_m.mean():.3f} sones")
    print(f"  LTL mean: {lLoud_m.mean():.3f} sones")
    print(f"  Max loudness: {mLoud_m.item():.3f} sones")
    
    print(f"\nBinaural summation:")
    print(f"  STL ratio (M16/G02): {ratio_stl:.2f}x")
    print(f"  LTL ratio (M16/G02): {ratio_ltl:.2f}x")
    print(f"  Expected: ~2x for diotic stimuli")
    
    # === Visualization ===
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    t_out_g = np.arange(sLoud_g.shape[1]) * 0.001
    t_out_m = np.arange(sLoud_m.shape[1]) * 0.001
    
    # Row 1: Input waveform
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, tone, 'k-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (Pa)')
    ax.set_title('Input: Diotic Tone (1 kHz, 60 dB SPL)')
    ax.grid(True, alpha=0.3)
    
    # Row 2: STL comparison
    ax_g = fig.add_subplot(gs[1, 0])
    ax_g.plot(t_out_g, sLoud_g[0].detach().numpy(), 'b-', linewidth=1.5)
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylabel('STL (sones)')
    ax_g.set_title(f'Glasberg2002 (Monaural)\nMean STL: {sLoud_g.mean():.2f} sones')
    ax_g.grid(True, alpha=0.3)
    
    ax_m = fig.add_subplot(gs[1, 1])
    ax_m.plot(t_out_m, sLoud_m[0].detach().numpy(), 'r-', linewidth=1.5)
    ax_m.set_xlabel('Time (s)')
    ax_m.set_ylabel('STL (sones)')
    ax_m.set_title(f'Moore2016 (Binaural)\nMean STL: {sLoud_m.mean():.2f} sones ({ratio_stl:.2f}x)')
    ax_m.grid(True, alpha=0.3)
    
    # Row 3: LTL comparison
    ax_g = fig.add_subplot(gs[2, 0])
    ax_g.plot(t_out_g, lLoud_g[0].detach().numpy(), 'b-', linewidth=1.5)
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylabel('LTL (sones)')
    ax_g.set_title(f'Glasberg2002 (Monaural)\nMean LTL: {lLoud_g.mean():.2f} sones')
    ax_g.grid(True, alpha=0.3)
    
    ax_m = fig.add_subplot(gs[2, 1])
    ax_m.plot(t_out_m, lLoud_m[0].detach().numpy(), 'r-', linewidth=1.5)
    ax_m.set_xlabel('Time (s)')
    ax_m.set_ylabel('LTL (sones)')
    ax_m.set_title(f'Moore2016 (Binaural)\nMean LTL: {lLoud_m.mean():.2f} sones ({ratio_ltl:.2f}x)')
    ax_m.grid(True, alpha=0.3)
    
    plt.suptitle('Glasberg2002 vs Moore2016: Diotic Tone', fontsize=14, fontweight='bold')
    
    TEST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = TEST_FIGURES_DIR / 'comparison_diotic_tone.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()


def compare_am_tone():
    """Compare temporal dynamics with AM tone."""
    print("\n" + "="*80)
    print("COMPARISON 2: AM TONE (temporal dynamics)")
    print("="*80)
    
    fs = 32000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # AM tone: 1 kHz carrier, 4 Hz modulation
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    am_tone = carrier * modulator * 0.02
    
    audio_mono = torch.from_numpy(am_tone).float().unsqueeze(0)
    audio_stereo = torch.stack([audio_mono[0], audio_mono[0]], dim=0).unsqueeze(0)
    
    # Glasberg2002
    glasberg = Glasberg2002(fs=32000)
    results_g = glasberg(audio_mono, return_intermediate=True)
    sLoud_g = results_g['stl']
    lLoud_g = results_g['ltl']
    
    # Moore2016
    moore = Moore2016(fs=32000)
    sLoud_m, lLoud_m, mLoud_m = moore(audio_stereo)
    
    print(f"\nGlasberg2002: STL mean = {sLoud_g.mean():.3f}, LTL mean = {lLoud_g.mean():.3f}")
    print(f"Moore2016: STL mean = {sLoud_m.mean():.3f}, LTL mean = {lLoud_m.mean():.3f}")
    
    # === Visualization ===
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    t_out_g = np.arange(sLoud_g.shape[1]) * 0.001
    t_out_m = np.arange(sLoud_m.shape[1]) * 0.001
    
    # Row 1: Input waveform
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, am_tone, 'k-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (Pa)')
    ax.set_title('Input: AM Tone (1 kHz carrier, 4 Hz modulation)')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Glasberg2002
    ax_g = fig.add_subplot(gs[1, 0])
    ax_g.plot(t_out_g, sLoud_g[0].detach().numpy(), 'b-', label='STL', linewidth=1.5)
    ax_g.plot(t_out_g, lLoud_g[0].detach().numpy(), 'r-', label='LTL', linewidth=1.5)
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylabel('Loudness (sones)')
    ax_g.set_title('Glasberg2002 (Monaural, Single Temporal Stage)')
    ax_g.legend()
    ax_g.grid(True, alpha=0.3)
    
    # Row 2: Moore2016
    ax_m = fig.add_subplot(gs[1, 1])
    ax_m.plot(t_out_m, sLoud_m[0].detach().numpy(), 'b-', label='STL', linewidth=1.5)
    ax_m.plot(t_out_m, lLoud_m[0].detach().numpy(), 'r-', label='LTL', linewidth=1.5)
    ax_m.set_xlabel('Time (s)')
    ax_m.set_ylabel('Loudness (sones)')
    ax_m.set_title('Moore2016 (Binaural, Dual Temporal Stage)')
    ax_m.legend()
    ax_m.grid(True, alpha=0.3)
    
    # Row 3: Overlay comparison
    ax = fig.add_subplot(gs[2, :])
    ax.plot(t_out_g, sLoud_g[0].detach().numpy(), 'b-', label='Glasberg STL', linewidth=1.5, alpha=0.7)
    ax.plot(t_out_g, lLoud_g[0].detach().numpy(), 'b--', label='Glasberg LTL', linewidth=1.5, alpha=0.7)
    ax.plot(t_out_m, sLoud_m[0].detach().numpy(), 'r-', label='Moore STL', linewidth=1.5, alpha=0.7)
    ax.plot(t_out_m, lLoud_m[0].detach().numpy(), 'r--', label='Moore LTL', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loudness (sones)')
    ax.set_title('Temporal Dynamics Comparison')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Glasberg2002 vs Moore2016: Temporal Dynamics', fontsize=14, fontweight='bold')
    
    fig_path = TEST_FIGURES_DIR / 'comparison_am_tone.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()


def compare_dichotic():
    """Compare Moore2016 binaural inhibition (Glasberg2002 is monaural, can't do dichotic)."""
    print("\n" + "="*80)
    print("COMPARISON 3: DICHOTIC TONE (Moore2016 only - binaural inhibition)")
    print("="*80)
    
    fs = 32000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    
    tone_left = np.sin(2 * np.pi * 1000 * t) * 0.02   # 60 dB
    tone_right = np.sin(2 * np.pi * 1000 * t) * 0.006  # 50 dB
    
    audio_left = torch.from_numpy(tone_left).float()
    audio_right = torch.from_numpy(tone_right).float()
    
    # Moore2016 binaural
    audio_stereo = torch.stack([audio_left, audio_right], dim=0).unsqueeze(0)
    moore = Moore2016(fs=32000)
    sLoud_dichotic, lLoud_dichotic, mLoud_dichotic = moore(audio_stereo)
    
    # Moore2016 diotic (for comparison)
    audio_diotic = torch.stack([audio_left, audio_left], dim=0).unsqueeze(0)
    sLoud_diotic, lLoud_diotic, mLoud_diotic = moore(audio_diotic)
    
    # Glasberg2002 on left channel only
    glasberg = Glasberg2002(fs=32000)
    results_g = glasberg(audio_left.unsqueeze(0), return_intermediate=True)
    sLoud_g = results_g['stl']
    lLoud_g = results_g['ltl']
    
    inhibition_ratio = sLoud_dichotic.mean() / sLoud_diotic.mean()
    
    print(f"\nGlasberg2002 (left channel only):")
    print(f"  STL mean: {sLoud_g.mean():.3f} sones")
    
    print(f"\nMoore2016 diotic (60 dB both ears):")
    print(f"  STL mean: {sLoud_diotic.mean():.3f} sones")
    
    print(f"\nMoore2016 dichotic (60 dB L, 50 dB R):")
    print(f"  STL mean: {sLoud_dichotic.mean():.3f} sones")
    print(f"  Inhibition ratio: {inhibition_ratio:.2f}x")
    print(f"  (Dichotic < Diotic due to binaural inhibition)")
    
    # === Visualization ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_out_g = np.arange(sLoud_g.shape[1]) * 0.001
    t_out_m = np.arange(sLoud_dichotic.shape[1]) * 0.001
    
    # Input waveforms
    axes[0, 0].plot(t[:1000], audio_left.numpy()[:1000], 'b-', label='Left (60 dB)', linewidth=1.0)
    axes[0, 0].plot(t[:1000], audio_right.numpy()[:1000], 'r-', label='Right (50 dB)', linewidth=1.0, alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (Pa)')
    axes[0, 0].set_title('Input: Dichotic Tone')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Glasberg2002 (monaural, left only)
    axes[0, 1].plot(t_out_g, sLoud_g[0].detach().numpy(), 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('STL (sones)')
    axes[0, 1].set_title(f'Glasberg2002 (Left Ear Only)\nMean: {sLoud_g.mean():.2f} sones')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Moore2016 diotic vs dichotic
    axes[1, 0].plot(t_out_m, sLoud_diotic[0].detach().numpy(), 'g-', label='Diotic (60 dB)', linewidth=1.5)
    axes[1, 0].plot(t_out_m, sLoud_dichotic[0].detach().numpy(), 'r-', label='Dichotic (60/50 dB)', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('STL (sones)')
    axes[1, 0].set_title('Moore2016: Diotic vs Dichotic')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Inhibition effect
    axes[1, 1].bar(['Glasberg2002\n(Monaural)', 'Moore2016\nDiotic', 'Moore2016\nDichotic'],
                   [sLoud_g.mean().item(), sLoud_diotic.mean().item(), sLoud_dichotic.mean().item()],
                   color=['blue', 'green', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Mean STL (sones)')
    axes[1, 1].set_title('Binaural Inhibition Effect')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Moore2016 Binaural Inhibition (vs Glasberg2002 Monaural)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = TEST_FIGURES_DIR / 'comparison_dichotic.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()


def compare_summary():
    """Generate summary comparison figure."""
    print("\n" + "="*80)
    print("COMPARISON 4: SUMMARY (4-panel)")
    print("="*80)
    
    fs = 32000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    
    # Pure tone
    tone = np.sin(2 * np.pi * 1000 * t) * 0.02
    audio_mono = torch.from_numpy(tone).float().unsqueeze(0)
    audio_stereo = torch.stack([audio_mono[0], audio_mono[0]], dim=0).unsqueeze(0)
    
    glasberg = Glasberg2002(fs=32000)
    moore = Moore2016(fs=32000)
    
    results_g = glasberg(audio_mono, return_intermediate=True)
    sLoud_g = results_g['stl']
    lLoud_g = results_g['ltl']
    sLoud_m, lLoud_m, mLoud_m = moore(audio_stereo)
    
    # === Visualization ===
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    t_out_g = np.arange(sLoud_g.shape[1]) * 0.001
    t_out_m = np.arange(sLoud_m.shape[1]) * 0.001
    
    # Panel 1: Architecture
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    arch_text = """
ARCHITECTURE COMPARISON

Glasberg2002 (Monaural):
1. Multi-resolution FFT
2. ERB integration
3. Excitation pattern
4. Specific loudness
5. Single temporal stage

Moore2016 (Binaural):
1. Sparse spectrum (6 windows)
2. Roex excitation pattern
3. Specific loudness
4. STL AGC (per ear)
5. Binaural inhibition
6. LTL AGC (per ear)
    """
    ax.text(0.1, 0.9, arch_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Panel 2: Binaural summation ratio
    ax = fig.add_subplot(gs[0, 1])
    ratio_stl = sLoud_m.mean() / sLoud_g.mean()
    ratio_ltl = lLoud_m.mean() / lLoud_g.mean()
    
    ax.bar(['STL', 'LTL'], [ratio_stl.item(), ratio_ltl.item()], color=['blue', 'red'], alpha=0.7)
    ax.axhline(2.0, color='k', linestyle='--', label='Expected 2x')
    ax.set_ylabel('Moore2016 / Glasberg2002 Ratio')
    ax.set_title('Binaural Summation for Diotic Tone')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: STL comparison
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_out_g, sLoud_g[0].detach().numpy(), 'b-', label='Glasberg2002', linewidth=2)
    ax.plot(t_out_m, sLoud_m[0].detach().numpy(), 'r-', label='Moore2016', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('STL (sones)')
    ax.set_title('Short-Term Loudness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: LTL comparison
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_out_g, lLoud_g[0].detach().numpy(), 'b-', label='Glasberg2002', linewidth=2)
    ax.plot(t_out_m, lLoud_m[0].detach().numpy(), 'r-', label='Moore2016', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LTL (sones)')
    ax.set_title('Long-Term Loudness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Glasberg2002 vs Moore2016: Summary Comparison', fontsize=14, fontweight='bold')
    
    fig_path = TEST_FIGURES_DIR / 'comparison_summary.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()


if __name__ == '__main__':
    """Run all comparison tests."""
    print("\n" + "="*80)
    print("GLASBERG2002 vs MOORE2016 - COMPARISON TESTS")
    print("="*80)
    
    compare_diotic_tone()
    compare_am_tone()
    compare_dichotic()
    compare_summary()
    
    print("\n" + "="*80)
    print("ALL COMPARISON TESTS COMPLETED ✓")
    print("="*80)
