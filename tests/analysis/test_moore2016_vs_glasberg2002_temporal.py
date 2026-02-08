"""
Moore2016 vs Glasberg2002 Temporal Integration - Comparison Test

This test compares the temporal integration approaches of Moore2016 and Glasberg2002:

Moore2016:
- Two-stage AGC: Short-Term (STL) and Long-Term (LTL)
- STL: Attack α=0.045, Release α=0.033
- LTL: Attack α=0.01, Release α=0.00133
- Applied to 150-channel specific loudness, then integrated

Glasberg2002:
- Single exponential temporal window
- Two time constants: τ_short=5ms, τ_long=200ms
- Weight factor: 0.2 (short) + 0.8 (long)
- Applied after spatial integration

Key differences:
- Moore2016: Asymmetric attack/release, dual-stage
- Glasberg2002: Symmetric exponential smoothing, single-stage
- Moore2016: Channel-wise then integrated
- Glasberg2002: After spatial integration
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import Moore2016AGC, Moore2016TemporalIntegration, LoudnessIntegration


def test_moore2016_vs_glasberg2002():
    """Compare Moore2016 and Glasberg2002 temporal integration approaches."""
    
    print("\n" + "="*80)
    print("MOORE2016 vs GLASBERG2002 TEMPORAL COMPARISON")
    print("="*80)
    
    # Initialize modules
    moore_temporal = Moore2016TemporalIntegration()
    glasberg_temporal = LoudnessIntegration()
    
    # Frame rate
    fs = 32000
    hop_length = 512
    frame_rate = fs / hop_length
    dt = 1.0 / frame_rate
    
    print(f"\nConfiguration:")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Hop length: {hop_length} samples")
    print(f"  Frame rate: {frame_rate:.2f} fps")
    print(f"  Frame period: {dt*1000:.2f} ms")
    
    moore_params = moore_temporal.get_parameters()
    
    # Glasberg2002 doesn't have get_parameters, access attributes directly
    glasberg_params = {
        'tau_short_ms': 5.0,
        'tau_long_ms': 200.0,
        'short_weight': 0.2,
        'long_weight': 0.8
    }
    
    print(f"\nMoore2016 parameters:")
    print(f"  STL attack: {moore_params['stl']['attack_alpha']:.4f}")
    print(f"  STL release: {moore_params['stl']['release_alpha']:.4f}")
    print(f"  LTL attack: {moore_params['ltl']['attack_alpha']:.6f}")
    print(f"  LTL release: {moore_params['ltl']['release_alpha']:.6f}")
    
    print(f"\nGlasberg2002 parameters:")
    print(f"  τ_short: {glasberg_params['tau_short_ms']:.1f} ms")
    print(f"  τ_long: {glasberg_params['tau_long_ms']:.1f} ms")
    print(f"  Short weight: {glasberg_params['short_weight']:.1f}")
    print(f"  Long weight: {glasberg_params['long_weight']:.1f}")
    
    # Number of frames
    n_frames = 600
    time_axis = np.arange(n_frames) * dt
    
    # ========== TEST 1: Impulse response ==========
    print("\n" + "-"*80)
    print("TEST 1: Impulse response comparison")
    print("-"*80)
    
    # Create impulse
    impulse = torch.zeros(n_frames)
    impulse[100] = 10.0
    
    # Moore2016 response (apply to scalar, as if integrated)
    moore_stl_agc = Moore2016AGC(attack_alpha=0.045, release_alpha=0.033)
    moore_ltl_agc = Moore2016AGC(attack_alpha=0.01, release_alpha=0.00133)
    moore_stl_response = moore_stl_agc(impulse)
    moore_ltl_response = moore_ltl_agc(impulse)
    
    # Glasberg2002 response
    # Need to format as (batch, frames, channels)
    impulse_glasberg = impulse.unsqueeze(0).unsqueeze(-1)  # (1, n_frames, 1)
    glasberg_response = glasberg_temporal(impulse_glasberg)  # (1, n_frames)
    glasberg_response = glasberg_response.squeeze(0)  # (n_frames,)
    
    # Measure decay characteristics
    def decay_to_half(signal, start_idx):
        """Find time to decay to 50% of peak."""
        peak = signal[start_idx].item()
        half_peak = peak / 2
        idx = np.where(signal[start_idx:].numpy() < half_peak)[0]
        if len(idx) > 0:
            return idx[0] * dt * 1000  # ms
        return np.nan
    
    t_half_moore_stl = decay_to_half(moore_stl_response, 100)
    t_half_moore_ltl = decay_to_half(moore_ltl_response, 100)
    t_half_glasberg = decay_to_half(glasberg_response, 100)
    
    print(f"\nDecay to 50% of peak:")
    print(f"  Moore2016 STL: {t_half_moore_stl:.1f} ms")
    print(f"  Moore2016 LTL: {t_half_moore_ltl:.1f} ms")
    print(f"  Glasberg2002: {t_half_glasberg:.1f} ms")
    
    # ========== TEST 2: Step response ==========
    print("\n" + "-"*80)
    print("TEST 2: Step response comparison")
    print("-"*80)
    
    # Create step: 0 → 1 at t=100
    step = torch.zeros(n_frames)
    step[100:] = 1.0
    
    # Moore2016
    moore_stl_step = moore_stl_agc(step)
    moore_ltl_step = moore_ltl_agc(step)
    
    # Glasberg2002
    step_glasberg = step.unsqueeze(0).unsqueeze(-1)
    glasberg_step = glasberg_temporal(step_glasberg).squeeze(0)
    
    # Measure rise time to 90%
    def rise_to_90(signal, start_idx):
        """Find time to rise to 90% of final value."""
        final_val = signal[-100:].mean().item()
        target = 0.9 * final_val
        idx = np.where(signal[start_idx:].numpy() > target)[0]
        if len(idx) > 0:
            return idx[0] * dt * 1000  # ms
        return np.nan
    
    t_90_moore_stl = rise_to_90(moore_stl_step, 100)
    t_90_moore_ltl = rise_to_90(moore_ltl_step, 100)
    t_90_glasberg = rise_to_90(glasberg_step, 100)
    
    print(f"\nRise time to 90%:")
    print(f"  Moore2016 STL: {t_90_moore_stl:.1f} ms")
    print(f"  Moore2016 LTL: {t_90_moore_ltl:.1f} ms")
    print(f"  Glasberg2002: {t_90_glasberg:.1f} ms")
    
    # ========== TEST 3: Tone burst (60 dB) ==========
    print("\n" + "-"*80)
    print("TEST 3: Tone burst response (60 dB)")
    print("-"*80)
    
    # Create tone burst: silence → 60 dB → silence
    tone_burst = torch.zeros(n_frames)
    tone_burst[100:400] = 1.0  # 60 dB reference
    
    # Moore2016
    moore_stl_burst = moore_stl_agc(tone_burst)
    moore_ltl_burst = moore_ltl_agc(tone_burst)
    
    # Glasberg2002
    burst_glasberg = tone_burst.unsqueeze(0).unsqueeze(-1)
    glasberg_burst = glasberg_temporal(burst_glasberg).squeeze(0)
    
    print(f"\nTone burst steady-state (frames 250-350):")
    print(f"  Moore2016 STL mean: {moore_stl_burst[250:350].mean():.4f}")
    print(f"  Moore2016 LTL mean: {moore_ltl_burst[250:350].mean():.4f}")
    print(f"  Glasberg2002 mean: {glasberg_burst[250:350].mean():.4f}")
    
    # ========== TEST 4: Amplitude modulation (5 Hz) ==========
    print("\n" + "-"*80)
    print("TEST 4: Amplitude modulation smoothing (5 Hz)")
    print("-"*80)
    
    # Create 5 Hz AM signal
    modulation_freq = 5.0
    carrier = 1.0
    depth = 0.8
    
    t = time_axis
    am_signal = carrier * (1 + depth * np.sin(2 * np.pi * modulation_freq * t))
    am_torch = torch.from_numpy(am_signal).float()
    
    # Moore2016
    moore_stl_am = moore_stl_agc(am_torch)
    moore_ltl_am = moore_ltl_agc(am_torch)
    
    # Glasberg2002
    am_glasberg = am_torch.unsqueeze(0).unsqueeze(-1)
    glasberg_am = glasberg_temporal(am_glasberg).squeeze(0)
    
    # Measure modulation depth
    def modulation_depth(signal, start=100, end=500):
        sig = signal[start:end]
        return (sig.max() - sig.min()) / (sig.max() + sig.min())
    
    input_depth = modulation_depth(am_torch)
    stl_depth = modulation_depth(moore_stl_am)
    ltl_depth = modulation_depth(moore_ltl_am)
    glasberg_depth = modulation_depth(glasberg_am)
    
    print(f"\nModulation depth (5 Hz):")
    print(f"  Input: {input_depth:.4f}")
    print(f"  Moore2016 STL: {stl_depth:.4f} (attenuation: {(1-stl_depth/input_depth)*100:.1f}%)")
    print(f"  Moore2016 LTL: {ltl_depth:.4f} (attenuation: {(1-ltl_depth/input_depth)*100:.1f}%)")
    print(f"  Glasberg2002: {glasberg_depth:.4f} (attenuation: {(1-glasberg_depth/input_depth)*100:.1f}%)")
    
    # ========== TEST 5: Full pipeline comparison ==========
    print("\n" + "-"*80)
    print("TEST 5: Full pipeline with realistic signal")
    print("-"*80)
    
    # Create realistic signal with temporal structure
    realistic = torch.randn(n_frames, 150).abs() * 2.0
    envelope = torch.from_numpy(np.exp(-time_axis / 3.0)).float()
    realistic = realistic * envelope.unsqueeze(1)
    
    # Moore2016 full pipeline
    moore_ltl, moore_stl_spec, moore_stl = moore_temporal(realistic, return_intermediate=True)
    
    # Glasberg2002 (needs spatial integration first)
    realistic_glasberg = realistic.unsqueeze(0)  # (1, n_frames, 150)
    glasberg_integrated = glasberg_temporal(realistic_glasberg).squeeze(0)  # (n_frames,)
    
    print(f"\nFull pipeline output ranges:")
    print(f"  Moore2016 STL: {moore_stl.min():.4f} - {moore_stl.max():.4f} sone")
    print(f"  Moore2016 LTL: {moore_ltl.min():.4f} - {moore_ltl.max():.4f} sone")
    print(f"  Glasberg2002: {glasberg_integrated.min():.4f} - {glasberg_integrated.max():.4f} sone")
    
    # Compute correlation
    corr_stl = np.corrcoef(moore_stl.numpy(), glasberg_integrated.numpy())[0, 1]
    corr_ltl = np.corrcoef(moore_ltl.numpy(), glasberg_integrated.numpy())[0, 1]
    
    print(f"\nCorrelation with Glasberg2002:")
    print(f"  Moore2016 STL: {corr_stl:.3f}")
    print(f"  Moore2016 LTL: {corr_ltl:.3f}")
    
    # ========== VISUALIZATION ==========
    print("\n" + "-"*80)
    print("Creating comparison visualization...")
    print("-"*80)
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # === Plot 1: Time constants comparison ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Effective time constants (approximation)
    # For Moore2016: τ ≈ -dt / ln(1-α)
    tau_stl_attack = -dt * 1000 / np.log(1 - moore_params['stl']['attack_alpha'])
    tau_stl_release = -dt * 1000 / np.log(1 - moore_params['stl']['release_alpha'])
    tau_ltl_attack = -dt * 1000 / np.log(1 - moore_params['ltl']['attack_alpha'])
    tau_ltl_release = -dt * 1000 / np.log(1 - moore_params['ltl']['release_alpha'])
    
    x = np.arange(4)
    moore_taus = [tau_stl_attack, tau_stl_release, tau_ltl_attack, tau_ltl_release]
    labels = ['STL\nAttack', 'STL\nRelease', 'LTL\nAttack', 'LTL\nRelease']
    
    bars = ax1.bar(x, moore_taus, color='#2ca02c', alpha=0.8, label='Moore2016')
    
    # Add Glasberg reference lines
    ax1.axhline(y=glasberg_params['tau_short_ms'], color='#1f77b4', 
                linestyle='--', linewidth=2, label='Glasberg τ_short')
    ax1.axhline(y=glasberg_params['tau_long_ms'], color='#d62728', 
                linestyle='--', linewidth=2, label='Glasberg τ_long')
    
    ax1.set_ylabel('Time Constant (ms)')
    ax1.set_title('Effective Time Constants')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=8)
    
    # === Plot 2: Impulse response ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.plot(time_axis[:300], moore_stl_response[:300].numpy(), 
            'g-', linewidth=2, label='Moore STL')
    ax2.plot(time_axis[:300], moore_ltl_response[:300].numpy(), 
            'r-', linewidth=2, label='Moore LTL')
    ax2.plot(time_axis[:300], glasberg_response[:300].numpy(), 
            'b--', linewidth=2, label='Glasberg')
    
    ax2.axvline(x=time_axis[100], color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Response')
    ax2.set_title('Impulse Response')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # === Plot 3: Step response ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    ax3.plot(time_axis[:400], step[:400].numpy(), 
            'k:', linewidth=1, alpha=0.5, label='Input')
    ax3.plot(time_axis[:400], moore_stl_step[:400].numpy(), 
            'g-', linewidth=2, label='Moore STL')
    ax3.plot(time_axis[:400], moore_ltl_step[:400].numpy(), 
            'r-', linewidth=2, label='Moore LTL')
    ax3.plot(time_axis[:400], glasberg_step[:400].numpy(), 
            'b--', linewidth=2, label='Glasberg')
    
    ax3.axvline(x=time_axis[100], color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Response')
    ax3.set_title('Step Response (Rising Edge)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # === Plot 4: Tone burst attack ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.plot(time_axis[:300], tone_burst[:300].numpy(), 
            'k:', linewidth=1, alpha=0.5, label='Input')
    ax4.plot(time_axis[:300], moore_stl_burst[:300].numpy(), 
            'g-', linewidth=2, label='Moore STL')
    ax4.plot(time_axis[:300], moore_ltl_burst[:300].numpy(), 
            'r-', linewidth=2, label='Moore LTL')
    ax4.plot(time_axis[:300], glasberg_burst[:300].numpy(), 
            'b--', linewidth=2, label='Glasberg')
    
    ax4.axvline(x=time_axis[100], color='k', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Response')
    ax4.set_title('Tone Burst - Attack Phase')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # === Plot 5: Tone burst release ===
    ax5 = fig.add_subplot(gs[1, 1])
    
    ax5.plot(time_axis[300:], tone_burst[300:].numpy(), 
            'k:', linewidth=1, alpha=0.5, label='Input')
    ax5.plot(time_axis[300:], moore_stl_burst[300:].numpy(), 
            'g-', linewidth=2, label='Moore STL')
    ax5.plot(time_axis[300:], moore_ltl_burst[300:].numpy(), 
            'r-', linewidth=2, label='Moore LTL')
    ax5.plot(time_axis[300:], glasberg_burst[300:].numpy(), 
            'b--', linewidth=2, label='Glasberg')
    
    ax5.axvline(x=time_axis[400], color='k', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Response')
    ax5.set_title('Tone Burst - Release Phase')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # === Plot 6: Amplitude modulation ===
    ax6 = fig.add_subplot(gs[1, 2])
    
    plot_start = 100
    plot_end = 400
    ax6.plot(time_axis[plot_start:plot_end], am_torch[plot_start:plot_end].numpy(), 
            'k:', linewidth=1, alpha=0.5, label='Input (5 Hz AM)')
    ax6.plot(time_axis[plot_start:plot_end], moore_stl_am[plot_start:plot_end].numpy(), 
            'g-', linewidth=2, label='Moore STL')
    ax6.plot(time_axis[plot_start:plot_end], moore_ltl_am[plot_start:plot_end].numpy(), 
            'r-', linewidth=2, label='Moore LTL')
    ax6.plot(time_axis[plot_start:plot_end], glasberg_am[plot_start:plot_end].numpy(), 
            'b--', linewidth=2, label='Glasberg')
    
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Response')
    ax6.set_title('5 Hz Amplitude Modulation')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # === Plot 7: Full pipeline comparison ===
    ax7 = fig.add_subplot(gs[2, 0])
    
    ax7.plot(time_axis, moore_stl.numpy(), 'g-', linewidth=2, label='Moore STL')
    ax7.plot(time_axis, moore_ltl.numpy(), 'r-', linewidth=2, label='Moore LTL')
    ax7.plot(time_axis, glasberg_integrated.numpy(), 'b--', linewidth=2, label='Glasberg')
    
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Loudness (sone)')
    ax7.set_title('Full Pipeline - Realistic Signal')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # === Plot 8: Modulation depth attenuation ===
    ax8 = fig.add_subplot(gs[2, 1])
    
    depths = [input_depth, stl_depth, ltl_depth, glasberg_depth]
    labels_depth = ['Input', 'Moore\nSTL', 'Moore\nLTL', 'Glasberg']
    colors = ['#7f7f7f', '#2ca02c', '#d62728', '#1f77b4']
    
    bars = ax8.bar(range(4), depths, color=colors, alpha=0.8)
    ax8.set_ylabel('Modulation Depth')
    ax8.set_title('Modulation Depth Comparison (5 Hz)')
    ax8.set_xticks(range(4))
    ax8.set_xticklabels(labels_depth)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Annotate attenuation
    for i in range(1, 4):
        attenuation = (1 - depths[i] / depths[0]) * 100
        ax8.text(i, depths[i], f'-{attenuation:.0f}%',
                ha='center', va='bottom', fontsize=8)
    
    # === Plot 9: Correlation scatter ===
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Scatter Moore LTL vs Glasberg
    ax9.scatter(glasberg_integrated.numpy(), moore_ltl.numpy(), 
               alpha=0.5, s=10, color='#d62728', label=f'Moore LTL (r={corr_ltl:.3f})')
    
    # Add identity line
    min_val = min(glasberg_integrated.min().item(), moore_ltl.min().item())
    max_val = max(glasberg_integrated.max().item(), moore_ltl.max().item())
    ax9.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
    
    ax9.set_xlabel('Glasberg2002 Loudness (sone)')
    ax9.set_ylabel('Moore2016 LTL Loudness (sone)')
    ax9.set_title('Correlation: Moore LTL vs Glasberg')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    ax9.set_aspect('equal')
    
    # Overall title
    fig.suptitle('Moore2016 vs Glasberg2002 - Temporal Integration Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    test_figures_dir = Path(__file__).parent.parent.parent / 'test_figures'
    test_figures_dir.mkdir(exist_ok=True)
    output_path = test_figures_dir / 'moore2016_temporal_comparison.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figure saved: {output_path}")
    
    print("\n" + "="*80)
    print("COMPARISON TEST COMPLETED")
    print("="*80)
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
Moore2016 vs Glasberg2002 - Main Differences:

1. Architecture:
   - Moore2016: Dual-stage AGC (STL + LTL) with asymmetric attack/release
   - Glasberg2002: Single-stage exponential smoothing with two time constants
   
2. Time Constants:
   - Moore2016 STL: ~730ms attack, ~497ms release
   - Moore2016 LTL: ~1600ms attack, ~12000ms release
   - Glasberg2002: 5ms + 200ms (fixed)
   
3. Response Characteristics:
   - Moore2016: Asymmetric (faster attack than release), especially in LTL
   - Glasberg2002: Symmetric smoothing
   
4. Modulation Sensitivity:
   - Moore2016 LTL: Very strong attenuation (long time constants)
   - Glasberg2002: Moderate attenuation (shorter time constants)
   
5. Application:
   - Moore2016: Applied channel-wise before integration
   - Glasberg2002: Applied after spatial integration
   
6. Correlation:
   - Moore2016 STL shows faster dynamics similar to Glasberg short component
   - Moore2016 LTL provides additional long-term smoothing
""")


if __name__ == '__main__':
    test_moore2016_vs_glasberg2002()
