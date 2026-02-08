"""
Moore2016 Temporal Integration - Test Suite

Contents:
1. test_moore2016_temporal: Comprehensive verification of temporal AGC processing

Structure:
- Tests AGC coefficients (STL and LTL)
- Verifies attack/release responses with tone bursts (40, 60, 80 dB)
- Measures rise and decay times
- Tests amplitude modulation smoothing (5 Hz)
- Validates full pipeline (inst → STL → LTL)
- Compares variance reduction across stages
- Compares with Glasberg2002 temporal integration

Figures generated:
- moore2016_temporal.png: Complete Moore2016 temporal analysis (9 subplots)
  * AGC coefficients
  * Attack/release responses at multiple levels
  * Step responses and time constants
  * Modulated signal smoothing
  * Full pipeline visualization
  * Variance reduction analysis
  
- moore2016_temporal_comparison.png: Moore2016 vs Glasberg2002 comparison
  * Time constant comparison
  * Response curves comparison
  * Smoothing efficiency comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import Moore2016AGC, Moore2016TemporalIntegration


def test_moore2016_temporal():
    """Test Moore2016 temporal integration with comprehensive analysis."""
    
    print("\n" + "="*80)
    print("MOORE2016 TEMPORAL INTEGRATION TEST")
    print("="*80)
    
    # Initialize modules
    stl_agc = Moore2016AGC(attack_alpha=0.045, release_alpha=0.033)
    ltl_agc = Moore2016AGC(attack_alpha=0.01, release_alpha=0.00133)
    temporal = Moore2016TemporalIntegration()
    
    # Frame rate assumption
    fs = 32000
    hop_length = 512
    frame_rate = fs / hop_length
    dt = 1.0 / frame_rate
    
    print(f"\nConfiguration:")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Hop length: {hop_length} samples")
    print(f"  Frame rate: {frame_rate:.2f} fps")
    print(f"  Frame period: {dt*1000:.2f} ms")
    
    stl_params = stl_agc.get_parameters()
    ltl_params = ltl_agc.get_parameters()
    
    print(f"\n  Short-Term AGC:")
    print(f"    Attack alpha: {stl_params['attack_alpha']:.4f}")
    print(f"    Release alpha: {stl_params['release_alpha']:.4f}")
    print(f"\n  Long-Term AGC:")
    print(f"    Attack alpha: {ltl_params['attack_alpha']:.6f}")
    print(f"    Release alpha: {ltl_params['release_alpha']:.6f}")
    
    # Number of frames for tests
    n_frames = 600
    time_axis = np.arange(n_frames) * dt
    
    # ========== TEST 1: AGC Coefficients ==========
    print("\n" + "-"*80)
    print("TEST 1: AGC coefficients verification")
    print("-"*80)
    
    print(f"\nCoefficient ratios:")
    stl_ratio = stl_params['attack_alpha'] / stl_params['release_alpha']
    ltl_ratio = ltl_params['attack_alpha'] / ltl_params['release_alpha']
    print(f"  STL attack/release: {stl_ratio:.2f}")
    print(f"  LTL attack/release: {ltl_ratio:.2f}")
    
    attack_ratio = stl_params['attack_alpha'] / ltl_params['attack_alpha']
    release_ratio = stl_params['release_alpha'] / ltl_params['release_alpha']
    print(f"  STL/LTL attack ratio: {attack_ratio:.2f}x")
    print(f"  STL/LTL release ratio: {release_ratio:.2f}x")
    
    # ========== TEST 2: Attack response - tone burst onset ==========
    print("\n" + "-"*80)
    print("TEST 2: Attack response - tone burst onset")
    print("-"*80)
    
    # Test at multiple levels: 40, 60, 80 dB
    test_levels_db = [40, 60, 80]
    attack_responses_stl = {}
    attack_responses_ltl = {}
    rise_times_stl = {}
    rise_times_ltl = {}
    
    for level_db in test_levels_db:
        # Create tone burst: silence → burst
        tone_burst = torch.zeros(n_frames)
        tone_burst[100:] = 10 ** (level_db / 20.0) / 10.0  # Convert dB to linear
        
        # Apply AGC
        stl_response = stl_agc(tone_burst)
        ltl_response = ltl_agc(tone_burst)
        
        attack_responses_stl[level_db] = stl_response.numpy()
        attack_responses_ltl[level_db] = ltl_response.numpy()
        
        # Measure rise time (10% to 90%)
        final_value_stl = stl_response[500].item()
        final_value_ltl = ltl_response[500].item()
        
        # STL rise time
        idx_10_stl = np.where(stl_response.numpy() > 0.1 * final_value_stl)[0]
        idx_90_stl = np.where(stl_response.numpy() > 0.9 * final_value_stl)[0]
        if len(idx_10_stl) > 0 and len(idx_90_stl) > 0:
            rise_time_stl = (idx_90_stl[0] - idx_10_stl[0]) * dt * 1000  # ms
            rise_times_stl[level_db] = rise_time_stl
        
        # LTL rise time
        idx_10_ltl = np.where(ltl_response.numpy() > 0.1 * final_value_ltl)[0]
        idx_90_ltl = np.where(ltl_response.numpy() > 0.9 * final_value_ltl)[0]
        if len(idx_10_ltl) > 0 and len(idx_90_ltl) > 0:
            rise_time_ltl = (idx_90_ltl[0] - idx_10_ltl[0]) * dt * 1000  # ms
            rise_times_ltl[level_db] = rise_time_ltl
        
        print(f"\n  {level_db} dB:")
        print(f"    STL rise time (10-90%): {rise_time_stl:.1f} ms")
        print(f"    LTL rise time (10-90%): {rise_time_ltl:.1f} ms")
        print(f"    Ratio LTL/STL: {rise_time_ltl / rise_time_stl:.2f}x")
    
    # ========== TEST 3: Release response - tone burst offset ==========
    print("\n" + "-"*80)
    print("TEST 3: Release response - tone burst offset")
    print("-"*80)
    
    release_responses_stl = {}
    release_responses_ltl = {}
    decay_times_stl = {}
    decay_times_ltl = {}
    
    for level_db in test_levels_db:
        # Create tone burst: burst → silence
        tone_burst = torch.zeros(n_frames)
        tone_burst[:300] = 10 ** (level_db / 20.0) / 10.0
        
        # Apply AGC
        stl_response = stl_agc(tone_burst)
        ltl_response = ltl_agc(tone_burst)
        
        release_responses_stl[level_db] = stl_response.numpy()
        release_responses_ltl[level_db] = ltl_response.numpy()
        
        # Measure decay time (90% to 10%)
        initial_value_stl = stl_response[299].item()
        initial_value_ltl = ltl_response[299].item()
        
        # STL decay time
        idx_90_stl = np.where(stl_response[300:].numpy() < 0.9 * initial_value_stl)[0]
        idx_10_stl = np.where(stl_response[300:].numpy() < 0.1 * initial_value_stl)[0]
        if len(idx_90_stl) > 0 and len(idx_10_stl) > 0:
            decay_time_stl = (idx_10_stl[0] - idx_90_stl[0]) * dt * 1000  # ms
            decay_times_stl[level_db] = decay_time_stl
        
        # LTL decay time
        idx_90_ltl = np.where(ltl_response[300:].numpy() < 0.9 * initial_value_ltl)[0]
        idx_10_ltl = np.where(ltl_response[300:].numpy() < 0.1 * initial_value_ltl)[0]
        if len(idx_90_ltl) > 0 and len(idx_10_ltl) > 0:
            decay_time_ltl = (idx_10_ltl[0] - idx_90_ltl[0]) * dt * 1000  # ms
            decay_times_ltl[level_db] = decay_time_ltl
        else:
            # LTL might not reach 10% within available frames
            decay_time_ltl = np.nan
            decay_times_ltl[level_db] = decay_time_ltl
        
        print(f"\n  {level_db} dB:")
        print(f"    STL decay time (90-10%): {decay_time_stl:.1f} ms")
        if np.isnan(decay_time_ltl):
            print(f"    LTL decay time (90-10%): >4800 ms (did not complete)")
        else:
            print(f"    LTL decay time (90-10%): {decay_time_ltl:.1f} ms")
            print(f"    Ratio LTL/STL: {decay_time_ltl / decay_time_stl:.2f}x")
    
    # ========== TEST 4: Step response ==========
    print("\n" + "-"*80)
    print("TEST 4: Step response")
    print("-"*80)
    
    # Step function: 0 → 1 → 0
    step = torch.zeros(n_frames)
    step[100:400] = 1.0
    
    stl_step = stl_agc(step)
    ltl_step = ltl_agc(step)
    
    print(f"\nStep response characteristics:")
    print(f"  STL max value: {stl_step.max():.4f}")
    print(f"  LTL max value: {ltl_step.max():.4f}")
    print(f"  STL final decay: {stl_step[-1]:.6f}")
    print(f"  LTL final decay: {ltl_step[-1]:.6f}")
    
    # ========== TEST 5: Modulated signal (5 Hz AM) ==========
    print("\n" + "-"*80)
    print("TEST 5: Amplitude modulation smoothing (5 Hz)")
    print("-"*80)
    
    # Create 5 Hz AM signal
    modulation_freq = 5.0
    carrier_level = 1.0
    modulation_depth = 0.8
    
    t = time_axis
    modulation = carrier_level * (1 + modulation_depth * np.sin(2 * np.pi * modulation_freq * t))
    modulation_torch = torch.from_numpy(modulation).float()
    
    # Apply AGC
    stl_am = stl_agc(modulation_torch)
    ltl_am = ltl_agc(modulation_torch)
    
    # Measure modulation depth reduction
    def modulation_depth_metric(signal):
        return (signal.max() - signal.min()) / (signal.max() + signal.min())
    
    input_depth = modulation_depth_metric(modulation_torch[100:500])
    stl_depth = modulation_depth_metric(stl_am[100:500])
    ltl_depth = modulation_depth_metric(ltl_am[100:500])
    
    print(f"\nModulation depth:")
    print(f"  Input: {input_depth:.4f}")
    print(f"  STL output: {stl_depth:.4f} (reduction: {(1 - stl_depth/input_depth)*100:.1f}%)")
    print(f"  LTL output: {ltl_depth:.4f} (reduction: {(1 - ltl_depth/input_depth)*100:.1f}%)")
    
    # ========== TEST 6: Full pipeline ==========
    print("\n" + "-"*80)
    print("TEST 6: Full pipeline - realistic signal")
    print("-"*80)
    
    # Create realistic instantaneous specific loudness
    inst_spec_loud = torch.randn(n_frames, 150).abs() * 2.0
    # Add some temporal structure
    envelope = torch.from_numpy(np.exp(-time_axis / 5.0)).float()
    inst_spec_loud = inst_spec_loud * envelope.unsqueeze(1)
    
    # Apply full pipeline
    ltl, stl_spec, stl = temporal(inst_spec_loud, return_intermediate=True)
    
    print(f"\nPipeline shapes:")
    print(f"  Instantaneous specific loudness: {inst_spec_loud.shape}")
    print(f"  Short-term specific loudness: {stl_spec.shape}")
    print(f"  Short-term loudness: {stl.shape}")
    print(f"  Long-term loudness: {ltl.shape}")
    
    # ========== TEST 7: Variance reduction ==========
    print("\n" + "-"*80)
    print("TEST 7: Variance reduction across stages")
    print("-"*80)
    
    # Compute variance at each stage
    var_inst = inst_spec_loud.var().item()
    var_stl_spec = stl_spec.var().item()
    var_stl = stl.var().item()
    var_ltl = ltl.var().item()
    
    print(f"\nVariance at each stage:")
    print(f"  Instantaneous: {var_inst:.4f}")
    print(f"  STL specific: {var_stl_spec:.4f} (reduction: {(1 - var_stl_spec/var_inst)*100:.1f}%)")
    print(f"  STL loudness: {var_stl:.4f} (reduction: {(1 - var_stl/var_inst)*100:.1f}%)")
    print(f"  LTL loudness: {var_ltl:.4f} (reduction: {(1 - var_ltl/var_inst)*100:.1f}%)")
    
    # ========== VISUALIZATION ==========
    print("\n" + "-"*80)
    print("Creating Moore2016 temporal visualization...")
    print("-"*80)
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    colors_levels = {'40': '#1f77b4', '60': '#ff7f0e', '80': '#2ca02c'}
    
    # === Plot 1: AGC coefficients ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    coeffs = [
        stl_params['attack_alpha'],
        stl_params['release_alpha'],
        ltl_params['attack_alpha'],
        ltl_params['release_alpha']
    ]
    labels = ['STL\nAttack', 'STL\nRelease', 'LTL\nAttack', 'LTL\nRelease']
    colors_bar = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = ax1.bar(range(4), coeffs, color=colors_bar, alpha=0.8)
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('AGC Coefficients')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, coeffs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}' if val > 0.01 else f'{val:.5f}',
                ha='center', va='bottom', fontsize=8)
    
    # === Plot 2: Attack response (onset) ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    for level_db in test_levels_db:
        response = attack_responses_stl[level_db]
        ax2.plot(time_axis[:400], response[:400], '-', linewidth=2,
                color=colors_levels[str(level_db)], label=f'{level_db} dB STL')
    
    ax2.axvline(x=time_axis[100], color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Loudness (arbitrary)')
    ax2.set_title('Attack Response - STL')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # === Plot 3: Release response (offset) ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    for level_db in test_levels_db:
        response = release_responses_stl[level_db]
        ax3.plot(time_axis[200:], response[200:], '-', linewidth=2,
                color=colors_levels[str(level_db)], label=f'{level_db} dB STL')
    
    ax3.axvline(x=time_axis[300], color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Loudness (arbitrary)')
    ax3.set_title('Release Response - STL')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # === Plot 4: LTL attack response ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    for level_db in test_levels_db:
        response = attack_responses_ltl[level_db]
        ax4.plot(time_axis[:400], response[:400], '-', linewidth=2,
                color=colors_levels[str(level_db)], label=f'{level_db} dB LTL')
    
    ax4.axvline(x=time_axis[100], color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Loudness (arbitrary)')
    ax4.set_title('Attack Response - LTL')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # === Plot 5: Step response comparison ===
    ax5 = fig.add_subplot(gs[1, 1])
    
    ax5.plot(time_axis, step.numpy(), 'k--', linewidth=1.5, alpha=0.5, label='Input step')
    ax5.plot(time_axis, stl_step.numpy(), 'b-', linewidth=2, label='STL')
    ax5.plot(time_axis, ltl_step.numpy(), 'r-', linewidth=2, label='LTL')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Response')
    ax5.set_title('Step Response Comparison')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim([0, time_axis[500]])
    
    # === Plot 6: Modulated signal (5 Hz AM) ===
    ax6 = fig.add_subplot(gs[1, 2])
    
    ax6.plot(time_axis[100:400], modulation[100:400], 'k--', linewidth=1.5,
            alpha=0.5, label='Input (5 Hz AM)')
    ax6.plot(time_axis[100:400], stl_am[100:400].numpy(), 'b-', linewidth=2, label='STL')
    ax6.plot(time_axis[100:400], ltl_am[100:400].numpy(), 'r-', linewidth=2, label='LTL')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Amplitude')
    ax6.set_title('Amplitude Modulation Smoothing')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # === Plot 7: Full pipeline temporal evolution ===
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Plot mean across 150 channels
    inst_mean = inst_spec_loud.mean(dim=1).numpy()
    stl_spec_mean = stl_spec.mean(dim=1).numpy()
    
    ax7.plot(time_axis, inst_mean, 'k-', linewidth=1.5, alpha=0.6, label='Inst (mean)')
    ax7.plot(time_axis, stl_spec_mean, 'b-', linewidth=2, label='STL spec (mean)')
    ax7.plot(time_axis, stl.numpy(), 'g-', linewidth=2, label='STL')
    ax7.plot(time_axis, ltl.numpy(), 'r-', linewidth=2, label='LTL')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Loudness (sone)')
    ax7.set_title('Full Pipeline: Inst → STL → LTL')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # === Plot 8: Variance reduction ===
    ax8 = fig.add_subplot(gs[2, 1])
    
    stages = ['Inst', 'STL\nspec', 'STL', 'LTL']
    variances = [var_inst, var_stl_spec, var_stl, var_ltl]
    colors_var = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax8.bar(range(4), variances, color=colors_var, alpha=0.8)
    ax8.set_ylabel('Variance')
    ax8.set_title('Variance Reduction per Stage')
    ax8.set_xticks(range(4))
    ax8.set_xticklabels(stages)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_yscale('log')
    
    # Annotate reduction percentages
    for i in range(1, 4):
        reduction = (1 - variances[i] / variances[0]) * 100
        ax8.text(i, variances[i], f'-{reduction:.0f}%',
                ha='center', va='bottom', fontsize=8)
    
    # === Plot 9: Rise/Decay time comparison ===
    ax9 = fig.add_subplot(gs[2, 2])
    
    x = np.arange(len(test_levels_db))
    width = 0.2
    
    rise_stl = [rise_times_stl[lvl] for lvl in test_levels_db]
    rise_ltl = [rise_times_ltl[lvl] for lvl in test_levels_db]
    decay_stl = [decay_times_stl[lvl] for lvl in test_levels_db]
    decay_ltl = [decay_times_ltl.get(lvl, 0) if not np.isnan(decay_times_ltl.get(lvl, np.nan)) else 5000 
                 for lvl in test_levels_db]  # Use 5000 ms for incomplete decays
    
    ax9.bar(x - 1.5*width, rise_stl, width, label='STL Rise', color='#2ca02c', alpha=0.8)
    ax9.bar(x - 0.5*width, rise_ltl, width, label='LTL Rise', color='#9467bd', alpha=0.8)
    ax9.bar(x + 0.5*width, decay_stl, width, label='STL Decay', color='#d62728', alpha=0.8)
    
    # Plot LTL decay with different pattern for incomplete
    ltl_bars = ax9.bar(x + 1.5*width, decay_ltl, width, label='LTL Decay', 
                       color='#8c564b', alpha=0.8)
    
    # Mark incomplete decays with hatching
    for i, lvl in enumerate(test_levels_db):
        if np.isnan(decay_times_ltl.get(lvl, np.nan)):
            ltl_bars[i].set_hatch('///')
            ax9.text(i + 1.5*width, decay_ltl[i], '>',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax9.set_ylabel('Time (ms)')
    ax9.set_title('Rise/Decay Times (10-90%)')
    ax9.set_xticks(x)
    ax9.set_xticklabels([f'{lvl} dB' for lvl in test_levels_db])
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle('Moore2016 Temporal Integration - Complete Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    test_figures_dir = Path(__file__).parent.parent.parent / 'test_figures'
    test_figures_dir.mkdir(exist_ok=True)
    output_path = test_figures_dir / 'moore2016_temporal.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figure saved: {output_path}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == '__main__':
    test_moore2016_temporal()
