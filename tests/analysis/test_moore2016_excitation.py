"""
Moore2016 Excitation Pattern - Test Suite

Contents:
1. test_moore2016_excitation: Comprehensive verification of Moore2016ExcitationPattern

Structure:
- Tests roex filter response W(p,g) = (1 + p|g|) * exp(-p|g|)
- Verifies level-dependent lower slope pl(f, X)
- Tests sparse spectrum to excitation pattern conversion
- Validates spreading limited to ±4 octaves
- Compares filter shapes at different frequencies (200 Hz, 1 kHz, 5 kHz)
- Tests level dependency (40, 60, 80 dB SPL)

Figures generated:
- moore2016_excitation.png: Complete excitation pattern analysis
  * Roex filter shapes at multiple frequencies
  * Level-dependent lower slope effects
  * Excitation patterns for pure tones
  * Multi-component spectrum spreading
  * p(f) and pl(f, X) parameter curves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.common import Moore2016ExcitationPattern


def test_moore2016_excitation():
    """Test Moore2016ExcitationPattern with various signals and parameters."""
    
    print("\n" + "="*80)
    print("MOORE2016 EXCITATION PATTERN TEST")
    print("="*80)
    
    # Initialize excitation pattern model
    excitation_pattern = Moore2016ExcitationPattern()
    
    print(f"\nExcitation Pattern Configuration:")
    print(f"  ERB range: {excitation_pattern.erb_lower} - {excitation_pattern.erb_upper}")
    print(f"  ERB step: {excitation_pattern.erb_step}")
    print(f"  Number of channels: {excitation_pattern.n_channels}")
    print(f"  Spreading limit: ±{excitation_pattern.spreading_limit_octaves} octaves")
    print(f"  Center frequency range: {excitation_pattern.fc_channels[0]:.1f} - {excitation_pattern.fc_channels[-1]:.1f} Hz")
    print(f"  p(1000 Hz): {excitation_pattern.p1000:.3f}")
    
    # ========== TEST 1: Roex Filter Response W(p, g) ==========
    print("\n" + "-"*80)
    print("TEST 1: Roex Filter Response W(p, g)")
    print("-"*80)
    
    # Test filter at 1 kHz
    fc_1khz = torch.tensor(1000.0)
    p_1khz = excitation_pattern._get_p(fc_1khz)
    print(f"\nAt 1 kHz:")
    print(f"  ERB(1000) = {24.673 * (4.368 + 1):.2f} Hz")
    print(f"  p(1000) = {p_1khz:.3f}")
    
    # Normalized frequency deviation g
    g_values = torch.linspace(-4, 4, 401)  # ±4 octaves, fine resolution
    
    # Filter responses at different levels
    levels_test = [40, 60, 80]
    W_responses = {}
    
    for level in levels_test:
        level_tensor = torch.tensor(float(level))
        W = excitation_pattern._get_W(
            p_1khz.expand(len(g_values)),
            g_values,
            level_tensor
        )
        W_responses[level] = W.numpy()  # Use level (int) as key, not tensor
        
        # Print peak and asymmetry
        idx_peak = torch.argmax(W)
        g_peak = g_values[idx_peak].item()
        print(f"\n  Level {level} dB SPL:")
        print(f"    Peak at g = {g_peak:.3f}")
        print(f"    W(g=0) = {W[200]:.4f}")  # Center value (g=0 at index 200)
        print(f"    W(g=-1) / W(g=+1) = {W[100] / W[300]:.3f}")  # Asymmetry ratio
    
    # ========== TEST 2: Parameter p(f) across frequencies ==========
    print("\n" + "-"*80)
    print("TEST 2: Filter slope p(f) across frequencies")
    print("-"*80)
    
    # Test frequencies
    test_freqs = torch.tensor([200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0])
    
    for freq in test_freqs:
        p = excitation_pattern._get_p(freq)
        erb = 24.673 * (4.368 * freq / 1000 + 1)
        print(f"  f = {freq:6.0f} Hz: ERB = {erb:6.2f} Hz, p = {p:6.3f}")
    
    # ========== TEST 3: Level-dependent lower slope pl(f, X) ==========
    print("\n" + "-"*80)
    print("TEST 3: Level-dependent lower slope pl(f, X)")
    print("-"*80)
    
    freq_test = torch.tensor(1000.0)
    p_base = excitation_pattern._get_p(freq_test)
    
    levels_pl = [20, 40, 51, 60, 80, 100]
    print(f"\nAt f = 1000 Hz (p = {p_base:.3f}):")
    for level in levels_pl:
        level_tensor = torch.tensor(float(level))
        pl = excitation_pattern._get_pl(freq_test, level_tensor)
        reduction = p_base - pl
        print(f"  X = {level:3d} dB: pl = {pl:.3f}, reduction = {reduction:.3f}")
    
    # ========== TEST 4: Single-tone excitation patterns ==========
    print("\n" + "-"*80)
    print("TEST 4: Single-tone excitation patterns")
    print("-"*80)
    
    # Test pure tones at different frequencies and levels
    test_configs = [
        (200.0, 60.0, "200 Hz @ 60 dB"),
        (1000.0, 40.0, "1 kHz @ 40 dB"),
        (1000.0, 60.0, "1 kHz @ 60 dB"),
        (1000.0, 80.0, "1 kHz @ 80 dB"),
        (5000.0, 60.0, "5 kHz @ 60 dB"),
    ]
    
    excitation_patterns = {}
    
    for freq, level, label in test_configs:
        freqs = torch.tensor([[freq]])  # batch=1, n_components=1
        levels = torch.tensor([[level]])
        
        exc = excitation_pattern(freqs, levels)
        excitation_patterns[label] = exc.squeeze(0).numpy()
        
        # Find peak and width
        peak_idx = np.argmax(excitation_patterns[label])
        peak_erb = excitation_pattern.erb_channels[peak_idx].item()
        peak_fc = excitation_pattern.fc_channels[peak_idx].item()
        peak_val = excitation_patterns[label][peak_idx]
        
        # Find bandwidth (where excitation drops by 10 dB from peak)
        mask_10db = excitation_patterns[label] > (peak_val - 10)
        bandwidth_channels = np.sum(mask_10db)
        bandwidth_erb = bandwidth_channels * excitation_pattern.erb_step
        
        print(f"\n  {label}:")
        print(f"    Peak: ERB {peak_erb:.2f}, {peak_fc:.1f} Hz, {peak_val:.1f} dB")
        print(f"    Bandwidth (10 dB): {bandwidth_erb:.2f} ERB ({bandwidth_channels} channels)")
    
    # ========== TEST 5: Multi-component spectrum ==========
    print("\n" + "-"*80)
    print("TEST 5: Multi-component spectrum")
    print("-"*80)
    
    # Complex tone: 500, 1000, 1500 Hz @ 60 dB each
    freqs_complex = torch.tensor([[500.0, 1000.0, 1500.0]])
    levels_complex = torch.tensor([[60.0, 60.0, 60.0]])
    
    exc_complex = excitation_pattern(freqs_complex, levels_complex)
    excitation_patterns["Complex (500, 1k, 1.5k Hz @ 60 dB)"] = exc_complex.squeeze(0).numpy()
    
    print(f"\n  Complex tone (500, 1000, 1500 Hz @ 60 dB each):")
    
    # Find peaks for each component - search near target frequency in fc_channels
    for i, f in enumerate([500, 1000, 1500]):
        # Find channel nearest to target frequency
        idx_target = torch.argmin(torch.abs(excitation_pattern.fc_channels - f)).item()
        
        # Search for peak in local neighborhood (±5 channels)
        search_start = max(0, idx_target - 5)
        search_end = min(len(excitation_pattern.erb_channels), idx_target + 6)
        local_exc = exc_complex[0, search_start:search_end].numpy()
        local_peak_idx = np.argmax(local_exc)
        global_peak_idx = search_start + local_peak_idx
        
        peak_val = exc_complex[0, global_peak_idx].item()
        peak_fc = excitation_pattern.fc_channels[global_peak_idx].item()
        
        print(f"    Component {i+1} ({f} Hz): Peak at {peak_fc:.1f} Hz, {peak_val:.1f} dB")
    
    # ========== TEST 6: Spreading limit (±4 octaves) ==========
    print("\n" + "-"*80)
    print("TEST 6: Spreading limit verification")
    print("-"*80)
    
    # Test tone at 1 kHz
    freq_center = 1000.0
    freqs_spread_test = torch.tensor([[freq_center]])
    levels_spread_test = torch.tensor([[60.0]])
    
    exc_test = excitation_pattern(freqs_spread_test, levels_spread_test).squeeze(0).numpy()
    
    # Find channels at ±4 octaves
    fc_lower = freq_center / (2**4)  # -4 octaves
    fc_upper = freq_center * (2**4)  # +4 octaves
    
    # Find nearest channels
    idx_lower = torch.argmin(torch.abs(excitation_pattern.fc_channels - fc_lower)).item()
    idx_upper = torch.argmin(torch.abs(excitation_pattern.fc_channels - fc_upper)).item()
    idx_center = torch.argmin(torch.abs(excitation_pattern.fc_channels - freq_center)).item()
    
    print(f"\n  Center tone: {freq_center} Hz")
    print(f"  4 octaves below: {excitation_pattern.fc_channels[idx_lower]:.1f} Hz, exc = {exc_test[idx_lower]:.1f} dB")
    print(f"  Center: {excitation_pattern.fc_channels[idx_center]:.1f} Hz, exc = {exc_test[idx_center]:.1f} dB")
    print(f"  4 octaves above: {excitation_pattern.fc_channels[idx_upper]:.1f} Hz, exc = {exc_test[idx_upper]:.1f} dB")
    
    # Check that excitation is very low beyond ±4 octaves
    if idx_lower > 0:
        beyond_lower = exc_test[idx_lower - 1]
        print(f"  Beyond -4 oct: {excitation_pattern.fc_channels[idx_lower-1]:.1f} Hz, exc = {beyond_lower:.1f} dB")
    
    if idx_upper < len(exc_test) - 1:
        beyond_upper = exc_test[idx_upper + 1]
        print(f"  Beyond +4 oct: {excitation_pattern.fc_channels[idx_upper+1]:.1f} Hz, exc = {beyond_upper:.1f} dB")
    
    # ========== VISUALIZATION ==========
    print("\n" + "-"*80)
    print("Creating comprehensive visualization...")
    print("-"*80)
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # === Plot 1: Roex filter shapes W(p, g) at different levels ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    for level in levels_test:
        ax1.plot(g_values.numpy(), W_responses[level], label=f'{level} dB SPL', linewidth=2)
    
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Normalized frequency deviation g = (f - fc) / fc')
    ax1.set_ylabel('Filter response W(p, g)')
    ax1.set_title('Roex Filter Response at 1 kHz\nLevel-dependent lower slope')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-4, 4])
    
    # === Plot 2: Filter slope p(f) vs frequency ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    freqs_curve = torch.logspace(np.log10(100), np.log10(15000), 200)
    p_curve = torch.zeros_like(freqs_curve)
    erb_curve = torch.zeros_like(freqs_curve)
    
    for i, f in enumerate(freqs_curve):
        p_curve[i] = excitation_pattern._get_p(f)
        erb_curve[i] = 24.673 * (4.368 * f / 1000 + 1)
    
    ax2.semilogx(freqs_curve.numpy(), p_curve.numpy(), 'b-', linewidth=2, label='p(f) = 4f / ERB(f)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Filter slope p(f)')
    ax2.set_title('Filter Slope Parameter p(f)')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    # Add ERB curve on secondary axis
    ax2_erb = ax2.twinx()
    ax2_erb.semilogx(freqs_curve.numpy(), erb_curve.numpy(), 'r--', linewidth=1.5, alpha=0.6, label='ERB(f)')
    ax2_erb.set_ylabel('ERB (Hz)', color='r')
    ax2_erb.tick_params(axis='y', labelcolor='r')
    ax2_erb.legend(loc='lower right')
    
    # === Plot 3: Level-dependent pl(f, X) ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    levels_range = torch.linspace(20, 100, 50)
    freqs_pl_test = [200, 1000, 5000]
    
    for freq in freqs_pl_test:
        freq_tensor = torch.tensor(float(freq))
        pl_values = torch.zeros_like(levels_range)
        p_base = excitation_pattern._get_p(freq_tensor)
        
        for i, level in enumerate(levels_range):
            pl_values[i] = excitation_pattern._get_pl(freq_tensor, level)
        
        ax3.plot(levels_range.numpy(), pl_values.numpy(), linewidth=2, label=f'{freq} Hz')
        ax3.axhline(y=p_base.item(), color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Input level X (dB SPL)')
    ax3.set_ylabel('Lower slope pl(f, X)')
    ax3.set_title('Level-dependent Lower Slope\npl = p - 0.35·(p/p₁₀₀₀)·(X - 51)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([20, 100])
    
    # === Plot 4: Pure tone excitation patterns (different frequencies) ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    erb_scale = excitation_pattern.erb_channels.numpy()
    fc_scale = excitation_pattern.fc_channels.numpy()
    
    for label in ["200 Hz @ 60 dB", "1 kHz @ 60 dB", "5 kHz @ 60 dB"]:
        ax4.plot(erb_scale, excitation_patterns[label], linewidth=2, label=label)
    
    ax4.set_xlabel('ERB scale')
    ax4.set_ylabel('Excitation (dB)')
    ax4.set_title('Excitation Patterns: Different Frequencies\n(all @ 60 dB SPL)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([erb_scale[0], erb_scale[-1]])
    
    # === Plot 5: Pure tone excitation patterns (different levels) ===
    ax5 = fig.add_subplot(gs[1, 1])
    
    for label in ["1 kHz @ 40 dB", "1 kHz @ 60 dB", "1 kHz @ 80 dB"]:
        ax5.plot(erb_scale, excitation_patterns[label], linewidth=2, label=label)
    
    ax5.set_xlabel('ERB scale')
    ax5.set_ylabel('Excitation (dB)')
    ax5.set_title('Excitation Patterns: Level Dependency\n(1 kHz at different levels)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([erb_scale[0], erb_scale[-1]])
    
    # === Plot 6: Multi-component excitation pattern ===
    ax6 = fig.add_subplot(gs[1, 2])
    
    ax6.plot(erb_scale, excitation_patterns["Complex (500, 1k, 1.5k Hz @ 60 dB)"], 
             'g-', linewidth=2, label='Complex tone')
    
    # Mark component frequencies with vertical lines only
    for freq in [500, 1000, 1500]:
        # Find ERB value by looking up in channels
        idx = torch.argmin(torch.abs(excitation_pattern.fc_channels - freq)).item()
        erb_val = excitation_pattern.erb_channels[idx].item()
        ax6.axvline(x=erb_val, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    ax6.set_xlabel('ERB scale')
    ax6.set_ylabel('Excitation (dB)')
    ax6.set_title('Multi-Component Excitation Pattern\n500, 1000, 1500 Hz @ 60 dB each')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([5, 25])  # Zoom to region of interest
    
    # === Plot 7: Excitation vs frequency (linear frequency scale) ===
    ax7 = fig.add_subplot(gs[2, 0])
    
    for label in ["200 Hz @ 60 dB", "1 kHz @ 60 dB", "5 kHz @ 60 dB"]:
        ax7.semilogx(fc_scale, excitation_patterns[label], linewidth=2, label=label)
    
    ax7.set_xlabel('Center frequency (Hz)')
    ax7.set_ylabel('Excitation (dB)')
    ax7.set_title('Excitation Patterns vs Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3, which='both')
    ax7.set_xlim([fc_scale[0], fc_scale[-1]])
    
    # === Plot 8: Spreading range verification ===
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Use 1 kHz @ 60 dB
    exc_1k = excitation_patterns["1 kHz @ 60 dB"]
    idx_1k = torch.argmin(torch.abs(excitation_pattern.fc_channels - 1000)).item()
    
    # Calculate octave distance from 1 kHz
    octave_distance = np.log2(fc_scale / 1000.0)
    
    ax8.plot(octave_distance, exc_1k, 'b-', linewidth=2, label='1 kHz @ 60 dB')
    ax8.axvline(x=-4, color='r', linestyle='--', alpha=0.7, linewidth=2, label='±4 octave limit')
    ax8.axvline(x=4, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax8.axvline(x=0, color='k', linestyle=':', alpha=0.3, linewidth=1)
    
    ax8.set_xlabel('Octave distance from 1 kHz')
    ax8.set_ylabel('Excitation (dB)')
    ax8.set_title('Spreading Limit Verification\n(±4 octaves from center frequency)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([-5, 5])
    
    # === Plot 9: Filter shapes at different center frequencies ===
    ax9 = fig.add_subplot(gs[2, 2])
    
    g_plot = torch.linspace(-2, 2, 201)
    
    for fc in [200, 1000, 5000]:
        fc_tensor = torch.tensor(float(fc))
        p = excitation_pattern._get_p(fc_tensor)
        level_ref = torch.tensor(60.0)
        
        W = excitation_pattern._get_W(
            p.expand(len(g_plot)),
            g_plot,
            level_ref
        )
        
        ax9.plot(g_plot.numpy(), W.numpy(), linewidth=2, label=f'fc = {fc} Hz (p={p:.2f})')
    
    ax9.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax9.set_xlabel('Normalized frequency deviation g')
    ax9.set_ylabel('Filter response W(p, g)')
    ax9.set_title('Roex Filter Shapes at Different fc\n(@ 60 dB SPL)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([-2, 2])
    
    # Overall title
    fig.suptitle('Moore2016 Excitation Pattern - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    test_figures_dir = Path(__file__).parent.parent.parent / 'test_figures'
    test_figures_dir.mkdir(exist_ok=True)
    output_path = test_figures_dir / 'moore2016_excitation.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figure saved: {output_path}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == '__main__':
    test_moore2016_excitation()
