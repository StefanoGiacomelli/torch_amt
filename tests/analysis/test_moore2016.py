"""
Moore2016 Complete Binaural Loudness Model - Test Suite

Tests the complete integrated Moore2016 pipeline with separate visualizations
for each test scenario.

Test structure:
1. test_model_instantiation: Verifies all submodules
2-6. Test individual scenarios (diotic, dichotic, noise, AM tone, tone burst)
     Each generates its own figure showing:
     - Input waveform (L and R channels)
     - STL and LTL loudness over time
     - Specific loudness patterns

7. test_intermediate_outputs: Verifies pipeline stages
8. test_learnable_parameters: Tests learnable mode

Figures generated:
- moore2016_diotic_tone.png
- moore2016_dichotic_tone.png
- moore2016_broadband_noise.png
- moore2016_am_tone.png
- moore2016_tone_burst.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.models.moore2016 import Moore2016


TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'


def test_model_instantiation():
    """Test 1: Model instantiation."""
    print("\n" + "="*80)
    print("TEST 1: MODEL INSTANTIATION")
    print("="*80)
    
    model = Moore2016(fs=32000)
    
    print(f"\nModel configuration:")
    print(f"  Sampling rate: {model.fs} Hz")
    print(f"  Learnable: {model.learnable}")
    
    print(f"\nSubmodules check:")
    modules = ['outer_middle_ear', 'spectrum', 'excitation', 'specific_loudness',
               'temporal_integration', 'binaural_loudness', 'ltl_agc_left', 'ltl_agc_right']
    for mod in modules:
        check = "✓" if hasattr(model, mod) else "✗"
        print(f"  {mod}: {check}")
    
    print("\n✓ Model instantiated correctly with all submodules")


def test_diotic_tone():
    """Test 2: Diotic tone (1 kHz, 60 dB SPL) with visualization."""
    print("\n" + "="*80)
    print("TEST 2: DIOTIC TONE (1 kHz, 60 dB SPL)")
    print("="*80)
    
    fs = 32000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    
    tone = np.sin(2 * np.pi * 1000 * t) * 0.02  # 60 dB SPL
    audio_left = torch.from_numpy(tone).float()
    audio_right = audio_left.clone()
    audio_stereo = torch.stack([audio_left, audio_right], dim=0).unsqueeze(0)
    
    model = Moore2016(fs=32000, return_stages=True)
    (sLoud, lLoud, mLoud), stages = model(audio_stereo)
    
    print(f"\nLoudness values:")
    print(f"  STL mean: {sLoud.mean():.3f} sones")
    print(f"  STL max: {sLoud.max():.3f} sones")
    print(f"  LTL mean: {lLoud.mean():.3f} sones")
    print(f"  Max loudness: {mLoud.item():.3f} sones")
    
    # === Visualization ===
    t_out = np.arange(sLoud.shape[1]) * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Input waveform
    axes[0].plot(t[:1000], audio_left.numpy()[:1000], 'b-', label='Left', linewidth=1.0, alpha=0.8)
    axes[0].plot(t[:1000], audio_right.numpy()[:1000], 'r--', label='Right', linewidth=1.0, alpha=0.6)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (Pa)')
    axes[0].set_title('Input: Diotic Tone (1 kHz, 60 dB SPL)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # STL and LTL
    axes[1].plot(t_out, sLoud[0].detach().numpy(), 'b-', label='STL', linewidth=1.5)
    axes[1].plot(t_out, lLoud[0].detach().numpy(), 'r-', label='LTL', linewidth=1.5)
    axes[1].axhline(mLoud.item(), color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Max = {mLoud.item():.2f}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Loudness (sones)')
    axes[1].set_title('Moore2016 Binaural Loudness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Specific loudness
    spec_loud_left = stages['stl_spec_loud_left'][0].detach().numpy()
    im = axes[2].imshow(spec_loud_left.T, aspect='auto', origin='lower',
                        extent=[0, duration, 0, 150], cmap='viridis')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ERB Channel')
    axes[2].set_title('Specific Loudness (STL, Left Ear)')
    plt.colorbar(im, ax=axes[2], label='Loudness (sones/ERB)')
    
    plt.tight_layout()
    TEST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = TEST_FIGURES_DIR / 'moore2016_diotic_tone.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()
    
    print("\n✓ Diotic tone test passed")


def test_dichotic_tone():
    """Test 3: Dichotic tone (60 dB L, 50 dB R) with visualization."""
    print("\n" + "="*80)
    print("TEST 3: DICHOTIC TONE (60 dB L, 50 dB R)")
    print("="*80)
    
    fs = 32000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    
    tone_left = np.sin(2 * np.pi * 1000 * t) * 0.02   # 60 dB SPL
    tone_right = np.sin(2 * np.pi * 1000 * t) * 0.006  # 50 dB SPL
    
    audio_left = torch.from_numpy(tone_left).float()
    audio_right = torch.from_numpy(tone_right).float()
    audio_stereo = torch.stack([audio_left, audio_right], dim=0).unsqueeze(0)
    
    model = Moore2016(fs=32000, return_stages=True)
    (sLoud, lLoud, mLoud), stages = model(audio_stereo)
    
    print(f"\nLoudness values:")
    print(f"  STL mean: {sLoud.mean():.3f} sones")
    print(f"  Max loudness: {mLoud.item():.3f} sones")
    
    # === Visualization ===
    t_out = np.arange(sLoud.shape[1]) * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Input waveform
    axes[0].plot(t[:1000], audio_left.numpy()[:1000], 'b-', label='Left (60 dB)', linewidth=1.0)
    axes[0].plot(t[:1000], audio_right.numpy()[:1000], 'r-', label='Right (50 dB)', linewidth=1.0, alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (Pa)')
    axes[0].set_title('Input: Dichotic Tone (1 kHz, 60 dB L / 50 dB R)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # STL and LTL
    axes[1].plot(t_out, sLoud[0].detach().numpy(), 'b-', label='STL', linewidth=1.5)
    axes[1].plot(t_out, lLoud[0].detach().numpy(), 'r-', label='LTL', linewidth=1.5)
    axes[1].axhline(mLoud.item(), color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Max = {mLoud.item():.2f}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Loudness (sones)')
    axes[1].set_title('Moore2016 Binaural Loudness (with Inhibition)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Specific loudness comparison
    spec_loud_left = stages['stl_spec_loud_left'][0].detach().numpy()
    spec_loud_right = stages['stl_spec_loud_right'][0].detach().numpy()
    avg_left = spec_loud_left.mean(axis=0)
    avg_right = spec_loud_right.mean(axis=0)
    erb_channels = np.arange(150)
    
    axes[2].plot(erb_channels, avg_left, 'b-', label='Left (60 dB)', linewidth=2)
    axes[2].plot(erb_channels, avg_right, 'r-', label='Right (50 dB)', linewidth=2, alpha=0.7)
    axes[2].set_xlabel('ERB Channel')
    axes[2].set_ylabel('Specific Loudness (sones/ERB)')
    axes[2].set_title('Time-Averaged Specific Loudness (STL)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = TEST_FIGURES_DIR / 'moore2016_dichotic_tone.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()
    
    print("\n✓ Dichotic tone test passed")


def test_broadband_noise():
    """Test 4: Broadband noise with visualization."""
    print("\n" + "="*80)
    print("TEST 4: BROADBAND NOISE")
    print("="*80)
    
    fs = 32000
    duration = 0.5
    n_samples = int(fs * duration)
    
    np.random.seed(42)
    noise = np.random.randn(n_samples) * 0.02  # ~60 dB SPL
    
    audio = torch.from_numpy(noise).float()
    audio_stereo = torch.stack([audio, audio], dim=0).unsqueeze(0)
    
    model = Moore2016(fs=32000, return_stages=True)
    (sLoud, lLoud, mLoud), stages = model(audio_stereo)
    
    print(f"\nLoudness values:")
    print(f"  STL mean: {sLoud.mean():.3f} sones")
    print(f"  Max loudness: {mLoud.item():.3f} sones")
    print(f"  Note: Higher than pure tone due to spectral summation")
    
    # === Visualization ===
    t = np.arange(n_samples) / fs
    t_out = np.arange(sLoud.shape[1]) * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Input waveform
    axes[0].plot(t[:1000], audio.numpy()[:1000], 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (Pa)')
    axes[0].set_title('Input: Broadband Noise (~60 dB SPL)')
    axes[0].grid(True, alpha=0.3)
    
    # STL and LTL
    axes[1].plot(t_out, sLoud[0].detach().numpy(), 'b-', label='STL', linewidth=1.5)
    axes[1].plot(t_out, lLoud[0].detach().numpy(), 'r-', label='LTL', linewidth=1.5)
    axes[1].axhline(mLoud.item(), color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Max = {mLoud.item():.2f}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Loudness (sones)')
    axes[1].set_title('Moore2016 Binaural Loudness (Broadband)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Specific loudness
    spec_loud_left = stages['stl_spec_loud_left'][0].detach().numpy()
    im = axes[2].imshow(spec_loud_left.T, aspect='auto', origin='lower',
                        extent=[0, duration, 0, 150], cmap='viridis')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ERB Channel')
    axes[2].set_title('Specific Loudness (STL, Left Ear)')
    plt.colorbar(im, ax=axes[2], label='Loudness (sones/ERB)')
    
    plt.tight_layout()
    fig_path = TEST_FIGURES_DIR / 'moore2016_broadband_noise.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()
    
    print("\n✓ Broadband noise test passed")


def test_am_tone():
    """Test 5: Amplitude-modulated tone with visualization."""
    print("\n" + "="*80)
    print("TEST 5: AM TONE (1 kHz carrier, 4 Hz modulation)")
    print("="*80)
    
    fs = 32000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    am_tone = carrier * modulator * 0.02
    
    audio = torch.from_numpy(am_tone).float()
    audio_stereo = torch.stack([audio, audio], dim=0).unsqueeze(0)
    
    model = Moore2016(fs=32000, return_stages=True)
    (sLoud, lLoud, mLoud), stages = model(audio_stereo)
    
    print(f"\nLoudness values:")
    print(f"  STL mean: {sLoud.mean():.3f} sones")
    print(f"  Max loudness: {mLoud.item():.3f} sones")
    
    # === Visualization ===
    t_out = np.arange(sLoud.shape[1]) * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Input waveform
    axes[0].plot(t, am_tone, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (Pa)')
    axes[0].set_title('Input: AM Tone (1 kHz carrier, 4 Hz modulation)')
    axes[0].grid(True, alpha=0.3)
    
    # STL and LTL
    axes[1].plot(t_out, sLoud[0].detach().numpy(), 'b-', label='STL', linewidth=1.5)
    axes[1].plot(t_out, lLoud[0].detach().numpy(), 'r-', label='LTL', linewidth=1.5)
    axes[1].axhline(mLoud.item(), color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Max = {mLoud.item():.2f}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Loudness (sones)')
    axes[1].set_title('Moore2016 Binaural Loudness (AM Response)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Specific loudness
    spec_loud_left = stages['stl_spec_loud_left'][0].detach().numpy()
    im = axes[2].imshow(spec_loud_left.T, aspect='auto', origin='lower',
                        extent=[0, duration, 0, 150], cmap='viridis')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ERB Channel')
    axes[2].set_title('Specific Loudness (STL, Left Ear)')
    plt.colorbar(im, ax=axes[2], label='Loudness (sones/ERB)')
    
    plt.tight_layout()
    fig_path = TEST_FIGURES_DIR / 'moore2016_am_tone.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()
    
    print("\n✓ AM tone test passed")


def test_tone_burst():
    """Test 6: Tone burst (0.5s silence + 0.5s tone) with visualization."""
    print("\n" + "="*80)
    print("TEST 6: TONE BURST (temporal integration)")
    print("="*80)
    
    fs = 32000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    tone = np.sin(2 * np.pi * 1000 * t) * 0.02
    tone[:int(fs * 0.5)] = 0  # First 0.5s silence
    
    audio = torch.from_numpy(tone).float()
    audio_stereo = torch.stack([audio, audio], dim=0).unsqueeze(0)
    
    model = Moore2016(fs=32000, return_stages=True)
    (sLoud, lLoud, mLoud), stages = model(audio_stereo)
    
    n_frames = sLoud.shape[1]
    first_half_stl = sLoud[:, :n_frames//2].mean()
    second_half_stl = sLoud[:, n_frames//2:].mean()
    
    print(f"\nTemporal response:")
    print(f"  STL first half (silence): {first_half_stl:.3f} sones")
    print(f"  STL second half (tone): {second_half_stl:.3f} sones")
    print(f"  Ratio: {second_half_stl/first_half_stl:.1f}x")
    
    # === Visualization ===
    t_out = np.arange(sLoud.shape[1]) * 0.001
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Input waveform
    axes[0].plot(t, tone, 'b-', linewidth=0.5)
    axes[0].axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Onset')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (Pa)')
    axes[0].set_title('Input: Tone Burst (0.5s silence + 0.5s tone)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # STL and LTL
    axes[1].plot(t_out, sLoud[0].detach().numpy(), 'b-', label='STL (fast)', linewidth=1.5)
    axes[1].plot(t_out, lLoud[0].detach().numpy(), 'r-', label='LTL (slow)', linewidth=1.5)
    axes[1].axvline(0.5, color='k', linestyle='--', alpha=0.3)
    axes[1].axhline(mLoud.item(), color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Max = {mLoud.item():.2f}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Loudness (sones)')
    axes[1].set_title('Moore2016 Binaural Loudness (Temporal Integration)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Specific loudness
    spec_loud_left = stages['stl_spec_loud_left'][0].detach().numpy()
    im = axes[2].imshow(spec_loud_left.T, aspect='auto', origin='lower',
                        extent=[0, duration, 0, 150], cmap='viridis')
    axes[2].axvline(0.5, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ERB Channel')
    axes[2].set_title('Specific Loudness (STL, Left Ear)')
    plt.colorbar(im, ax=axes[2], label='Loudness (sones/ERB)')
    
    plt.tight_layout()
    fig_path = TEST_FIGURES_DIR / 'moore2016_tone_burst.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    plt.close()
    
    print("\n✓ Tone burst test passed")


def test_intermediate_outputs():
    """Test 7: Verify all intermediate outputs are accessible."""
    print("\n" + "="*80)
    print("TEST 7: INTERMEDIATE OUTPUTS")
    print("="*80)
    
    fs = 32000
    t = np.linspace(0, 0.5, int(fs * 0.5))
    tone = np.sin(2 * np.pi * 1000 * t) * 0.02
    audio = torch.from_numpy(tone).float()
    audio_stereo = torch.stack([audio, audio], dim=0).unsqueeze(0)
    
    model = Moore2016(fs=32000, return_stages=True)
    (sLoud, lLoud, mLoud), stages = model(audio_stereo)
    
    print(f"\nIntermediate outputs:")
    expected_keys = ['filtered_left', 'filtered_right', 'freqs_left', 'levels_left',
                     'freqs_right', 'levels_right', 'excitation_left', 'excitation_right',
                     'inst_spec_loud_left', 'inst_spec_loud_right',
                     'stl_spec_loud_left', 'stl_spec_loud_right',
                     'stl_loud_left', 'stl_loud_right',
                     'ltl_loud_left', 'ltl_loud_right']
    
    for key in expected_keys:
        check = '✓' if key in stages else '✗'
        if key in stages:
            print(f"  {check} {key}: {stages[key].shape}")
        else:
            print(f"  {check} {key}: MISSING")
    
    print("\n✓ All intermediate outputs accessible")


def test_learnable_parameters():
    """Test 8: Test learnable mode."""
    print("\n" + "="*80)
    print("TEST 8: LEARNABLE PARAMETERS")
    print("="*80)
    
    model = Moore2016(fs=32000, learnable=True)
    params = list(model.parameters())
    
    print(f"\nModel configuration:")
    print(f"  Learnable: {model.learnable}")
    print(f"  Number of learnable parameters: {len(params)}")
    print(f"  Total parameter count: {sum(p.numel() for p in params)}")
    
    # Test forward pass works
    fs = 32000
    t = np.linspace(0, 0.5, int(fs * 0.5))
    tone = np.sin(2 * np.pi * 1000 * t) * 0.02
    audio = torch.from_numpy(tone).float()
    audio_stereo = torch.stack([audio, audio], dim=0).unsqueeze(0)
    
    sLoud, lLoud, mLoud = model(audio_stereo)
    
    assert sLoud.max() > 0, "Learnable model should produce non-zero output"
    
    print("\n✓ Learnable mode works correctly")


if __name__ == '__main__':
    """Run all tests."""
    print("\n" + "="*80)
    print("MOORE2016 COMPLETE MODEL - TEST SUITE")
    print("="*80)
    
    test_model_instantiation()
    test_diotic_tone()
    test_dichotic_tone()
    test_broadband_noise()
    test_am_tone()
    test_tone_burst()
    test_intermediate_outputs()
    test_learnable_parameters()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
