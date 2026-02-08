"""
Paulick2024 Model - Test Suite

Contents:
1. test_forward_validation: Basic forward pass validation
2. test_processing_chain_visualization: Processing chain visualization
3. test_decision_methods: Decision-making methods
4. test_stimulus_comparison: Stimulus comparison
5. test_intensity_encoding: Intensity encoding (rate-level functions)
6. test_frequency_selectivity: Frequency selectivity (tuning curves)

Structure:
- Forward pass with different configurations
- Processing chain visualization (all stages)
- Decision-making methods
- Stimulus comparison (tone, AM tone, noise, silence)
- Intensity encoding and frequency selectivity

Figures generated:
- paulick2024_processing_chain.png: Processing chain visualization
- paulick2024_stimulus_comparison.png: Stimulus comparison
- paulick2024_intensity_frequency.png: Intensity and frequency analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch_amt.models.paulick2024 import Paulick2024


def test_forward_validation():
    """
    Test 1: Forward pass validation
    
    Tests basic functionality:
    - Shape checks
    - NaN/Inf validation
    - Batch processing
    - Return stages functionality
    """
    print("="*80)
    print("TEST 1: Forward Pass Validation")
    print("="*80)
    
    # Model configuration
    fs = 44100
    duration = 0.5  # 500 ms
    n_samples = int(fs * duration)
    
    model = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=False,
        dtype=torch.float32
    )
    
    print(f"\nModel configuration:")
    print(model)
    
    # Test signal: 1 kHz pure tone (normalized -1 to 1)
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)
    tone = torch.sin(2 * np.pi * 1000 * t)
    
    # Single sample test
    print("\n" + "-"*80)
    print("Single sample test:")
    print("-"*80)
    x_single = tone.unsqueeze(0)  # [1, T]
    print(f"Input shape: {x_single.shape}")
    
    output_single = model(x_single)
    print(f"Output type: {type(output_single)}")
    print(f"Output length (channels): {len(output_single)}")
    
    # Check each channel
    all_valid = True
    for i, ch_output in enumerate(output_single):
        has_nan = torch.isnan(ch_output).any()
        has_inf = torch.isinf(ch_output).any()
        if has_nan or has_inf:
            print(f"  Channel {i}: shape={ch_output.shape}, NaN={has_nan}, Inf={has_inf} ❌")
            all_valid = False
        elif i < 3 or i >= len(output_single) - 3:  # Print first and last 3
            print(f"  Channel {i}: shape={ch_output.shape}, NaN={has_nan}, Inf={has_inf} ✓")
    
    if len(output_single) > 6:
        print(f"  ... (channels 3-{len(output_single)-4} omitted) ...")
    
    assert all_valid, "Some channels contain NaN or Inf values!"
    print(f"\nAll {len(output_single)} channels are valid ✓")
    
    # Batch processing test
    print("\n" + "-"*80)
    print("Batch processing test:")
    print("-"*80)
    batch_size = 4
    x_batch = tone.unsqueeze(0).repeat(batch_size, 1)  # [4, T]
    print(f"Input shape: {x_batch.shape}")
    
    output_batch = model(x_batch)
    print(f"Output length (channels): {len(output_batch)}")
    
    all_valid = True
    for i, ch_output in enumerate(output_batch):
        expected_shape = (batch_size, ch_output.shape[1], ch_output.shape[2])
        shape_ok = ch_output.shape[0] == batch_size
        has_nan = torch.isnan(ch_output).any()
        has_inf = torch.isinf(ch_output).any()
        
        if not shape_ok or has_nan or has_inf:
            print(f"  Channel {i}: shape={ch_output.shape}, NaN={has_nan}, Inf={has_inf} ❌")
            all_valid = False
        elif i < 3 or i >= len(output_batch) - 3:
            print(f"  Channel {i}: shape={ch_output.shape}, NaN={has_nan}, Inf={has_inf} ✓")
    
    if len(output_batch) > 6:
        print(f"  ... (channels 3-{len(output_batch)-4} omitted) ...")
    
    assert all_valid, "Batch processing failed!"
    print(f"\nBatch processing successful ✓")
    
    # Return stages test
    print("\n" + "-"*80)
    print("Return stages test:")
    print("-"*80)
    model_stages = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=True,
        dtype=torch.float32
    )
    
    output, stages = model_stages(x_single)
    print(f"Output type: {type(output)}, length: {len(output)}")
    print(f"Stages type: {type(stages)}, keys: {list(stages.keys())}")
    
    stage_names = {
        'outerear': 'Outer/Middle Ear',
        'drnl': 'DRNL Filterbank',
        'ihc': 'IHC Transduction',
        'adaptation': 'Adaptation',
        'resampled': 'Resampling',
    }
    
    print("\nIntermediate stages:")
    for key, name in stage_names.items():
        if key in stages:
            stage_data = stages[key]
            print(f"  {name}: shape={stage_data.shape}")
    
    print(f"\nFinal output (Modulation Filterbank): list of {len(output)} channels")
    print(f"  Channel 0 shape: {output[0].shape}")
    
    print("\n✓ All validation tests passed!")


def test_processing_chain_visualization():
    """
    Test 2: Processing chain visualization
    
    Generates a comprehensive figure showing all processing stages
    for a 1 kHz pure tone stimulus.
    """
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("TEST 2: Processing Chain Visualization")
    print("="*80)
    
    # Model configuration
    fs = 44100
    duration = 0.2  # 200 ms for clearer visualization
    n_samples = int(fs * duration)
    
    model = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=True,
        dtype=torch.float32
    )
    
    # Test signal: 1 kHz pure tone (normalized -1 to 1)
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)
    tone = torch.sin(2 * np.pi * 1000 * t)
    x = tone.unsqueeze(0)
    
    print(f"\nProcessing {duration*1000} ms tone @ 1000 Hz (normalized)")
    print(f"Input shape: {x.shape}")
    
    # Get all stages
    output, stages = model(x)
    
    # Extract data for visualization
    stage_data = []
    
    # Add intermediate stages from the stages dict
    if 'outerear' in stages:
        stage_data.append(stages['outerear'].squeeze(0).detach().cpu().numpy())
    if 'drnl' in stages:
        stage_data.append(stages['drnl'].squeeze(0).detach().cpu().numpy())
    if 'ihc' in stages:
        stage_data.append(stages['ihc'].squeeze(0).detach().cpu().numpy())
    if 'adaptation' in stages:
        stage_data.append(stages['adaptation'].squeeze(0).detach().cpu().numpy())
    if 'resampled' in stages:
        stage_data.append(stages['resampled'].squeeze(0).detach().cpu().numpy())
    
    # Add final modulation output - convert list to stacked array
    # Stack all channels: each output[i] is [M_i, T], stack to [F, max_M, T]
    mod_data = []
    for ch_out in output:
        mod_data.append(ch_out.detach().cpu().numpy())
    stage_data.append(mod_data)  # Keep as list for now
    
    print(f"\nStage shapes:")
    for i, data in enumerate(stage_data[:-1]):  # All except modulation
        print(f"  Stage {i+1}: {data.shape}")
    print(f"  Stage {len(stage_data)} (Modulation): list of {len(stage_data[-1])} channels")
    
    # Create figure with 3x4 grid
    print("\n" + "-"*80)
    print("Generating visualization figure...")
    print("-"*80)
    
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Stage 0: Input vs Outer/Middle Ear (waveform) - 10ms zoom at center
    ax1 = fig.add_subplot(gs[0, 0])
    input_data = x.squeeze(0).detach().cpu().numpy()
    outerear_data = stage_data[0] if len(stage_data) > 0 else input_data
    
    # Calculate 10ms zoom window at center
    total_duration_ms = duration * 1000  # Total duration in ms
    center_ms = total_duration_ms / 2
    zoom_start_ms = center_ms - 5  # 10ms window centered
    zoom_end_ms = center_ms + 5
    
    # Convert to sample indices
    zoom_start_idx = int(zoom_start_ms * fs / 1000)
    zoom_end_idx = int(zoom_end_ms * fs / 1000)
    
    # Extract zoom regions
    input_zoom = input_data[zoom_start_idx:zoom_end_idx]
    outerear_zoom = outerear_data[zoom_start_idx:zoom_end_idx]
    
    t_zoom = np.arange(len(input_zoom)) / fs * 1000 + zoom_start_ms
    ax1.plot(t_zoom, input_zoom, linewidth=0.8, label='Input', alpha=0.7)
    ax1.plot(t_zoom, outerear_zoom, linewidth=0.8, label='Outer/Middle Ear', alpha=0.7)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.set_title('Stage 0 & 1: Input vs Outer/Middle Ear (10ms zoom)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stage 1: DRNL Filterbank (time-frequency)
    ax2 = fig.add_subplot(gs[0, 1])
    drnl_data = stage_data[1]
    t_drnl = np.arange(drnl_data.shape[1]) / fs * 1000
    im2 = ax2.imshow(drnl_data, aspect='auto', origin='lower', 
                     extent=[t_drnl[0], t_drnl[-1], 0, drnl_data.shape[0]],
                     cmap='viridis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Channel #')
    ax2.set_title('Stage 2: DRNL Filterbank')
    plt.colorbar(im2, ax=ax2, label='Amplitude')
    
    # Stage 1b: DRNL Filterbank spectrum (mean over time)
    ax2b = fig.add_subplot(gs[0, 2])
    drnl_mean = np.mean(np.abs(drnl_data), axis=1)
    ax2b.plot(drnl_mean)
    ax2b.set_xlabel('Channel #')
    ax2b.set_ylabel('Mean Amplitude (a.u.)')
    ax2b.set_title('Stage 2: DRNL Mean Response')
    ax2b.grid(True, alpha=0.3)
    
    # Stage 2: IHC (time-frequency)
    ax3 = fig.add_subplot(gs[0, 3])
    ihc_data = stage_data[2]
    t_ihc = np.arange(ihc_data.shape[1]) / fs * 1000
    im3 = ax3.imshow(ihc_data, aspect='auto', origin='lower',
                     extent=[t_ihc[0], t_ihc[-1], 0, ihc_data.shape[0]],
                     cmap='viridis')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Channel #')
    ax3.set_title('Stage 3: IHC Transduction')
    plt.colorbar(im3, ax=ax3, label='Amplitude')
    
    # Stage 3: Adaptation (time-frequency)
    ax4 = fig.add_subplot(gs[1, 0])
    adapt_data = stage_data[3]
    t_adapt = np.arange(adapt_data.shape[1]) / fs * 1000
    im4 = ax4.imshow(adapt_data, aspect='auto', origin='lower',
                     extent=[t_adapt[0], t_adapt[-1], 0, adapt_data.shape[0]],
                     cmap='viridis')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Channel #')
    ax4.set_title('Stage 4: Adaptation')
    plt.colorbar(im4, ax=ax4, label='Amplitude')
    
    # Stage 3b: Adaptation spectrum
    ax4b = fig.add_subplot(gs[1, 1])
    adapt_mean = np.mean(adapt_data, axis=1)
    ax4b.plot(adapt_mean)
    ax4b.set_xlabel('Channel #')
    ax4b.set_ylabel('Mean Firing Rate (a.u.)')
    ax4b.set_title('Stage 4: Adaptation Mean Response')
    ax4b.grid(True, alpha=0.3)
    
    # Stage 4: Resampling (time-frequency)
    ax5 = fig.add_subplot(gs[1, 2])
    resample_data = stage_data[4]
    fs_resample = fs / 4
    t_resample = np.arange(resample_data.shape[1]) / fs_resample * 1000
    im5 = ax5.imshow(resample_data, aspect='auto', origin='lower',
                     extent=[t_resample[0], t_resample[-1], 0, resample_data.shape[0]],
                     cmap='viridis')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Channel #')
    ax5.set_title('Stage 5: Resampling (fs/4)')
    plt.colorbar(im5, ax=ax5, label='Amplitude')
    
    # Stage 5: Modulation Filterbank (3D representation)
    ax6 = fig.add_subplot(gs[1, 3])
    # Calculate on-frequency channel for 1000 Hz
    # Channels are spaced in ERB scale from 125 to 8000 Hz
    from scipy.interpolate import interp1d
    flow, fhigh, n_channels = 125.0, 8000.0, 50
    # ERB scale
    erb_low = 21.4 * np.log10(1 + flow / 228.7)
    erb_high = 21.4 * np.log10(1 + fhigh / 228.7)
    erb_cfs = np.linspace(erb_low, erb_high, n_channels)
    cfs = (10**(erb_cfs / 21.4) - 1) * 228.7
    # Find closest channel to 1000 Hz
    on_freq_ch = np.argmin(np.abs(cfs - 1000))
    on_freq_cf = cfs[on_freq_ch]
    
    mod_outputs = stage_data[5]  # List of channel outputs
    mod_data = mod_outputs[on_freq_ch]  # [B, n_mod_channels, T] or [n_mod_channels, T]
    if mod_data.ndim == 3:
        mod_data = mod_data.squeeze(0)  # Remove batch dimension if present
    t_mod = np.arange(mod_data.shape[1]) / fs_resample * 1000
    im6 = ax6.imshow(mod_data, aspect='auto', origin='lower',
                     extent=[t_mod[0], t_mod[-1], 0, mod_data.shape[0]],
                     cmap='viridis')
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Mod Channel #')
    ax6.set_title(f'Stage 6: Mod Filterbank (ch {on_freq_ch}, CF={on_freq_cf:.0f} Hz, on-frequency)')
    plt.colorbar(im6, ax=ax6, label='Amplitude')
    
    # Additional analysis plots (bottom row)
    
    # Channel activation over time - all channels as image
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(adapt_data, aspect='auto', origin='lower',
                     extent=[t_adapt[0], t_adapt[-1], 0, adapt_data.shape[0]],
                     cmap='viridis')
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Channel #')
    ax7.set_title('Adaptation: All Channels')
    plt.colorbar(im7, ax=ax7, label='Firing Rate')
    
    # Modulation spectrum (mean over time) - HEATMAP showing all channels and mod channels
    ax8 = fig.add_subplot(gs[2, 1])
    # Build 2D matrix: [n_channels, max_n_mod_channels] with padding
    # Each audio channel has different number of mod channels
    max_mod_channels = max([mod_outputs[ch].shape[1] if mod_outputs[ch].ndim == 3 else mod_outputs[ch].shape[0] 
                            for ch in range(len(mod_outputs))])
    mod_spectrum_matrix = np.zeros((len(mod_outputs), max_mod_channels))
    
    for ch_idx in range(len(mod_outputs)):
        mod_ch_data = mod_outputs[ch_idx]
        if mod_ch_data.ndim == 3:
            mod_ch_data = mod_ch_data.squeeze(0)
        mod_profile = np.mean(np.abs(mod_ch_data), axis=1)
        mod_spectrum_matrix[ch_idx, :len(mod_profile)] = mod_profile
    
    im8 = ax8.imshow(mod_spectrum_matrix, aspect='auto', origin='lower', cmap='viridis')
    ax8.set_xlabel('Mod Channel #')
    ax8.set_ylabel('Audio Channel #')
    ax8.set_title('Stage 6: Mean Modulation Spectrum (Heatmap)')
    plt.colorbar(im8, ax=ax8, label='Mean Amplitude (model units)')
    
    # Modulation channel profiles - line plot with continuous colormap
    ax9 = fig.add_subplot(gs[2, 2])
    # Plot all channels with continuous colormap
    n_channels_total = len(mod_outputs)
    cmap = plt.cm.viridis
    for ch_idx, mod_ch_data in enumerate(mod_outputs):
        if mod_ch_data.ndim == 3:
            mod_ch_data = mod_ch_data.squeeze(0)
        mod_profile = np.mean(np.abs(mod_ch_data), axis=1)
        color = cmap(ch_idx / n_channels_total)
        ax9.plot(mod_profile, linewidth=0.8, alpha=0.6, color=color)
    
    ax9.set_xlabel('Mod Channel #')
    ax9.set_ylabel('Mean Amplitude (model units)')
    ax9.set_title('Modulation Profiles: All Channels')
    ax9.grid(True, alpha=0.3)
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_channels_total-1))
    sm.set_array([])
    cbar9 = plt.colorbar(sm, ax=ax9, label='Audio Channel #')
    
    # Final output RMS per channel
    ax10 = fig.add_subplot(gs[2, 3])
    final_rms = []
    for m in mod_outputs:
        if m.ndim == 3:
            m = m.squeeze(0)
        final_rms.append(np.sqrt(np.mean(m**2)))
    final_rms = np.array(final_rms)
    ax10.plot(final_rms, 'o-', linewidth=1.2, markersize=4)
    ax10.set_xlabel('Channel #')
    ax10.set_ylabel('RMS (a.u.)')
    ax10.set_title('Final Output: RMS per Channel')
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle(f'Paulick2024 Model: Complete Processing Chain (1 kHz tone, {duration*1000:.0f} ms input)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = TEST_FIGURES_DIR / 'paulick2024_processing_chain.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_path}")
    plt.close()
    
    # Generate per-channel modulation filterbank plots
    print(f"\n  Generating modulation filterbank plots for all {len(mod_outputs)} channels...")
    
    output_dir = TEST_FIGURES_DIR / 'paulick2024_processing_chain_modulation_channels'
    output_dir.mkdir(exist_ok=True)
    
    for ch_idx, mod_ch_out in enumerate(mod_outputs):
        fc_val = model.fc[ch_idx].item()
        # mod_ch_out is already a numpy array from stage_data processing
        if isinstance(mod_ch_out, torch.Tensor):
            mod_ch = mod_ch_out.squeeze(0).cpu().numpy()
        else:
            mod_ch = mod_ch_out.squeeze(0) if mod_ch_out.ndim > 2 else mod_ch_out
        extent_mod = [0, duration * 1000, 0, mod_ch.shape[0]]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'Paulick2024 Modulation Filterbank - Channel {ch_idx}: fc = {fc_val:.1f} Hz', 
                         fontsize=12, fontweight='bold')
        
        im = ax_mod.imshow(mod_ch, aspect='auto', origin='lower', extent=extent_mod, cmap='plasma')
        ax_mod.set_title(f'{mod_ch.shape[0]} modulation bands')
        ax_mod.set_xlabel('Time [s]')
        ax_mod.set_ylabel('Modulation Band')
        plt.colorbar(im, ax=ax_mod, label='Amplitude [MU]')
        
        plt.tight_layout()
        output_filename = f'modulation_ch{ch_idx:02d}_fc{fc_val:05.0f}Hz.png'
        output_path_ch = output_dir / output_filename
        plt.savefig(output_path_ch, dpi=600, bbox_inches='tight')
        plt.close(fig_mod)
        
        if ch_idx == 0 or ch_idx == len(mod_outputs) - 1 or ch_idx % 10 == 0:
            print(f"    Saved: {output_filename}")
    
    print(f"\n✓ All {len(mod_outputs)} modulation filterbank plots saved to: {output_dir}/")


def test_decision_methods():
    """
    Test 3: Decision-making methods
    
    Tests all 6 decision methods with various configurations.
    """
    print("\n" + "="*80)
    print("TEST 3: Decision-Making Methods")
    print("="*80)
    
    # Model configuration
    fs = 44100
    duration = 0.5
    n_samples = int(fs * duration)
    
    model = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=False,
        
        dtype=torch.float32
    )
    
    # Create signal and noise (normalized -1 to 1)
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)
    signal = torch.sin(2 * np.pi * 1000 * t)
    noise = torch.randn_like(signal) * 0.1  # 10% amplitude noise
    
    x_signal = signal.unsqueeze(0)
    x_noise = noise.unsqueeze(0)
    
    # Get model outputs
    print("\nGenerating model outputs...")
    out_signal = model(x_signal)
    out_noise = model(x_noise)
    
    # Test 1: ROI Selection
    print("\n" + "-"*80)
    print("Method 1: ROI Selection")
    print("-"*80)
    
    # Time window selection
    roi_time = model.roi_selection(
        out_signal,
        time_window=(0.1, 0.4),
        channel_range=None,
        modulation_range=None
    )
    print(f"Time window (0.1-0.4 s):")
    print(f"  Original length: {out_signal[0].shape[2]} samples")
    print(f"  ROI length: {roi_time[0].shape[2]} samples")
    print(f"  Reduction: {(1 - roi_time[0].shape[2]/out_signal[0].shape[2])*100:.1f}%")
    
    # Channel selection
    roi_channels = model.roi_selection(
        out_signal,
        time_window=None,
        channel_range=(20, 30),
        modulation_range=None
    )
    print(f"\nChannel range (20-30):")
    print(f"  Original channels: {len(out_signal)}")
    print(f"  ROI channels: {len(roi_channels)}")
    
    # Modulation channel selection
    roi_mod = model.roi_selection(
        out_signal,
        time_window=None,
        channel_range=None,
        modulation_range=(3, 6)
    )
    print(f"\nModulation channel range (3-6):")
    print(f"  Original mod shape: {out_signal[0].shape}")
    print(f"  ROI mod shape: {roi_mod[0].shape}")
    
    # Test 2: Template Correlation
    print("\n" + "-"*80)
    print("Method 2: Template Correlation")
    print("-"*80)
    
    # Self-correlation (should be high)
    corr_self = model.template_correlation(
        out_signal, out_signal, 
        normalize=True
    )
    print(f"Self-correlation (normalized): {corr_self.item():.4f}")
    assert corr_self.item() > 0.95, "Self-correlation should be high!"
    
    # Signal-noise correlation (should be low)
    corr_noise = model.template_correlation(
        out_signal, out_noise,
        normalize=True
    )
    print(f"Signal-noise correlation (normalized): {corr_noise.item():.4f}")
    assert corr_noise.item() < 0.9, "Signal-noise correlation should be lower!"
    
    # Unnormalized correlation
    corr_unnorm = model.template_correlation(
        out_signal, out_signal,
        normalize=False
    )
    print(f"Self-correlation (unnormalized): {corr_unnorm.item():.4e}")
    
    # Test 3: Compute Decision Variable
    print("\n" + "-"*80)
    print("Method 3: Compute Decision Variable")
    print("-"*80)
    
    metrics = ['rms', 'mean', 'max', 'l2']
    for metric in metrics:
        dv = model.compute_decision_variable(
            out_signal, metric=metric
        )
        print(f"Metric '{metric}': {dv.item():.6e}")
    
    # Test 4: Make Decision
    print("\n" + "-"*80)
    print("Method 4: Make Decision")
    print("-"*80)
    
    dv_high = torch.tensor([1.5])
    dv_low = torch.tensor([0.5])
    threshold = 1.0
    
    decision_high = model.make_decision(dv_high, threshold)
    decision_low = model.make_decision(dv_low, threshold)
    
    print(f"Decision variable: {dv_high.item():.2f}, Threshold: {threshold:.2f}")
    print(f"  Decision: {decision_high.item()} (expected: 1)")
    print(f"Decision variable: {dv_low.item():.2f}, Threshold: {threshold:.2f}")
    print(f"  Decision: {decision_low.item()} (expected: 0)")
    
    assert decision_high.item() == 1, "High DV should trigger detection!"
    assert decision_low.item() == 0, "Low DV should not trigger detection!"
    
    # Test 5: Detection Task
    print("\n" + "-"*80)
    print("Method 5: Detection Task")
    print("-"*80)
    
    result_detect_dv = model.detection_task(
        x_signal, 
        noise_audio=x_noise,
        metric='rms',
        threshold=None  # Just get decision variable
    )
    
    print(f"Signal-noise decision variable: {result_detect_dv.item():.6e}")
    
    # With threshold
    decision = model.detection_task(
        x_signal,
        noise_audio=x_noise,
        metric='rms',
        threshold=0.0  # Threshold at 0
    )
    
    print(f"Detection decision (threshold=0): {decision.item()}")
    print(f"Expected: {1 if result_detect_dv.item() > 0 else 0}")
    
    # Test 6: Discrimination Task
    print("\n" + "-"*80)
    print("Method 6: Discrimination Task")
    print("-"*80)
    
    # Create two different tones
    signal1 = torch.sin(2 * np.pi * 1000 * t) * 10**(60/20) * 20e-6
    signal2 = torch.sin(2 * np.pi * 1500 * t) * 10**(60/20) * 20e-6
    x_signal1 = signal1.unsqueeze(0)
    x_signal2 = signal2.unsqueeze(0)
    
    # Discrimination with 'greater' rule
    decision_discrim = model.discrimination_task(
        x_signal1, 
        x_signal2,
        metric='rms',
        decision_rule='greater'
    )
    
    print(f"Discrimination decision (stimulus1 vs stimulus2): {decision_discrim.item()}")
    print(f"Interpretation: Stimulus {'1' if decision_discrim.item() == 1 else '2'} has higher response")
    
    # Self-discrimination (should choose randomly or based on noise)
    decision_self = model.discrimination_task(
        x_signal1, 
        x_signal1,
        metric='rms',
        decision_rule='greater'
    )
    print(f"\nSelf-discrimination:")
    print(f"  Decision: {decision_self.item()}")
    print(f"  (Same signal, decision based on numerical precision/noise)")
    
    print("\n✓ All decision methods working correctly!")


def test_stimulus_comparison():
    """
    Test 4: Stimulus comparison
    
    Compares model responses to different stimulus types:
    - Pure tone
    - Amplitude-modulated tone
    - White noise
    - Silence
    """
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    print("\n" + "="*80)
    print("TEST 4: Stimulus Comparison")
    print("="*80)
    
    # Model configuration
    fs = 44100
    duration = 0.3
    n_samples = int(fs * duration)
    
    model = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=False,
        
        dtype=torch.float32
    )
    
    print(f"\nGenerating stimuli ({duration*1000} ms, fs={fs} Hz)...")
    
    # Create stimuli (normalized -1 to 1)
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)
    
    # 1. Pure tone (1 kHz)
    tone = torch.sin(2 * np.pi * 1000 * t)
    tone_freq = 1000  # Hz
    
    # 2. AM tone (1 kHz carrier, 10 Hz modulation)
    am_tone = torch.sin(2 * np.pi * 1000 * t) * (1 + 0.5 * torch.sin(2 * np.pi * 10 * t))
    am_carrier_freq = 1000  # Hz
    am_mod_freq = 10  # Hz
    
    # 3. White noise (10% amplitude)
    noise = torch.randn_like(t) * 0.1
    
    # 4. Silence
    silence = torch.zeros_like(t)
    
    stimuli = {
        'Pure Tone': (tone, f'{tone_freq} Hz'),
        'AM Tone': (am_tone, f'Car. {am_carrier_freq} Hz, Mod. {am_mod_freq} Hz'),
        'Noise': (noise, 'White noise'),
        'Silence': (silence, '')
    }
    
    # Process all stimuli
    print("\n" + "-"*80)
    print("Processing stimuli...")
    print("-"*80)
    
    outputs = {}
    for name, (stimulus, info) in stimuli.items():
        x = stimulus.unsqueeze(0)
        out = model(x)
        outputs[name] = (out, stimulus, info)
        
        # Compute statistics
        rms_per_ch = []
        for ch_out in out:
            rms = torch.sqrt(torch.mean(ch_out**2))
            rms_per_ch.append(rms.item())
        
        mean_rms = np.mean(rms_per_ch)
        max_rms = np.max(rms_per_ch)
        
        print(f"{name:12s}: RMS mean={mean_rms:.6e}, max={max_rms:.6e}")
    
    # Create comparison figure
    print("\n" + "-"*80)
    print("Generating comparison figure...")
    print("-"*80)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fs_resample = fs / 4
    
    for idx, name in enumerate(stimuli.keys()):
        out, stimulus, info = outputs[name]
        
        # Convert to numpy
        out_np = []
        for ch_out in out:
            out_np.append(ch_out.squeeze(0).detach().cpu().numpy())
        out_stacked = np.array([o.mean(axis=0) for o in out_np])  # Mean over mod channels
        
        # Column 1: Input waveform (zoom 0-50ms)
        ax_wave = fig.add_subplot(gs[idx, 0])
        t_plot = np.arange(len(stimulus)) / fs * 1000
        
        # Zoom to first 50ms
        zoom_end_ms = 50
        zoom_end_idx = int(zoom_end_ms * fs / 1000)
        t_zoom = t_plot[:zoom_end_idx]
        stimulus_zoom = stimulus.numpy()[:zoom_end_idx]
        
        ax_wave.plot(t_zoom, stimulus_zoom, linewidth=0.8)
        ax_wave.set_xlabel('Time (ms)')
        ax_wave.set_ylabel('Amplitude')
        ax_wave.set_xlim(0, zoom_end_ms)
        # Create title with stimulus info
        title = f'Input Waveform: {name}'
        if info:
            title += f' ({info})'
        ax_wave.set_title(title)
        ax_wave.grid(True, alpha=0.3)
        
        # Column 2: Spectrogram (mean over mod channels)
        ax_spec = fig.add_subplot(gs[idx, 1])
        t_spec = np.arange(out_stacked.shape[1]) / fs_resample * 1000
        im = ax_spec.imshow(out_stacked, aspect='auto', origin='lower',
                           extent=[t_spec[0], t_spec[-1], 0, out_stacked.shape[0]],
                           cmap='viridis')
        ax_spec.set_xlabel('Time (ms)')
        ax_spec.set_ylabel('Channel #')
        ax_spec.set_title(f'{name}: Model Output')
        plt.colorbar(im, ax=ax_spec, label='Response (model units)')
        
        # Column 3: Mean channel response
        ax_mean = fig.add_subplot(gs[idx, 2])
        mean_response = np.mean(out_stacked, axis=1)
        ax_mean.plot(mean_response, 'o-', linewidth=1.2, markersize=4)
        ax_mean.set_xlabel('Channel #')
        ax_mean.set_ylabel('Mean Response (model units)')
        ax_mean.set_title(f'{name}: Mean Channel Activity')
        ax_mean.grid(True, alpha=0.3)
    
    plt.suptitle('Paulick2024 Model: Stimulus Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    output_path = TEST_FIGURES_DIR / 'paulick2024_stimulus_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_path}")
    plt.close()


def test_intensity_encoding():
    """
    Test 5: Intensity encoding
    
    Tests rate-level functions by varying stimulus intensity.
    """
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    print("\n" + "="*80)
    print("TEST 5: Intensity Encoding (Rate-Level Functions)")
    print("="*80)
    
    # Model configuration
    fs = 44100
    duration = 0.3
    n_samples = int(fs * duration)
    
    model = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=False,
        
        dtype=torch.float32
    )
    
    # Test parameters
    freq = 1000  # Hz
    levels_db = np.arange(20, 81, 10)  # 20 to 80 dB SPL in 10 dB steps
    
    print(f"\nTesting {freq} Hz tone at {len(levels_db)} intensity levels:")
    print(f"  Levels: {levels_db} dB SPL")
    
    # Generate stimuli and process
    print("\n" + "-"*80)
    print("Processing...")
    print("-"*80)
    
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)
    
    results = {
        'levels': levels_db,
        'mean_response': [],
        'max_response': [],
        'responses_per_channel': []
    }
    
    for level in levels_db:
        # Generate tone at specified level
        tone = torch.sin(2 * np.pi * freq * t) * 10**(level/20) * 20e-6
        x = tone.unsqueeze(0)
        
        # Process
        out = model(x)
        
        # Compute mean response across all channels and time
        responses = []
        for ch_out in out:
            mean_resp = torch.mean(torch.abs(ch_out)).item()
            responses.append(mean_resp)
        
        mean_response = np.mean(responses)
        max_response = np.max(responses)
        
        results['mean_response'].append(mean_response)
        results['max_response'].append(max_response)
        results['responses_per_channel'].append(responses)
        
        print(f"  {level} dB SPL: mean={mean_response:.6e}, max={max_response:.6e}")
    
    results['mean_response'] = np.array(results['mean_response'])
    results['max_response'] = np.array(results['max_response'])
    results['responses_per_channel'] = np.array(results['responses_per_channel'])
    
    # Check monotonicity
    print("\n" + "-"*80)
    print("Checking monotonicity...")
    print("-"*80)
    
    diffs = np.diff(results['mean_response'])
    monotonic = np.all(diffs > 0)
    print(f"Mean response monotonically increasing: {monotonic}")
    if not monotonic:
        print(f"  Warning: Found {np.sum(diffs <= 0)} non-increasing steps")
    
    # Store results for figure
    return results


def test_frequency_selectivity():
    """
    Test 6: Frequency selectivity
    
    Tests tuning curves by varying stimulus frequency.
    """
    print("\n" + "="*80)
    print("TEST 6: Frequency Selectivity (Tuning Curves)")
    print("="*80)
    
    # Model configuration
    fs = 44100
    duration = 0.3
    n_samples = int(fs * duration)
    
    model = Paulick2024(
        fs=fs,
        flow=125.0,
        fhigh=8000.0,
        n_channels=50,
        use_outerear=True,
        learnable=False,
        return_stages=False,
        
        dtype=torch.float32
    )
    
    # Test parameters
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 6000, 8000])  # Hz
    
    print(f"\nTesting {len(freqs)} frequencies (normalized):")
    print(f"  Frequencies: {freqs} Hz")
    
    # Generate stimuli and process
    print("\n" + "-"*80)
    print("Processing...")
    print("-"*80)
    
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)
    
    results = {
        'freqs': freqs,
        'responses': []  # [n_freqs, n_channels]
    }
    
    for freq in freqs:
        # Generate tone at specified frequency (normalized -1 to 1)
        tone = torch.sin(2 * np.pi * freq * t)
        x = tone.unsqueeze(0)
        
        # Process
        out = model(x)
        
        # Compute mean response per channel
        responses = []
        for ch_out in out:
            mean_resp = torch.mean(torch.abs(ch_out)).item()
            responses.append(mean_resp)
        
        results['responses'].append(responses)
        
        max_ch = np.argmax(responses)
        max_resp = responses[max_ch]
        print(f"  {freq:5.0f} Hz: max response at channel {max_ch} ({max_resp:.6e})")
    
    results['responses'] = np.array(results['responses'])
    
    # Check frequency selectivity
    print("\n" + "-"*80)
    print("Checking frequency selectivity...")
    print("-"*80)
    
    best_channels = np.argmax(results['responses'], axis=1)
    print(f"Best channels: {best_channels}")
    print(f"Range: {best_channels.min()} to {best_channels.max()}")
    
    # Check if higher frequencies activate higher channels
    is_monotonic = np.all(np.diff(best_channels) >= 0)
    print(f"Frequency-to-channel mapping is monotonic: {is_monotonic}")
    
    # Store results for figure
    return results


def generate_combined_figure(intensity_results, frequency_results):
    """
    Generate combined figure for intensity and frequency tests.
    """
    print("\n" + "="*80)
    print("Generating combined intensity/frequency figure...")
    print("="*80)
    
    # Generate test signals for visualization (normalized -1 to 1)
    fs = 44100
    duration = 0.3
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    
    # Intensity test signals (1 kHz tone, normalized to -1 to 1 for visualization)
    # Note: actual test uses dB SPL, these are for visual comparison only
    intensity_signals = []
    for level in intensity_results['levels']:
        signal = np.sin(2 * np.pi * 1000 * t)
        intensity_signals.append(signal)
    
    # Frequency test signals (different frequencies, normalized)
    frequency_signals = []
    for freq in frequency_results['freqs']:
        signal = np.sin(2 * np.pi * freq * t)
        frequency_signals.append(signal)
    
    fig = plt.figure(figsize=(20, 22))  # Increased height for 4 rows
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1, 2, 1, 2])
    
    # Row 0: Intensity test signals (spanning all 3 columns)
    
    ax0 = fig.add_subplot(gs[0, :])  # Span all columns
    t_ms = t * 1000
    for i, (level, signal) in enumerate(zip(intensity_results['levels'], intensity_signals)):
        # Offset each waveform vertically
        offset = i * 2.5
        ax0.plot(t_ms, signal + offset, linewidth=0.5, label=f'{level} dB', alpha=0.8)
    ax0.set_xlabel('Time (ms)')
    ax0.set_ylabel('Amplitude (offset)')
    ax0.set_title('Intensity Test Signals: 1 kHz Tone at Different Levels (Normalized)')
    ax0.set_xlim(0, 50)  # Show first 50ms
    ax0.legend(ncol=len(intensity_results['levels']), loc='upper right', fontsize=8)
    ax0.grid(True, alpha=0.3)
    
    # Row 1: Intensity encoding
    
    # Plot 1: Rate-level function (mean)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(intensity_results['levels'], intensity_results['mean_response'], 
             'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Level (dB SPL)')
    ax1.set_ylabel('Mean Response (model units)')
    ax1.set_title('Rate-Level Function (Mean)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rate-level function (max)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(intensity_results['levels'], intensity_results['max_response'],
             'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Level (dB SPL)')
    ax2.set_ylabel('Max Response (model units)')
    ax2.set_title('Rate-Level Function (Max)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Response map (all channels)
    ax3 = fig.add_subplot(gs[1, 2])
    im1 = ax3.imshow(intensity_results['responses_per_channel'].T, 
                     aspect='auto', origin='lower', cmap='viridis',
                     extent=[intensity_results['levels'][0], 
                            intensity_results['levels'][-1],
                            0, intensity_results['responses_per_channel'].shape[1]])
    ax3.set_xlabel('Level (dB SPL)')
    ax3.set_ylabel('Channel #')
    ax3.set_title('Response Map: Intensity')
    plt.colorbar(im1, ax=ax3, label='Response (model units)')
    
    # Row 2: Frequency test signals (spanning all 3 columns)
    
    ax2_signals = fig.add_subplot(gs[2, :])  # Span all columns
    for i, (freq, signal) in enumerate(zip(frequency_results['freqs'], frequency_signals)):
        # Offset each waveform vertically
        offset = i * 2.5
        ax2_signals.plot(t_ms, signal + offset, linewidth=0.5, label=f'{freq:.0f} Hz', alpha=0.8)
    ax2_signals.set_xlabel('Time (ms)')
    ax2_signals.set_ylabel('Amplitude (offset)')
    ax2_signals.set_title('Frequency Test Signals: Different Frequencies (Normalized)')
    ax2_signals.set_xlim(0, 50)  # Show first 50ms
    ax2_signals.legend(ncol=len(frequency_results['freqs']), loc='upper right', fontsize=8)
    ax2_signals.grid(True, alpha=0.3)
    
    # Row 3: Frequency selectivity
    
    # Plot 4: Tuning curves (ALL channels with colormap)
    ax4 = fig.add_subplot(gs[3, 0])
    # Plot all channels with continuous colormap
    n_channels = frequency_results['responses'].shape[1]
    cmap = plt.cm.viridis
    for ch in range(n_channels):
        color = cmap(ch / n_channels)
        ax4.plot(frequency_results['freqs'], 
                frequency_results['responses'][:, ch],
                linewidth=0.8, alpha=0.6, color=color)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Response (model units)')
    ax4.set_xscale('log')
    ax4.set_title('Tuning Curves (All Channels)')
    ax4.grid(True, alpha=0.3)
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_channels-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, label='Channel #')
    
    # Plot 5: Best frequency per channel
    ax5 = fig.add_subplot(gs[3, 1])
    best_freqs = []
    for ch in range(frequency_results['responses'].shape[1]):
        best_freq_idx = np.argmax(frequency_results['responses'][:, ch])
        best_freqs.append(frequency_results['freqs'][best_freq_idx])
    ax5.plot(best_freqs, 'o-', linewidth=2, markersize=6, color='green')
    ax5.set_xlabel('Channel #')
    ax5.set_ylabel('Best Frequency (Hz)')
    ax5.set_yscale('log')
    ax5.set_title('Best Frequency per Channel')
    # Add more yticks for better readability
    ax5.set_yticks([100, 200, 500, 1000, 2000, 4000, 8000])
    ax5.set_yticklabels(['100', '200', '500', '1k', '2k', '4k', '8k'])
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Response map (all frequencies)
    ax6 = fig.add_subplot(gs[3, 2])
    im2 = ax6.imshow(frequency_results['responses'].T, 
                     aspect='auto', origin='lower', cmap='hot')
    ax6.set_xlabel('Frequency #')
    ax6.set_ylabel('Channel #')
    ax6.set_title('Response Map: Frequency')
    ax6.set_xticks(range(len(frequency_results['freqs'])))
    ax6.set_xticklabels([f"{f:.0f}" for f in frequency_results['freqs']], rotation=45)
    plt.colorbar(im2, ax=ax6, label='Response (model units)')
    
    plt.suptitle('Paulick2024 Model: Intensity Encoding & Frequency Selectivity', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = TEST_FIGURES_DIR / 'paulick2024_intensity_frequency.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("PAULICK2024 MODEL TEST SUITE")
    print("="*80)
    print("\nThis test suite validates the complete Paulick2024 CASP model.")
    print("Tests include:")
    print("  1. Forward pass validation")
    print("  2. Processing chain visualization")
    print("  3. Decision-making methods")
    print("  4. Stimulus comparison")
    print("  5. Intensity encoding")
    print("  6. Frequency selectivity")
    print("\n" + "="*80)
    
    # Run all tests
    test_forward_validation()
    test_processing_chain_visualization()
    test_decision_methods()
    test_stimulus_comparison()
    
    # Run intensity and frequency tests (return results for combined figure)
    intensity_results = test_intensity_encoding()
    frequency_results = test_frequency_selectivity()
    
    # Generate combined figure
    generate_combined_figure(intensity_results, frequency_results)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated figures:")
    print("  1. paulick2024_processing_chain.png")
    print("  2. paulick2024_stimulus_comparison.png")
    print("  3. paulick2024_intensity_frequency.png")
    print("\n✓ Test suite finished.\n")
