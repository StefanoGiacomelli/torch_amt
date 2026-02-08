"""
King2019 Auditory Model - Test Suite

Contents:
1. test_king2019_1khz_am_local_analysis: Local analysis (basef ± 2 ERB)
2. test_king2019_1khz_am_wideband_analysis: Wideband analysis (full range)
3. test_king2019_1khz_local_fixed_modbank: Local with fixed modulation filterbank
4. test_king2019_1khz_am_wideband_fixed_modbank: Wideband with fixed modulation filterbank

Structure:
- AM-modulated 1 kHz tone test signal
- Stage-by-stage processing visualization
- Modulation filterbank channel analysis
- Q-factor and configuration variations

Figures generated (per test):
- king2019_{config}.png: 5-panel main analysis (600 DPI)
- king2019_{config}_modulation_channels/: Per-channel modulation analysis (600 DPI)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_amt.models import King2019


def test_king2019_1khz_am_local_analysis():
    """Test King2019 with 1 kHz AM tone - Local analysis (default config)."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("KING2019 Model Test: 1 kHz AM Tone - Local Analysis")
    print("="*80)
    
    # Parameters
    fs = 48000
    duration = 0.5
    fc = 1000  # Carrier frequency
    fm = 20    # Modulation frequency
    m = 0.8    # Modulation depth
    
    print("\nTest Parameters:")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Carrier frequency: {fc} Hz")
    print(f"  Modulation frequency: {fm} Hz")
    print(f"  Modulation depth: {m}")
    
    # Generate AM signal: s(t) = (1 + m*sin(2*pi*fm*t)) * sin(2*pi*fc*t)
    t = torch.linspace(0, duration, int(fs * duration))
    carrier = torch.sin(2 * torch.pi * fc * t)
    envelope = 1 + m * torch.sin(2 * torch.pi * fm * t)
    signal = envelope * carrier
    
    # Instantiate model with LOCAL analysis (basef ± 2 ERB)
    model = King2019(fs=fs, basef=fc, return_stages=True, 
                     dboffset=120.0, compression_knee_db=80.0)
    
    print("\nModel Configuration:")
    print(f"  Analysis type: LOCAL (basef ± 2 ERB)")
    print(f"  Frequency range: {model.flow:.1f} - {model.fhigh:.1f} Hz")
    print(f"  Number of frequency channels: {model.num_channels}")
    print(f"  Compression type: {model.compression_type}")
    print(f"  Compression knee: {model.compression_knee_db} dB")
    print(f"  Compression dboffset: {model.dboffset} dB")
    print(f"  Modulation filterbank: AUTO (Q={model.modbank_qfactor})")
    print(f"  Number of modulation channels: {len(model.mfc)}")
    
    # Process through model (get intermediate stages)
    with torch.no_grad():
        output, stages = model(signal)
    
    print("\n" + "="*80)
    print("Processing Stages Output")
    print("="*80)
    
    # Stage outputs
    print(f"\n1. Gammatone Filterbank:")
    print(f"   Shape: {stages['gtone_response'].shape}")
    print(f"   Range: [{stages['gtone_response'].min():.6f}, {stages['gtone_response'].max():.6f}]")
    
    print(f"\n2. Compression:")
    print(f"   Shape: {stages['compressed_response'].shape}")
    print(f"   Range: [{stages['compressed_response'].min():.6f}, {stages['compressed_response'].max():.6f}]")
    
    print(f"\n3. IHC Envelope:")
    print(f"   Shape: {stages['ihc'].shape}")
    print(f"   Range: [{stages['ihc'].min():.6f}, {stages['ihc'].max():.6f}]")
    
    print(f"\n4. Adaptation:")
    print(f"   Shape: {stages['adapted_response'].shape}")
    print(f"   Range: [{stages['adapted_response'].min():.6f}, {stages['adapted_response'].max():.6f}]")
    
    print(f"\n5. Modulation Filterbank:")
    print(f"   Shape: {output.shape}")
    print(f"   Range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('King2019: 1 kHz AM Tone - Local Analysis (basef ± 2 ERB)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Input signal
    axes[0].plot(t.numpy()[:2000], signal.numpy()[:2000], 'b-', linewidth=1)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'1. Input Signal: AM Tone (fc={fc}Hz, fm={fm}Hz, m={m})')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Gammatone filterbank output
    filterbank_data = stages['gtone_response'].squeeze(0).numpy()  # [F, T]
    time_axis = np.linspace(0, duration, filterbank_data.shape[1])
    im2 = axes[1].imshow(filterbank_data, aspect='auto', origin='lower', 
                         extent=[0, duration, 0, model.num_channels],
                         cmap='seismic', interpolation='nearest')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Channel')
    axes[1].set_title('2. Gammatone Filterbank Output')
    plt.colorbar(im2, ax=axes[1], label='Amplitude')
    
    # Plot 3: Compression output
    compression_data = stages['compressed_response'].squeeze(0).numpy()
    im3 = axes[2].imshow(compression_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='seismic', interpolation='nearest')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Channel')
    axes[2].set_title(f'3. Compression Output ({model.compression_type})')
    plt.colorbar(im3, ax=axes[2], label='Amplitude')
    
    # Plot 4: IHC envelope
    ihc_data = stages['ihc'].squeeze(0).numpy()
    im4 = axes[3].imshow(ihc_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='hot', interpolation='nearest')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Channel')
    axes[3].set_title('4. IHC Envelope Extraction')
    plt.colorbar(im4, ax=axes[3], label='Amplitude')
    
    # Plot 5: Adaptation
    adapted_data = stages['adapted_response'].squeeze(0).numpy()
    im5 = axes[4].imshow(adapted_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='viridis', interpolation='nearest')
    axes[4].set_xlabel('Time [s]')
    axes[4].set_ylabel('Channel')
    axes[4].set_title('5. Adaptation (High-pass filtered)')
    plt.colorbar(im5, ax=axes[4], label='Amplitude [MU]')
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'king2019_1khz_am_local_analysis.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    
    # Save modulation filterbank channels to separate folder
    modulation_data = output.squeeze(0).numpy()  # [T, F, M]
    output_dir = TEST_FIGURES_DIR / 'king2019_local_modulation_channels'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n  [6] Generating modulation filterbank plots for all {model.num_channels} channels...")
    
    for ch in range(model.num_channels):
        fc_val = model.fc[ch].item()
        mod_ch_data = modulation_data[:, ch, :].T  # [M, T]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'King2019 Modulation Filterbank - Channel {ch}: fc = {fc_val:.1f} Hz', 
                         fontsize=12, fontweight='bold')
        
        im = ax_mod.imshow(mod_ch_data, aspect='auto', origin='lower',
                          extent=[0, duration, 0, len(model.mfc)],
                          cmap='plasma', interpolation='nearest')
        ax_mod.set_title(f'{len(model.mfc)} modulation bands')
        ax_mod.set_xlabel('Time [s]')
        ax_mod.set_ylabel('Modulation Channel')
        plt.colorbar(im, ax=ax_mod, label='Amplitude [MU]')
        
        plt.tight_layout()
        output_filename = f'modulation_ch{ch:02d}_fc{fc_val:05.0f}Hz.png'
        output_path_ch = output_dir / output_filename
        plt.savefig(output_path_ch, dpi=600, bbox_inches='tight')
        plt.close(fig_mod)
        
        if ch == 0 or ch == model.num_channels - 1:
            print(f"    Saved: {output_filename}")
    
    print(f"\nAll {model.num_channels} modulation filterbank plots saved to: {output_dir}/")
    print(f"\nPlot saved: {output_path}")
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


def test_king2019_1khz_am_wideband_analysis():
    """Test King2019 with 1 kHz AM tone - Wideband analysis (like DAU1997)."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("KING2019 Model Test: 1 kHz AM Tone - Wideband Analysis")
    print("="*80)
    
    # Parameters
    fs = 48000
    duration = 0.5
    fc = 1000
    fm = 20
    m = 0.8
    
    print("\nTest Parameters:")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Carrier frequency: {fc} Hz")
    print(f"  Modulation frequency: {fm} Hz")
    print(f"  Modulation depth: {m}")
    
    # Generate AM signal
    t = torch.linspace(0, duration, int(fs * duration))
    carrier = torch.sin(2 * torch.pi * fc * t)
    envelope = 1 + m * torch.sin(2 * torch.pi * fm * t)
    signal = envelope * carrier
    
    # Instantiate model with WIDEBAND analysis
    model = King2019(fs=fs, flow=20, fhigh=20000, return_stages=True,
                     dboffset=120.0, compression_knee_db=80.0)
    
    print("\nModel Configuration:")
    print(f"  Analysis type: WIDEBAND (like DAU1997)")
    print(f"  Frequency range: {model.flow:.1f} - {model.fhigh:.1f} Hz")
    print(f"  Number of frequency channels: {model.num_channels}")
    print(f"  Compression knee: {model.compression_knee_db} dB")
    print(f"  Compression dboffset: {model.dboffset} dB")
    print(f"  Modulation filterbank: AUTO (Q={model.modbank_qfactor})")
    print(f"  Number of modulation channels: {len(model.mfc)}")
    
    # Process through model
    with torch.no_grad():
        output, stages = model(signal)
    
    print("\n" + "="*80)
    print("Processing Stages Output")
    print("="*80)
    
    print(f"\n1. Gammatone Filterbank: {stages['gtone_response'].shape}")
    print(f"2. Compression: {stages['compressed_response'].shape}")
    print(f"3. IHC Envelope: {stages['ihc'].shape}")
    print(f"4. Adaptation: {stages['adapted_response'].shape}")
    print(f"5. Modulation: {output.shape}")
    
    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('King2019: 1 kHz AM Tone - Wideband Analysis (20 Hz - 20 kHz)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Input signal
    axes[0].plot(t.numpy()[:2000], signal.numpy()[:2000], 'b-', linewidth=1)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'1. Input Signal: AM Tone')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2-6: Same structure
    filterbank_data = stages['gtone_response'].squeeze(0).numpy()
    im2 = axes[1].imshow(filterbank_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='seismic', interpolation='nearest')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Channel')
    axes[1].set_title(f'2. Gammatone Filterbank ({model.num_channels} channels)')
    plt.colorbar(im2, ax=axes[1], label='Amplitude')
    
    compression_data = stages['compressed_response'].squeeze(0).numpy()
    im3 = axes[2].imshow(compression_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='seismic', interpolation='nearest')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Channel')
    axes[2].set_title('3. Compression Output')
    plt.colorbar(im3, ax=axes[2], label='Amplitude')
    
    ihc_data = stages['ihc'].squeeze(0).numpy()
    im4 = axes[3].imshow(ihc_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='hot', interpolation='nearest')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Channel')
    axes[3].set_title('4. IHC Envelope')
    plt.colorbar(im4, ax=axes[3], label='Amplitude')
    
    adapted_data = stages['adapted_response'].squeeze(0).numpy()
    im5 = axes[4].imshow(adapted_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='viridis', interpolation='nearest')
    axes[4].set_xlabel('Time [s]')
    axes[4].set_ylabel('Channel')
    axes[4].set_title('5. Adaptation')
    plt.colorbar(im5, ax=axes[4], label='Amplitude [MU]')
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'king2019_1khz_am_wideband_analysis.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    
    # Save modulation filterbank channels to separate folder
    modulation_data = output.squeeze(0).numpy()  # [T, F, M]
    output_dir = TEST_FIGURES_DIR / 'king2019_wideband_modulation_channels'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n  [6] Generating modulation filterbank plots for all {model.num_channels} channels...")
    
    for ch in range(model.num_channels):
        fc_val = model.fc[ch].item()
        mod_ch_data = modulation_data[:, ch, :].T  # [M, T]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'King2019 Modulation Filterbank - Channel {ch}: fc = {fc_val:.1f} Hz', 
                         fontsize=12, fontweight='bold')
        
        im = ax_mod.imshow(mod_ch_data, aspect='auto', origin='lower',
                          extent=[0, duration, 0, len(model.mfc)],
                          cmap='plasma', interpolation='nearest')
        ax_mod.set_title(f'{len(model.mfc)} modulation bands')
        ax_mod.set_xlabel('Time [s]')
        ax_mod.set_ylabel('Modulation Channel')
        plt.colorbar(im, ax=ax_mod, label='Amplitude [MU]')
        
        plt.tight_layout()
        output_filename = f'modulation_ch{ch:02d}_fc{fc_val:05.0f}Hz.png'
        output_path_ch = output_dir / output_filename
        plt.savefig(output_path_ch, dpi=600, bbox_inches='tight')
        plt.close(fig_mod)
        
        if ch == 0 or ch == model.num_channels - 1 or ch % 20 == 0:
            print(f"    Saved: {output_filename}")
    
    print(f"\nAll {model.num_channels} modulation filterbank plots saved to: {output_dir}/")
    print(f"\nPlot saved: {output_path}")
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


def test_king2019_1khz_local_fixed_modbank():
    """Test King2019 with 1 kHz AM tone - Local analysis with fixed modulation filters."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("KING2019 Model Test: 1 kHz AM Tone - Local + Fixed Modulation Filters")
    print("="*80)
    
    # Parameters
    fs = 48000
    duration = 0.5
    fc = 1000
    fm = 20
    m = 0.8
    
    print("\nTest Parameters:")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Carrier/modulation: {fc} Hz / {fm} Hz / m={m}")
    
    # Generate AM signal
    t = torch.linspace(0, duration, int(fs * duration))
    carrier = torch.sin(2 * torch.pi * fc * t)
    envelope = 1 + m * torch.sin(2 * torch.pi * fm * t)
    signal = envelope * carrier
    
    # Instantiate model with FIXED number of modulation filters
    model = King2019(fs=fs, basef=fc, modbank_nmod=15, return_stages=True,
                     dboffset=120.0, compression_knee_db=80.0)
    
    print("\nModel Configuration:")
    print(f"  Analysis type: LOCAL (basef ± 2 ERB)")
    print(f"  Frequency range: {model.flow:.1f} - {model.fhigh:.1f} Hz")
    print(f"  Number of frequency channels: {model.num_channels}")
    print(f"  Compression knee: {model.compression_knee_db} dB")
    print(f"  Compression dboffset: {model.dboffset} dB")
    print(f"  Modulation filterbank: FIXED (N={len(model.mfc)})")
    print(f"  Modulation frequencies: {model.mfc.numpy()}")
    
    # Process through model
    with torch.no_grad():
        output, stages = model(signal)
    
    print("\n" + "="*80)
    print("Processing Stages Output")
    print("="*80)
    
    print(f"\n1. Gammatone: {stages['gtone_response'].shape}")
    print(f"2. Compression: {stages['compressed_response'].shape}")
    print(f"3. IHC: {stages['ihc'].shape}")
    print(f"4. Adaptation: {stages['adapted_response'].shape}")
    print(f"5. Modulation: {output.shape}")
    
    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('King2019: 1 kHz AM Tone - Fixed 15 Modulation Filters', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes[0].plot(t.numpy()[:2000], signal.numpy()[:2000], 'b-', linewidth=1)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('1. Input Signal')
    axes[0].grid(True, alpha=0.3)
    
    filterbank_data = stages['gtone_response'].squeeze(0).numpy()
    im2 = axes[1].imshow(filterbank_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='seismic', interpolation='nearest')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Channel')
    axes[1].set_title('2. Gammatone Filterbank')
    plt.colorbar(im2, ax=axes[1], label='Amplitude')
    
    compression_data = stages['compressed_response'].squeeze(0).numpy()
    im3 = axes[2].imshow(compression_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='seismic', interpolation='nearest')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Channel')
    axes[2].set_title('3. Compression')
    plt.colorbar(im3, ax=axes[2], label='Amplitude')
    
    ihc_data = stages['ihc'].squeeze(0).numpy()
    im4 = axes[3].imshow(ihc_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='hot', interpolation='nearest')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Channel')
    axes[3].set_title('4. IHC Envelope')
    plt.colorbar(im4, ax=axes[3], label='Amplitude')
    
    adapted_data = stages['adapted_response'].squeeze(0).numpy()
    im5 = axes[4].imshow(adapted_data, aspect='auto', origin='lower',
                         extent=[0, duration, 0, model.num_channels],
                         cmap='viridis', interpolation='nearest')
    axes[4].set_xlabel('Time [s]')
    axes[4].set_ylabel('Channel')
    axes[4].set_title('5. Adaptation')
    plt.colorbar(im5, ax=axes[4], label='Amplitude [MU]')
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'king2019_1khz_local_fixed_modbank.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    
    # Save modulation filterbank channels to separate folder
    modulation_data = output.squeeze(0).numpy()  # [T, F, M]
    output_dir = TEST_FIGURES_DIR / 'king2019_local_fixed_modulation_channels'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n  [6] Generating modulation filterbank plots for all {model.num_channels} channels...")
    
    for ch in range(model.num_channels):
        fc_val = model.fc[ch].item()
        mod_ch_data = modulation_data[:, ch, :].T  # [M, T]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'King2019 Modulation Filterbank - Channel {ch}: fc = {fc_val:.1f} Hz', 
                         fontsize=12, fontweight='bold')
        
        im = ax_mod.imshow(mod_ch_data, aspect='auto', origin='lower',
                          extent=[0, duration, 0, len(model.mfc)],
                          cmap='plasma', interpolation='nearest')
        ax_mod.set_title(f'Fixed N={len(model.mfc)} modulation bands')
        ax_mod.set_xlabel('Time [s]')
        ax_mod.set_ylabel('Modulation Channel')
        plt.colorbar(im, ax=ax_mod, label='Amplitude [MU]')
        
        plt.tight_layout()
        output_filename = f'modulation_ch{ch:02d}_fc{fc_val:05.0f}Hz.png'
        output_path_ch = output_dir / output_filename
        plt.savefig(output_path_ch, dpi=600, bbox_inches='tight')
        plt.close(fig_mod)
        
        if ch == 0 or ch == model.num_channels - 1:
            print(f"    Saved: {output_filename}")
    
    print(f"\nAll {model.num_channels} modulation filterbank plots saved to: {output_dir}/")
    print(f"\nPlot saved: {output_path}")
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


def test_king2019_1khz_am_wideband_fixed_modbank():
    """Test King2019 - Wideband analysis with fixed modulation filters."""
    
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    print("\n\n" + "="*80)
    print("KING2019 Model Test: 1 kHz AM Tone - Wideband + Fixed Modulation")
    print("="*80)
    
    # Parameters
    fs = 48000
    duration = 0.5
    fc = 1000
    fm = 20
    m = 0.8
    
    # Generate AM signal
    t = torch.linspace(0, duration, int(fs * duration))
    carrier = torch.sin(2 * torch.pi * fc * t)
    envelope = 1 + m * torch.sin(2 * torch.pi * fm * t)
    signal = envelope * carrier
    
    # Wideband + Fixed modulation
    model = King2019(fs=fs, flow=20, fhigh=20000, modbank_nmod=15, return_stages=True,
                     dboffset=120.0, compression_knee_db=80.0)
    
    print("\nModel Configuration:")
    print(f"  Analysis: WIDEBAND + Fixed Modulation Filters")
    print(f"  Frequency: {model.flow:.1f} - {model.fhigh:.1f} Hz ({model.num_channels} ch)")
    print(f"  Compression knee: {model.compression_knee_db} dB")
    print(f"  Compression dboffset: {model.dboffset} dB")
    print(f"  Modulation: Fixed N={len(model.mfc)}")
    
    with torch.no_grad():
        output, stages = model(signal)
    
    # Visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('King2019: Wideband (20 Hz - 20 kHz) + Fixed 15 Modulation Filters', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes[0].plot(t.numpy()[:2000], signal.numpy()[:2000], 'b-', linewidth=1)
    axes[0].set_title('1. Input Signal')
    axes[0].grid(True, alpha=0.3)
    
    for i, (data, title, cmap) in enumerate([
        (stages['gtone_response'].squeeze(0).numpy(), '2. Gammatone', 'seismic'),
        (stages['compressed_response'].squeeze(0).numpy(), '3. Compression', 'seismic'),
        (stages['ihc'].squeeze(0).numpy(), '4. IHC', 'hot'),
        (stages['adapted_response'].squeeze(0).numpy(), '5. Adaptation', 'viridis')
    ], 1):
        im = axes[i].imshow(data, aspect='auto', origin='lower',
                           extent=[0, duration, 0, model.num_channels],
                           cmap=cmap, interpolation='nearest')
        axes[i].set_title(title)
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'king2019_1khz_am_wideband_fixed_modbank.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    
    # Save modulation filterbank channels to separate folder
    modulation_data = output.squeeze(0).numpy()  # [T, F, M]
    output_dir = TEST_FIGURES_DIR / 'king2019_wideband_fixed_modulation_channels'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n  [6] Generating modulation filterbank plots for all {model.num_channels} channels...")
    
    for ch in range(model.num_channels):
        fc_val = model.fc[ch].item()
        mod_ch_data = modulation_data[:, ch, :].T  # [M, T]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'King2019 Modulation Filterbank - Channel {ch}: fc = {fc_val:.1f} Hz', 
                         fontsize=12, fontweight='bold')
        
        im = ax_mod.imshow(mod_ch_data, aspect='auto', origin='lower',
                          extent=[0, duration, 0, len(model.mfc)],
                          cmap='plasma', interpolation='nearest')
        ax_mod.set_title(f'Fixed N={len(model.mfc)} modulation bands')
        ax_mod.set_xlabel('Time [s]')
        ax_mod.set_ylabel('Modulation Channel')
        plt.colorbar(im, ax=ax_mod, label='Amplitude [MU]')
        
        plt.tight_layout()
        output_filename = f'modulation_ch{ch:02d}_fc{fc_val:05.0f}Hz.png'
        output_path_ch = output_dir / output_filename
        plt.savefig(output_path_ch, dpi=600, bbox_inches='tight')
        plt.close(fig_mod)
        
        if ch == 0 or ch == model.num_channels - 1 or ch % 20 == 0:
            print(f"    Saved: {output_filename}")
    
    print(f"\nAll {model.num_channels} modulation filterbank plots saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    # Run all configuration tests
    test_king2019_1khz_am_local_analysis()
    test_king2019_1khz_am_wideband_analysis()
    test_king2019_1khz_local_fixed_modbank()
    test_king2019_1khz_am_wideband_fixed_modbank()
    
    print("\n\n" + "="*80)
    print("ALL KING2019 CONFIGURATION TESTS COMPLETED!")
    print("="*80)

