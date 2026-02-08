"""
DAU1997 Model - Test Suite

Contents:
1. test_dau1997_processing_pipeline: Complete DAU1997 pipeline test with 1kHz signal

Structure:
- Gammatone filterbank output
- IHC envelope extraction
- Adaptation stage
- Modulation filterbank (saved per-channel)

Figures generated:
- dau1997_1khz.png: 4-panel processing pipeline overview
- dau1997_modulation_channels/*.png: Per-channel modulation filterbank output plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.models.dau1997 import Dau1997


def test_dau1997_processing_pipeline():
    """Test Dau1997 model processing pipeline with 1kHz tone."""
    
    # Create test_figures directory and subdirectory
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate 1kHz signal
    freq = 1000
    signal = np.sin(2 * np.pi * freq * t)
    x = torch.tensor(signal, dtype=torch.float32)

    print("="*80)
    print("Dau1997 MODEL TEST - 1kHz Tone")
    print("="*80)
    print(f"\nParameters:")
    print(f"  fs = {fs} Hz")
    print(f"  duration = {duration} s")
    print(f"  tone frequency = {freq} Hz")
    print(f"  samples = {len(x)}")

    # Create model
    model = Dau1997(fs=fs)
    print(f"\nDau1997 Model:")
    print(f"  Filterbank channels: {model.filterbank.num_channels}")
    print(f"  fc range: {model.filterbank.fc[0]:.2f} - {model.filterbank.fc[-1]:.2f} Hz")

    # Processing
    print(f"\nProcessing...")
    with torch.no_grad():
    # Add batch dimension
        x_batch = x.unsqueeze(0)  # [1, T]
    
    # Stage 1: Gammatone filterbank
        fb_out = model.filterbank(x_batch)  # [1, F, T]
        print(f"  [1] Filterbank output: shape={fb_out.shape}, range=[{fb_out.min():.2e}, {fb_out.max():.2e}]")
    
    # Stage 2: IHC envelope extraction
        ihc_out = model.ihc(fb_out)  # [1, F, T]
        print(f"  [2] IHC envelope: shape={ihc_out.shape}, range=[{ihc_out.min():.2e}, {ihc_out.max():.2e}]")
    
    # Stage 3: Adaptation
        adapt_out = model.adaptation(ihc_out)  # [1, F, T]
        print(f"  [3] Adaptation: shape={adapt_out.shape}, range=[{adapt_out.min():.2e}, {adapt_out.max():.2e}]")
    
    # Stage 4: Modulation filterbank
        mod_out = model.modulation(adapt_out)  # List of [M_i, T] tensors
        print(f"  [4] Modulation filterbank: {len(mod_out)} channels")

    # Create figures
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f'Dau1997 Processing Pipeline - {freq} Hz Tone', fontsize=14, fontweight='bold', y=0.995)

    # 1. Input signal
    axes[0].plot(t[:500], signal[:500], 'b-', linewidth=0.5)
    axes[0].set_title('1. Input Signal (first 500 samples)')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 2. Gammatone filterbank output
    fb_np = fb_out[0].cpu().numpy()  # [num_channels, time]
    extent = [0, duration, 0, model.filterbank.num_channels]
    im2 = axes[1].imshow(fb_np, aspect='auto', origin='lower', extent=extent, cmap='seismic')
    axes[1].set_title('2. Gammatone Filterbank Output')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Channel')
    plt.colorbar(im2, ax=axes[1], label='Amplitude')

    # 3. IHC envelope
    ihc_np = ihc_out[0].cpu().numpy()
    im3 = axes[2].imshow(ihc_np, aspect='auto', origin='lower', extent=extent, cmap='hot')
    axes[2].set_title('3. Inner Hair Cell Envelope')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Channel')
    plt.colorbar(im3, ax=axes[2], label='Amplitude')

    # 4. Adaptation
    adapt_np = adapt_out[0].cpu().numpy()
    im4 = axes[3].imshow(adapt_np, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[3].set_title('4. Adaptation')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Channel')
    plt.colorbar(im4, ax=axes[3], label='Amplitude [MU]')

    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'dau1997_1khz.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    # 5. Modulation filterbank (save each channel to separate file in a folder)
    print(f"\n  [5] Generating modulation filterbank plots for all {len(mod_out)} channels...")

    output_dir = TEST_FIGURES_DIR / 'dau1997_modulation_channels'
    output_dir.mkdir(exist_ok=True)

    for ch_idx, mod_ch_out in enumerate(mod_out):
        fc_val = model.filterbank.fc[ch_idx].item()
        mod_ch = mod_ch_out.squeeze(0).cpu().numpy()  # [num_mod_bands, time]
        extent_mod = [0, duration, 0, mod_ch.shape[0]]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'DAU1997 Modulation Filterbank - Channel {ch_idx}: fc = {fc_val:.1f} Hz', 
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
        
        if ch_idx == 0 or ch_idx == len(mod_out) - 1 or ch_idx % 10 == 0:
            print(f"    Saved: {output_filename}")

    print(f"\nAll {len(mod_out)} modulation filterbank plots saved to: {output_dir}/")
    print(f"\nPlot saved: {output_path}")
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_dau1997_processing_pipeline()
