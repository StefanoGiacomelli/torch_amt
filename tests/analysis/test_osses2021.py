"""
Osses2021 Model - Test Suite

Contents:
1. test_osses2021: Tests complete Osses2021 auditory model pipeline

Structure:
- 2-tone mixture (150 Hz, 1500 Hz)
- Peripheral filtering stages (headphone + middle ear)
- Full processing pipeline visualization
- Per-channel modulation filterbank outputs

Figures generated:
- osses2021_mixture.png: Main pipeline analysis (6 stages)
- osses2021_modulation_channels/: Per-channel modulation outputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch_amt.models.osses2021 import Osses2021


def test_osses2021():
    """Test Osses2021 model with 2-tone mixture."""
    TEST_FIGURES_DIR = Path(__file__).parent.parent.parent / 'test_figures'
    TEST_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Parameters
    fs = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Generate 2-tone mixture (150 Hz, 1500 Hz)
    frequencies = [150, 1500]
    amplitude = 0.5  # Equal amplitude for both tones
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    x = torch.tensor(signal, dtype=torch.float32)
    
    print("="*80)
    print("OSSES2021 MODEL TEST - 2-Tone Mixture")
    print("="*80)
    print(f"\nParameters:")
    print(f"  fs = {fs} Hz")
    print(f"  duration = {duration} s")
    print(f"  tone frequencies = {frequencies} Hz")
    print(f"  amplitude per tone = {amplitude}")
    print(f"  samples = {len(x)}")
    
    # Create model
    model = Osses2021(fs=fs, phase_type='minimum')
    print(f"\nOsses2021 Model:")
    print(f"  Phase type: {model.phase_type}")
    print(f"  Filterbank channels: {model.filterbank.num_channels}")
    print(f"  fc range: {model.filterbank.fc[0]:.2f} - {model.filterbank.fc[-1]:.2f} Hz")
    print(f"  Adaptation preset: osses2021 (limit={model.adaptation.limit})")
    print(f"  Modulation preset: jepsen2008 (lp_cutoff={model.modulation.lp_cutoff} Hz)")
    
    # Processing
    print(f"\nProcessing...")
    with torch.no_grad():
        # Add batch dimension
        x_batch = x.unsqueeze(0)  # [1, T]
        
        # Stage 0a: Headphone filter
        hp_out = model.headphone(x_batch)  # [1, T]
        print(f"  [0a] Headphone filter: shape={hp_out.shape}, range=[{hp_out.min():.2e}, {hp_out.max():.2e}]")
        
        # Stage 0b: Middle ear filter
        me_out = model.middleear(hp_out)  # [1, T]
        print(f"  [0b] Middle ear filter: shape={me_out.shape}, range=[{me_out.min():.2e}, {me_out.max():.2e}]")
        
        # Stage 1: Gammatone filterbank
        fb_out = model.filterbank(me_out)  # [1, F, T]
        print(f"  [1] Filterbank output: shape={fb_out.shape}, range=[{fb_out.min():.2e}, {fb_out.max():.2e}]")
        
        # Stage 2: IHC envelope extraction
        ihc_out = model.ihc(fb_out)  # [1, F, T]
        print(f"  [2] IHC envelope: shape={ihc_out.shape}, range=[{ihc_out.min():.2e}, {ihc_out.max():.2e}]")
        
        # Stage 3: Adaptation
        adapt_out = model.adaptation(ihc_out)  # [1, F, T]
        print(f"  [3] Adaptation: shape={adapt_out.shape}, range=[{adapt_out.min():.2e}, {adapt_out.max():.2e}]")
        
        # Stage 4: Modulation filterbank
        mod_out = model.modulation(adapt_out)  # List of [1, M_i, T] tensors
        print(f"  [4] Modulation filterbank: {len(mod_out)} channels")
    
    # Create figures
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(6, 1, figsize=(14, 14))
    fig.suptitle('Osses2021 Processing Pipeline - 2-Tone Mixture (150, 1500 Hz)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 1. Input signal
    zoom_samples = 1000
    axes[0].plot(t[:zoom_samples], signal[:zoom_samples], 'b-', linewidth=0.5)
    axes[0].set_title('1. Input Signal (first 1000 samples)')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Headphone filter output
    hp_np = hp_out[0].cpu().numpy()
    axes[1].plot(t[:zoom_samples], hp_np[:zoom_samples], 'g-', linewidth=0.5)
    axes[1].set_title('2. Headphone Filter Output (Pralong & Carlile 1996)')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Middle ear filter output
    me_np = me_out[0].cpu().numpy()
    axes[2].plot(t[:zoom_samples], me_np[:zoom_samples], 'orange', linewidth=0.5)
    axes[2].set_title('3. Middle Ear Filter Output (Jepsen 2008)')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Gammatone filterbank output
    fb_np = fb_out[0].cpu().numpy()  # [num_channels, time]
    extent = [0, duration, 0, model.filterbank.num_channels]
    im4 = axes[3].imshow(fb_np, aspect='auto', origin='lower', extent=extent, cmap='seismic')
    axes[3].set_title('4. Gammatone Filterbank Output')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Channel')
    plt.colorbar(im4, ax=axes[3], label='Amplitude')
    
    # 5. IHC envelope
    ihc_np = ihc_out[0].cpu().numpy()
    im5 = axes[4].imshow(ihc_np, aspect='auto', origin='lower', extent=extent, cmap='hot')
    axes[4].set_title('5. Inner Hair Cell Envelope (Breebaart 2001)')
    axes[4].set_xlabel('Time [s]')
    axes[4].set_ylabel('Channel')
    plt.colorbar(im5, ax=axes[4], label='Amplitude')
    
    # 6. Adaptation
    adapt_np = adapt_out[0].cpu().numpy()
    im6 = axes[5].imshow(adapt_np, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[5].set_title('6. Adaptation (limit=5.0, osses2021 preset)')
    axes[5].set_xlabel('Time [s]')
    axes[5].set_ylabel('Channel')
    plt.colorbar(im6, ax=axes[5], label='Amplitude [MU]')
    
    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'osses2021_mixture.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nMain plot saved: {output_path}")
    
    # 7. Modulation filterbank - save each channel to separate file
    print(f"\n  [7] Generating modulation filterbank plots for all {len(mod_out)} channels...")
    
    output_dir = TEST_FIGURES_DIR / 'osses2021_modulation_channels'
    output_dir.mkdir(exist_ok=True)
    
    for ch_idx, mod_ch_out in enumerate(mod_out):
        fc_val = model.filterbank.fc[ch_idx].item()
        mod_ch = mod_ch_out.squeeze(0).cpu().numpy()  # [num_mod_bands, time]
        extent_mod = [0, duration, 0, mod_ch.shape[0]]
        
        fig_mod, ax_mod = plt.subplots(1, 1, figsize=(12, 4))
        fig_mod.suptitle(f'Osses2021 Modulation Filterbank (jepsen2008) - Channel {ch_idx}: fc = {fc_val:.1f} Hz', 
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
    print(f"\nMain plot saved: {output_path}")
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_osses2021()
