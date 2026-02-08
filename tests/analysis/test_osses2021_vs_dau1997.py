"""
Osses2021 vs Dau1997 Comparison - Test Suite

Contents:
1. test_osses2021_vs_dau1997: Compares Osses2021 and Dau1997 models

Structure:
- Side-by-side comparison on 2-tone mixture (150 Hz, 1500 Hz)
- Highlights differences in peripheral filtering, IHC, adaptation, modulation
- Visualizes all processing stages

Figures generated:
- osses2021_vs_dau1997_comparison.png: Complete side-by-side comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_amt.models.osses2021 import Osses2021
from torch_amt.models.dau1997 import Dau1997


def test_osses2021_vs_dau1997():
    """Compare Osses2021 and Dau1997 models on same 2-tone mixture."""
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
    print("OSSES2021 vs Dau1997 COMPARISON - 2-Tone Mixture")
    print("="*80)
    print(f"\nParameters:")
    print(f"  fs = {fs} Hz")
    print(f"  duration = {duration} s")
    print(f"  tone frequencies = {frequencies} Hz")
    print(f"  amplitude per tone = {amplitude}")
    print(f"  samples = {len(x)}")
    
    # Create models
    print(f"\nCreating models...")
    model_osses = Osses2021(fs=fs, phase_type='minimum')
    model_dau = Dau1997(fs=fs)
    
    print(f"\nOsses2021 Configuration:")
    print(f"  Filterbank channels: {model_osses.filterbank.num_channels}")
    print(f"  Adaptation limit: {model_osses.adaptation.limit}")
    print(f"  Modulation preset: jepsen2008 (lp_cutoff={model_osses.modulation.lp_cutoff} Hz)")
    
    print(f"\nDau1997 Configuration:")
    print(f"  Filterbank channels: {model_dau.filterbank.num_channels}")
    print(f"  Adaptation limit: {model_dau.adaptation.limit}")
    print(f"  Modulation lp_cutoff: {model_dau.modulation.lp_cutoff} Hz")
    
    # Processing both models
    print(f"\nProcessing through Osses2021...")
    with torch.no_grad():
        x_batch = x.unsqueeze(0)  # [1, T]
        
        # Osses2021: with peripheral filters
        hp_out_osses = model_osses.headphone(x_batch)
        me_out_osses = model_osses.middleear(hp_out_osses)
        fb_out_osses = model_osses.filterbank(me_out_osses)
        ihc_out_osses = model_osses.ihc(fb_out_osses)
        adapt_out_osses = model_osses.adaptation(ihc_out_osses)
        mod_out_osses = model_osses.modulation(adapt_out_osses)
        
        print(f"  Filterbank: {fb_out_osses.shape}, range=[{fb_out_osses.min():.2e}, {fb_out_osses.max():.2e}]")
        print(f"  IHC: {ihc_out_osses.shape}, range=[{ihc_out_osses.min():.2e}, {ihc_out_osses.max():.2e}]")
        print(f"  Adaptation: {adapt_out_osses.shape}, range=[{adapt_out_osses.min():.2e}, {adapt_out_osses.max():.2e}]")
        print(f"  Modulation: {len(mod_out_osses)} channels")
    
    print(f"\nProcessing through Dau1997...")
    with torch.no_grad():
        # Dau1997: direct input to filterbank
        fb_out_dau = model_dau.filterbank(x_batch)
        ihc_out_dau = model_dau.ihc(fb_out_dau)
        adapt_out_dau = model_dau.adaptation(ihc_out_dau)
        mod_out_dau = model_dau.modulation(adapt_out_dau)
        
        print(f"  Filterbank: {fb_out_dau.shape}, range=[{fb_out_dau.min():.2e}, {fb_out_dau.max():.2e}]")
        print(f"  IHC: {ihc_out_dau.shape}, range=[{ihc_out_dau.min():.2e}, {ihc_out_dau.max():.2e}]")
        print(f"  Adaptation: {adapt_out_dau.shape}, range=[{adapt_out_dau.min():.2e}, {adapt_out_dau.max():.2e}]")
        print(f"  Modulation: {len(mod_out_dau)} channels")
    
    # Compute quantitative differences
    print(f"\n" + "="*80)
    print("QUANTITATIVE COMPARISON")
    print("="*80)
    
    # Find channel closest to 1 kHz for modulation comparison
    target_fc = 1000
    ch_idx_osses = (model_osses.fc - target_fc).abs().argmin().item()
    ch_idx_dau = (model_dau.fc - target_fc).abs().argmin().item()
    fc_osses = model_osses.fc[ch_idx_osses].item()
    fc_dau = model_dau.fc[ch_idx_dau].item()
    
    print(f"\nSelected channel for modulation comparison:")
    print(f"  Osses2021: ch={ch_idx_osses}, fc={fc_osses:.1f} Hz")
    print(f"  Dau1997: ch={ch_idx_dau}, fc={fc_dau:.1f} Hz")
    
    # Stage-by-stage metrics
    stages = ['Filterbank', 'IHC', 'Adaptation']
    osses_stages = [fb_out_osses, ihc_out_osses, adapt_out_osses]
    dau_stages = [fb_out_dau, ihc_out_dau, adapt_out_dau]
    
    print(f"\nNote: Osses2021 output is shorter due to peripheral filter delay compensation")
    print(f"  Osses2021 length: {osses_stages[0].shape[-1]} samples")
    print(f"  Dau1997 length: {dau_stages[0].shape[-1]} samples")
    print(f"  Comparing first {osses_stages[0].shape[-1]} samples")
    
    for stage_name, osses_data, dau_data in zip(stages, osses_stages, dau_stages):
        osses_np = osses_data[0].cpu().numpy()
        dau_np = dau_data[0].cpu().numpy()
        
        # Align signals: truncate DAU to match Osses length
        min_len = min(osses_np.shape[-1], dau_np.shape[-1])
        osses_np = osses_np[..., :min_len]
        dau_np = dau_np[..., :min_len]
        
        # RMS difference
        rms_diff = np.sqrt(np.mean((osses_np - dau_np)**2))
        
        # Peak amplitude ratio
        peak_osses = np.abs(osses_np).max()
        peak_dau = np.abs(dau_np).max()
        peak_ratio = peak_osses / peak_dau if peak_dau > 0 else 0
        
        # Correlation
        osses_flat = osses_np.flatten()
        dau_flat = dau_np.flatten()
        correlation = np.corrcoef(osses_flat, dau_flat)[0, 1]
        
        print(f"\n{stage_name}:")
        print(f"  RMS difference: {rms_diff:.6f}")
        print(f"  Peak ratio (Osses/DAU): {peak_ratio:.4f}")
        print(f"  Correlation: {correlation:.6f}")
    
    # Modulation filterbank comparison (1 kHz channel)
    mod_osses_1k = mod_out_osses[ch_idx_osses].squeeze(0).cpu().numpy()  # [M_osses, T]
    mod_dau_1k = mod_out_dau[ch_idx_dau].squeeze(0).cpu().numpy()  # [M_dau, T]
    
    print(f"\nModulation Filterbank (1 kHz channel):")
    print(f"  Osses2021: {mod_osses_1k.shape[0]} modulation bands")
    print(f"  Dau1997: {mod_dau_1k.shape[0]} modulation bands")
    print(f"  Peak Osses: {np.abs(mod_osses_1k).max():.6f}")
    print(f"  Peak DAU: {np.abs(mod_dau_1k).max():.6f}")
    
    # Create comparison plot
    print(f"\nGenerating comparison plots...")
    fig, axes = plt.subplots(5, 2, figsize=(16, 15))
    fig.suptitle('Osses2021 vs Dau1997 Comparison - 2-Tone Mixture (150, 1500 Hz)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Align time for visualization
    osses_len = fb_out_osses.shape[-1]
    dau_len = fb_out_dau.shape[-1]
    min_len = min(osses_len, dau_len)
    duration_plot = min_len / fs
    
    extent = [0, duration_plot, 0, model_osses.filterbank.num_channels]
    extent_mod_osses = [0, duration_plot, 0, mod_osses_1k.shape[0]]
    extent_mod_dau = [0, duration_plot, 0, mod_dau_1k.shape[0]]
    
    # Column titles
    axes[0, 0].text(0.5, 1.08, 'Osses2021 (with peripheral filters)', 
                    ha='center', va='bottom', transform=axes[0, 0].transAxes, 
                    fontsize=12, fontweight='bold')
    axes[0, 1].text(0.5, 1.08, 'Dau1997 (direct input)', 
                    ha='center', va='bottom', transform=axes[0, 1].transAxes, 
                    fontsize=12, fontweight='bold')
    
    # Row 1: Gammatone filterbank
    fb_osses_np = fb_out_osses[0, :, :min_len].cpu().numpy()
    fb_dau_np = fb_out_dau[0, :, :min_len].cpu().numpy()
    
    im1_osses = axes[0, 0].imshow(fb_osses_np, aspect='auto', origin='lower', 
                                   extent=extent, cmap='seismic', vmin=-0.3, vmax=0.3)
    axes[0, 0].set_ylabel('Channel')
    axes[0, 0].set_title('Gammatone Filterbank\n(post headphone + middleear)')
    plt.colorbar(im1_osses, ax=axes[0, 0], label='Amplitude')
    
    im1_dau = axes[0, 1].imshow(fb_dau_np, aspect='auto', origin='lower', 
                                 extent=extent, cmap='seismic', vmin=-0.3, vmax=0.3)
    axes[0, 1].set_ylabel('Channel')
    axes[0, 1].set_title('Gammatone Filterbank\n(direct from input)')
    plt.colorbar(im1_dau, ax=axes[0, 1], label='Amplitude')

    # Row 2: IHC envelope
    ihc_osses_np = ihc_out_osses[0, :, :min_len].cpu().numpy()
    ihc_dau_np = ihc_out_dau[0, :, :min_len].cpu().numpy()

    im2_osses = axes[1, 0].imshow(ihc_osses_np, aspect='auto', origin='lower', 
                                   extent=extent, cmap='hot')
    axes[1, 0].set_ylabel('Channel')
    axes[1, 0].set_title('IHC Envelope\n(Breebaart 2001)')
    plt.colorbar(im2_osses, ax=axes[1, 0], label='Amplitude')

    im2_dau = axes[1, 1].imshow(ihc_dau_np, aspect='auto', origin='lower', 
                                 extent=extent, cmap='hot')
    axes[1, 1].set_ylabel('Channel')
    axes[1, 1].set_title('IHC Envelope\n(DAU 1996)')
    plt.colorbar(im2_dau, ax=axes[1, 1], label='Amplitude')

    # Row 3: Adaptation
    adapt_osses_np = adapt_out_osses[0, :, :min_len].cpu().numpy()
    adapt_dau_np = adapt_out_dau[0, :, :min_len].cpu().numpy()

    im3_osses = axes[2, 0].imshow(adapt_osses_np, aspect='auto', origin='lower', 
                                   extent=extent, cmap='viridis')
    axes[2, 0].set_ylabel('Channel')
    axes[2, 0].set_title('Adaptation\n(limit=5.0, osses2021)')
    plt.colorbar(im3_osses, ax=axes[2, 0], label='Amplitude [MU]')

    im3_dau = axes[2, 1].imshow(adapt_dau_np, aspect='auto', origin='lower', 
                                 extent=extent, cmap='viridis')
    axes[2, 1].set_ylabel('Channel')
    axes[2, 1].set_title('Adaptation\n(limit=10.0, dau1997)')
    plt.colorbar(im3_dau, ax=axes[2, 1], label='Amplitude [MU]')

    # Row 4: Modulation filterbank (1 kHz channel)
    mod_osses_1k_plot = mod_osses_1k[:, :min_len]
    mod_dau_1k_plot = mod_dau_1k[:, :min_len]

    im4_osses = axes[3, 0].imshow(mod_osses_1k_plot, aspect='auto', origin='lower', 
                                   extent=extent_mod_osses, cmap='plasma')
    axes[3, 0].set_xlabel('Time [s]')
    axes[3, 0].set_ylabel('Modulation Band')
    axes[3, 0].set_title(f'Modulation Filterbank\n(jepsen2008, LP 150Hz, fc={fc_osses:.0f}Hz)')
    plt.colorbar(im4_osses, ax=axes[3, 0], label='Amplitude [MU]')

    im4_dau = axes[3, 1].imshow(mod_dau_1k_plot, aspect='auto', origin='lower', 
                                 extent=extent_mod_dau, cmap='plasma')
    axes[3, 1].set_xlabel('Time [s]')
    axes[3, 1].set_ylabel('Modulation Band')
    axes[3, 1].set_title(f'Modulation Filterbank\n(dau1997, LP 2.5Hz, fc={fc_dau:.0f}Hz)')
    plt.colorbar(im4_dau, ax=axes[3, 1], label='Amplitude [MU]')

    # Row 5: Modulation filterbank frequency response comparison
    print(f"  Computing modulation filterbank frequency responses...")

    # Generate impulse and process through both modulation filterbanks
    impulse = torch.zeros(1, model_osses.num_channels, 8000)
    impulse[:, ch_idx_osses, 4000] = 1.0  # Impulse at center

    with torch.no_grad():
        # Osses modulation response
        mod_response_osses = model_osses.modulation(impulse)
        mod_ir_osses = mod_response_osses[ch_idx_osses].squeeze(0).cpu().numpy()  # [M_osses, T]
    
        # DAU modulation response
        mod_response_dau = model_dau.modulation(impulse)
        mod_ir_dau = mod_response_dau[ch_idx_dau].squeeze(0).cpu().numpy()  # [M_dau, T]

    # Compute frequency responses for ALL modulation bands
    freqs_osses = np.fft.rfftfreq(mod_ir_osses.shape[1], 1/fs)
    freqs_dau = np.fft.rfftfreq(mod_ir_dau.shape[1], 1/fs)

    # Color map for bandpass filters (skip lowpass at index 0)
    colors_osses = plt.cm.viridis(np.linspace(0, 1, mod_ir_osses.shape[0] - 1))
    colors_dau = plt.cm.viridis(np.linspace(0, 1, mod_ir_dau.shape[0] - 1))

    # Plot Osses2021 - Lowpass first (dashed)
    fft_lp_osses = np.fft.rfft(mod_ir_osses[0, :])
    mag_db_lp = 20*np.log10(np.abs(fft_lp_osses) + 1e-10)
    axes[4, 0].plot(freqs_osses, mag_db_lp, 'k--', linewidth=2.5, 
                    alpha=0.8, label=f'LP (cutoff={model_osses.modulation.lp_cutoff} Hz)')

    # Plot Osses2021 - Bandpass filters
    for i in range(1, mod_ir_osses.shape[0]):
        fft_osses = np.fft.rfft(mod_ir_osses[i, :])
        mag_db = 20*np.log10(np.abs(fft_osses) + 1e-10)
        axes[4, 0].plot(freqs_osses, mag_db, color=colors_osses[i-1], linewidth=1.5, 
                        alpha=0.7, label=f'BP {i}')

    axes[4, 0].set_xlabel('Frequency [Hz]')
    axes[4, 0].set_ylabel('Magnitude [dB]')
    axes[4, 0].set_title(f'Modulation Filterbank Frequency Responses (all {mod_ir_osses.shape[0]} bands)\n(jepsen2008: LP {model_osses.modulation.lp_cutoff} Hz, att_factor={model_osses.modulation.att_factor:.3f})')
    axes[4, 0].set_xscale('log')
    axes[4, 0].set_xlim([1, 200])
    axes[4, 0].set_ylim([-80, 10])
    axes[4, 0].grid(True, alpha=0.3, which='both')
    axes[4, 0].legend(loc='upper right', fontsize=7, ncol=2)

    # Plot Dau1997 - Lowpass first (dashed)
    fft_lp_dau = np.fft.rfft(mod_ir_dau[0, :])
    mag_db_lp = 20*np.log10(np.abs(fft_lp_dau) + 1e-10)
    axes[4, 1].plot(freqs_dau, mag_db_lp, 'k--', linewidth=2.5, 
                    alpha=0.8, label=f'LP (cutoff={model_dau.modulation.lp_cutoff} Hz)')

    # Plot Dau1997 - Bandpass filters
    for i in range(1, mod_ir_dau.shape[0]):
        fft_dau = np.fft.rfft(mod_ir_dau[i, :])
        mag_db = 20*np.log10(np.abs(fft_dau) + 1e-10)
        axes[4, 1].plot(freqs_dau, mag_db, color=colors_dau[i-1], linewidth=1.5, 
                        alpha=0.7, label=f'BP {i}')

    axes[4, 1].set_xlabel('Frequency [Hz]')
    axes[4, 1].set_ylabel('Magnitude [dB]')
    axes[4, 1].set_title(f'Modulation Filterbank Frequency Responses (all {mod_ir_dau.shape[0]} bands)\n(dau1997: LP {model_dau.modulation.lp_cutoff} Hz, att_factor={model_dau.modulation.att_factor:.3f})')
    axes[4, 1].set_xscale('log')
    axes[4, 1].set_xlim([1, 200])
    axes[4, 1].set_ylim([-80, 10])
    axes[4, 1].grid(True, alpha=0.3, which='both')
    axes[4, 1].legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    output_path = TEST_FIGURES_DIR / 'osses2021_vs_dau1997_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nComparison plot saved: {output_path}")
    
    print("\n" + "="*80)
    print("COMPARISON TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    test_osses2021_vs_dau1997()
