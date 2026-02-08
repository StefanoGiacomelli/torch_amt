"""
Moore2016 Binaural Loudness Model
==================================

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the complete binaural loudness model from Moore et al. (2016),
providing a full 1:1 port of the MATLAB implementation from the AMT. The model computes
short-term and long-term binaural loudness from stereo audio input, incorporating
outer/middle ear filtering, multi-resolution spectral analysis, auditory excitation
patterns, specific loudness computation, temporal integration, and binaural inhibition.

The implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT) 
and extended with PyTorch for gradient-based optimization and GPU acceleration.

References
----------
.. [1] B. C. J. Moore, B. R. Glasberg, and J. Schlittenlacher, "A model of binaural 
       loudness perception based on the banded loudness model," *Acta Acust. united 
       with Acust.*, vol. 102, no. 5, pp. 824-837, Sep. 2016.

.. [2] B. C. J. Moore and B. R. Glasberg, "Modeling binaural loudness," 
       *J. Acoust. Soc. Am.*, vol. 121, no. 3, pp. 1604-1612, Mar. 2007.

.. [3] ANSI S3.4-2007, "Procedure for the Computation of Loudness of Steady Sounds," 
       American National Standards Institute, 2007.

.. [4] B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter shapes 
       from notched-noise data," *Hear. Res.*, vol. 47, no. 1-2, pp. 103-138, 
       Aug. 1990.

.. [5] ISO 226:2003, "Acoustics - Normal equal-loudness-level contours," 
       International Organization for Standardization, 2003.

.. [6] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
       p. 19, 2022.
"""

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

from ..common.ears import OuterMiddleEarFilter
from ..common.filterbanks import Moore2016Spectrum
from ..common.filterbanks import Moore2016ExcitationPattern
from ..common.loudness import (Moore2016SpecificLoudness,
                               Moore2016TemporalIntegration,
                               Moore2016AGC,
                               Moore2016BinauralLoudness)


class Moore2016(nn.Module):
    r"""
    Moore et al. (2016) model for binaural loudness perception.
    
    Implements the complete binaural loudness processing pipeline from Moore, Glasberg, 
    and Schlittenlacher (2016), providing perceptual loudness measures in sone from 
    stereo audio waveforms. The model extends monaural loudness computation with binaural 
    inhibition effects, accounting for both diotic summation and dichotic interactions 
    between ears.
    
    This implementation is a 1:1 port of the MATLAB Auditory Modeling Toolbox (AMT) 
    ``moore2016`` function and provides a differentiable, GPU-accelerated version 
    suitable for neural network training and binaural loudness optimization.
    
    Algorithm Overview
    ------------------
    The model implements an 8-stage binaural loudness processing pipeline:
    
    **Stage 1: Outer/Middle Ear Filtering (per ear)**
    
    Simulates transmission through outer and middle ear using ANSI S3.4-2007 
    transfer function:
    
    .. math::
        x_L'(t) = h_{\\text{ear}}(t) * x_L(t), \\quad x_R'(t) = h_{\\text{ear}}(t) * x_R(t)
    
    where :math:`h_{\\text{ear}}` is the tfOuterMiddle2007 filter (1025 coefficients).
    
    **Stage 2: Multi-Resolution Spectral Analysis (per ear)**
    
    Uses 6 time windows (2.048ms to 65.536ms) with frequency-dependent selection:
    
    .. math::
        S(t, f) = \\text{FFT}_{N(f)}(w(t) \\cdot x'(t))
    
    where :math:`N(f)` varies from 64 to 2048 samples based on frequency.
    Output: sparse spectrum with ERB-spaced frequencies.
    
    **Stage 3: Excitation Pattern (per ear)**
    
    Models basilar membrane response with rounded-exponential (roex) filters 
    on 150 ERB-spaced channels (80 Hz - 15 kHz):
    
    .. math::
        E(t, f_{ERB}) = \\sum_k S(t, f_k) \\cdot W_{roex}(f_k, f_{ERB})
    
    Output in dB SPL.
    
    **Stage 4: Specific Loudness (per ear)**
    
    Applies instantaneous nonlinear compression via 88-value lookup table:
    
    .. math::
        N_{inst}(t, f) = \\text{LUT}(E(t, f))
    
    Output: instantaneous specific loudness in sone/ERB.
    
    **Stage 5: Short-Term Temporal Integration (per ear)**
    
    AGC smoothing with asymmetric attack/release dynamics:
    
    .. math::
        N_{STL}(t, f) = \\alpha N_{inst}(t, f) + (1-\\alpha) N_{STL}(t-1, f)
    
    where :math:`\\alpha = 0.045` (attack), :math:`\\alpha = 0.033` (release).
    
    **Stage 6: Binaural Inhibition (frame-by-frame)**
    
    Spatial smoothing followed by cross-ear inhibition:
    
    .. math::
        N_{smooth}(f) = \\sum_g N_{STL}(f+g) \\cdot G(g, \\sigma=0.08)
    
    .. math::
        N_{inhib,L} = N_{smooth,L} \\cdot \\text{sech}(p \\cdot N_{smooth,R}), \\quad p=1.5978
    
    Integration yields scalar loudness per ear: :math:`L_L = \\sum_f N_{inhib,L}(f) / 4`.
    
    **Stage 7: Long-Term Temporal Integration (per ear)**
    
    AGC on scalar loudness with slower dynamics:
    
    .. math::
        L_{LTL}(t) = \\alpha L_{STL}(t) + (1-\\alpha) L_{LTL}(t-1)
    
    where :math:`\\alpha = 0.01` (attack), :math:`\\alpha = 0.00133` (release).
    
    **Stage 8: Binaural Output**
    
    .. math::
        L_{STL} = L_{STL,L} + L_{STL,R}, \\quad L_{LTL} = L_{LTL,L} + L_{LTL,R}
    
    Output: short-term loudness (STL), long-term loudness (LTL), max loudness.
    
    Parameters
    ----------
    fs : int, optional
        Sampling rate in Hz. **Must be 32000 Hz**. Default: 32000.
        This requirement is enforced by the outer/middle ear filter design (ANSI S3.4-2007).
        If you have audio at different sampling rates, resample to 32 kHz before processing.
    
    learnable : bool, optional
        If True, all model stages become trainable with gradient-based optimization. 
        Default: False (fixed parameters).
        When True, enables end-to-end model training for task-specific optimization.
    
    return_stages : bool, optional
        If True, returns intermediate processing stages along with final output. 
        Default: False (only final STL, LTL, mLoud).
        Useful for visualization, analysis, and multi-stage training.
    
    dtype : torch.dtype, optional
        Data type for computations and parameters. Default: torch.float32.
        Use torch.float64 for higher precision if needed (increases memory/computation).
    
    **outer_middle_ear_kwargs : dict, optional
        Additional keyword arguments passed to :class:`OuterMiddleEarFilter`.
        Common options:
        
        - ``compensation_type`` (str): Filter type. Default: 'tfOuterMiddle2007'.
        - ``field_type`` (str): Sound field type. Default: 'free'.
        - Other parameters accepted by OuterMiddleEarFilter.
    
    **spectrum_kwargs : dict, optional
        Additional keyword arguments passed to :class:`Moore2016Spectrum`.
        Common options:
        
        - ``hop_length`` (int): Hop size in samples. Default: 512.
        - ``n_windows`` (int): Number of window sizes. Default: 6.
        - Other parameters accepted by Moore2016Spectrum.
    
    **excitation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`Moore2016ExcitationPattern`.
        Common options:
        
        - ``erb_lower`` (float): Lower ERB limit. Default: 1.75.
        - ``erb_upper`` (float): Upper ERB limit. Default: 39.0.
        - ``erb_step`` (float): ERB step size. Default: 0.25.
        - Other parameters accepted by Moore2016ExcitationPattern.
    
    **specific_loudness_kwargs : dict, optional
        Additional keyword arguments passed to :class:`Moore2016SpecificLoudness`.
        Parameters for the 88-value loudness lookup table.
    
    **temporal_integration_kwargs : dict, optional
        Additional keyword arguments passed to :class:`Moore2016TemporalIntegration`.
        Common options:
        
        - ``attack_alpha`` (float): STL attack coefficient. Default: 0.045.
        - ``release_alpha`` (float): STL release coefficient. Default: 0.033.
        - Other parameters accepted by Moore2016TemporalIntegration.
    
    **binaural_loudness_kwargs : dict, optional
        Additional keyword arguments passed to :class:`Moore2016BinauralLoudness`.
        Common options:
        
        - ``sigma`` (float): Spatial smoothing width. Default: 0.08.
        - ``p`` (float): Inhibition strength parameter. Default: 1.5978.
        - Other parameters accepted by Moore2016BinauralLoudness.
    
    **ltl_agc_kwargs : dict, optional
        Additional keyword arguments passed to :class:`Moore2016AGC` (for LTL).
        Common options:
        
        - ``attack_alpha`` (float): LTL attack coefficient. Default: 0.01.
        - ``release_alpha`` (float): LTL release coefficient. Default: 0.00133.
        - Applied to both left and right LTL AGCs.
        - Applied to both left and right LTL AGCs.
        
    Attributes
    ----------
    fs : int
        Sampling rate in Hz (fixed at 32000).
    
    learnable : bool
        Whether model parameters are trainable.
    
    return_stages : bool
        Whether to return intermediate processing stages.
    
    dtype : torch.dtype
        Data type for computations.
    
    outer_middle_ear : OuterMiddleEarFilter
        Stage 1: Outer and middle ear transfer function filter.
    
    spectrum : Moore2016Spectrum
        Stage 2: Multi-resolution spectral analysis module.
    
    excitation : Moore2016ExcitationPattern
        Stage 3: Auditory excitation pattern computation module.
    
    specific_loudness : Moore2016SpecificLoudness
        Stage 4: Instantaneous specific loudness module.
    
    temporal_integration : Moore2016TemporalIntegration
        Stage 5: Short-term temporal AGC module.
    
    binaural_loudness : Moore2016BinauralLoudness
        Stage 6: Binaural inhibition and spatial smoothing module.
    
    ltl_agc_left : Moore2016AGC
        Stage 7: Long-term AGC for left ear.
    
    ltl_agc_right : Moore2016AGC
        Stage 7: Long-term AGC for right ear.
    
    Input Shape
    -----------
    audio : torch.Tensor
        Stereo audio signal with shape:
        
        - :math:`(B, 2, T)` - Batch of stereo audio samples
        - :math:`(2, T)` - Single stereo sample
        
        where:
        
        - :math:`B` = batch size
        - :math:`2` = channels (left, right)
        - :math:`T` = time samples at 32 kHz
        
        **Important:** Audio must be in Pascal (Pa) units. 
        For conversion: :math:`\\text{Pa} = 2 \\times 10^{-5} \\times 10^{\\text{dB SPL}/20}`.
    
    Output Shape
    ------------
    When ``return_stages=False`` (default):
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Three tensors:
            
            - ``sLoud``: Short-term binaural loudness, shape :math:`(B, F)` in sone
            - ``lLoud``: Long-term binaural loudness, shape :math:`(B, F)` in sone
            - ``mLoud``: Maximum long-term loudness, shape :math:`(B,)` in sone
            
            where :math:`F` = number of time frames (depends on hop_length).
            
    When ``return_stages=True``:
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
            - First element: (sLoud, lLoud, mLoud) as above
            - Second element: dict with keys:
              
              - ``'filtered_left'``: After outer/middle ear, shape :math:`(B, T')`
              - ``'filtered_right'``: shape :math:`(B, T')`
              - ``'freqs_left'``: Sparse spectrum frequencies, shape :math:`(B, F, N_{comp})`
              - ``'levels_left'``: Sparse spectrum levels in dB, shape :math:`(B, F, N_{comp})`
              - ``'freqs_right'``, ``'levels_right'``: same for right ear
              - ``'excitation_left'``: Excitation pattern, shape :math:`(B, F, 150)` in dB SPL
              - ``'excitation_right'``: shape :math:`(B, F, 150)`
              - ``'inst_spec_loud_left'``: Instantaneous specific loudness, shape :math:`(B, F, 150)` sone/ERB
              - ``'inst_spec_loud_right'``: shape :math:`(B, F, 150)`
              - ``'stl_spec_loud_left'``: Short-term specific loudness, shape :math:`(B, F, 150)`
              - ``'stl_spec_loud_right'``: shape :math:`(B, F, 150)`
              - ``'stl_loud_left'``: Short-term scalar loudness left, shape :math:`(B, F)` sone
              - ``'stl_loud_right'``: shape :math:`(B, F)`
              - ``'ltl_loud_left'``: Long-term scalar loudness left, shape :math:`(B, F)`
              - ``'ltl_loud_right'``: shape :math:`(B, F)`
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> import numpy as np
    >>> from torch_amt.models import Moore2016
    >>> 
    >>> # Create model
    >>> model = Moore2016(fs=32000)
    >>> 
    >>> # Generate stereo tone at 1 kHz, 60 dB SPL
    >>> t = np.linspace(0, 1, 32000)
    >>> tone = np.sin(2 * np.pi * 1000 * t)
    >>> tone_pa = tone * 0.02  # 60 dB SPL
    >>> audio_stereo = torch.from_numpy(np.stack([tone_pa, tone_pa])).float().unsqueeze(0)
    >>> 
    >>> # Process
    >>> sLoud, lLoud, mLoud = model(audio_stereo)
    >>> print(f"STL: {sLoud.mean():.2f} sone")
    STL: 6.17 sone
    
    **With stages:**
    
    >>> model_debug = Moore2016(fs=32000, return_stages=True)
    >>> (sLoud, lLoud, mLoud), stages = model_debug(audio_stereo)
    >>> print(f"Excitation: {stages['excitation_left'].shape}")
    Excitation: torch.Size([1, 62, 150])
    
    Notes
    -----
    **Model Configuration:**
    
    - **Outer/Middle Ear**: ANSI S3.4-2007 tfOuterMiddle2007 transfer function
    - **Spectrum**: 6 windows (64-2048 samples), hop=512
    - **Excitation**: 150 ERB channels (1.75-39.0 ERB, 80 Hz - 15 kHz)
    - **STL AGC**: attack=0.045, release=0.033
    - **Binaural**: Gaussian σ=0.08, inhibition p=1.5978
    - **LTL AGC**: attack=0.01, release=0.00133
    
    **Customizing Submodules:**
    
    All submodules accept kwargs through dedicated dictionaries. The ``learnable`` 
    and ``dtype`` parameters are centralized. See Parameters section for details.
    
    **Sampling Rate:**
    
    **Must be 32000 Hz** due to outer/middle ear filter design. Resample if needed:
    
    .. code-block:: python
    
        import torchaudio
        audio = torchaudio.functional.resample(audio, fs_orig, 32000)
    
    **Binaural Effects:**
    
    1. **Summation**: Diotic stimuli ~2x louder than monaural
    2. **Inhibition**: Dichotic stimuli show cross-ear suppression
    
    **Computational Complexity:**
    
    Frame-by-frame binaural processing requires nested loops. 
    For 1s stereo @ 32kHz: ~0.5-2s CPU, ~0.05-0.2s GPU.
    
    See Also
    --------
    OuterMiddleEarFilter : Stage 1 - Outer/middle ear
    Moore2016Spectrum : Stage 2 - Spectral analysis
    Moore2016ExcitationPattern : Stage 3 - Excitation
    Moore2016SpecificLoudness : Stage 4 - Specific loudness
    Moore2016TemporalIntegration : Stage 5 - STL AGC
    Moore2016BinauralLoudness : Stage 6 - Binaural inhibition
    Moore2016AGC : Stage 7 - LTL AGC
    
    References
    ----------
    .. [1] B. C. J. Moore, B. R. Glasberg, and J. Schlittenlacher, "A model of binaural 
           loudness perception based on the banded loudness model," *Acta Acust. united 
           with Acust.*, vol. 102, no. 5, pp. 824-837, Sep. 2016.
    
    .. [2] B. C. J. Moore and B. R. Glasberg, "Modeling binaural loudness," 
           *J. Acoust. Soc. Am.*, vol. 121, no. 3, pp. 1604-1612, Mar. 2007.
    
    .. [3] ANSI S3.4-2007, "Procedure for the Computation of Loudness of Steady Sounds," 
           American National Standards Institute, 2007.
    
    .. [4] B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter shapes 
           from notched-noise data," *Hear. Res.*, vol. 47, no. 1-2, pp. 103-138, 
           Aug. 1990.
    
    .. [5] ISO 226:2003, "Acoustics - Normal equal-loudness-level contours," 
           International Organization for Standardization, 2003.
    
    .. [6] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
           reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
           p. 19, 2022.
    """
    
    def __init__(self,
                 fs: int = 32000,
                 learnable: bool = False,
                 return_stages: bool = False,
                 dtype: torch.dtype = torch.float32,
                 outer_middle_ear_kwargs: Dict[str, Any] = None,
                 spectrum_kwargs: Dict[str, Any] = None,
                 excitation_kwargs: Dict[str, Any] = None,
                 specific_loudness_kwargs: Dict[str, Any] = None,
                 temporal_integration_kwargs: Dict[str, Any] = None,
                 binaural_loudness_kwargs: Dict[str, Any] = None,
                 ltl_agc_kwargs: Dict[str, Any] = None):
        super().__init__()
        
        if fs != 32000:
            raise ValueError(f"Moore2016 model requires fs=32000 Hz (outer/middle ear filter design). "
                             f"Got fs={fs} Hz. Resample your audio to 32 kHz before processing.")
        
        self.fs = fs
        self.learnable = learnable
        self.return_stages = return_stages
        self.dtype = dtype
        
        # Initialize kwargs dictionaries
        outer_middle_ear_kwargs = outer_middle_ear_kwargs or {}
        spectrum_kwargs = spectrum_kwargs or {}
        excitation_kwargs = excitation_kwargs or {}
        specific_loudness_kwargs = specific_loudness_kwargs or {}
        temporal_integration_kwargs = temporal_integration_kwargs or {}
        binaural_loudness_kwargs = binaural_loudness_kwargs or {}
        ltl_agc_kwargs = ltl_agc_kwargs or {}
        
        # Stage 1: Outer/Middle Ear Filtering
        ear_defaults = {'compensation_type': 'tfOuterMiddle2007', 'field_type': 'free'}
        ear_params = {**ear_defaults, **outer_middle_ear_kwargs}
        self.outer_middle_ear = OuterMiddleEarFilter(fs=fs, learnable=learnable, dtype=dtype, **ear_params)
        
        # Stage 2: Multi-Resolution Spectrum
        self.spectrum = Moore2016Spectrum(fs=fs, learnable=learnable, dtype=dtype, **spectrum_kwargs)
        
        # Stage 3: Excitation Pattern
        exc_defaults = {'erb_lower': 1.75, 'erb_upper': 39.0, 'erb_step': 0.25}
        exc_params = {**exc_defaults, **excitation_kwargs}
        self.excitation = Moore2016ExcitationPattern(learnable=learnable, dtype=dtype, **exc_params)
        
        # Stage 4: Specific Loudness
        self.specific_loudness = Moore2016SpecificLoudness(learnable=learnable, dtype=dtype, **specific_loudness_kwargs)
        
        # Stage 5: Short-Term Temporal Integration
        self.temporal_integration = Moore2016TemporalIntegration(learnable=learnable, dtype=dtype, **temporal_integration_kwargs)
        
        # Stage 6: Binaural Inhibition
        self.binaural_loudness = Moore2016BinauralLoudness(learnable=learnable, dtype=dtype, **binaural_loudness_kwargs)
        
        # Stage 7: Long-Term AGC (separate for left and right)
        ltl_defaults = {'attack_alpha': 0.01, 'release_alpha': 0.00133}
        ltl_params = {**ltl_defaults, **ltl_agc_kwargs}
        self.ltl_agc_left = Moore2016AGC(learnable=learnable, dtype=dtype, **ltl_params)
        
        self.ltl_agc_right = Moore2016AGC(learnable=learnable, dtype=dtype, **ltl_params)
    
    def forward(self,
                audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Process stereo audio through complete Moore2016 binaural loudness model.
        
        Parameters
        ----------
        audio : torch.Tensor
            Stereo audio input. Shape: (batch, 2, n_samples) or (2, n_samples).
            Audio must be in Pascal (Pa) units and sampled at 32 kHz.
            
        Returns
        -------
        tuple or tuple of tuple and dict
            If return_stages=False:
                (sLoud, lLoud, mLoud) where:
                - sLoud: Short-term binaural loudness, shape (batch, n_frames) in sone
                - lLoud: Long-term binaural loudness, shape (batch, n_frames) in sone
                - mLoud: Maximum long-term loudness, shape (batch,) in sone
            
            If return_stages=True:
                ((sLoud, lLoud, mLoud), stages) where stages is a dict with:
                - 'filtered_left', 'filtered_right': After outer/middle ear
                - 'freqs_left', 'levels_left', 'freqs_right', 'levels_right': Sparse spectrum
                - 'excitation_left', 'excitation_right': Excitation patterns
                - 'inst_spec_loud_left', 'inst_spec_loud_right': Instantaneous specific loudness
                - 'stl_spec_loud_left', 'stl_spec_loud_right': Short-term specific loudness
                - 'stl_loud_left', 'stl_loud_right': Short-term scalar loudness
                - 'ltl_loud_left', 'ltl_loud_right': Long-term scalar loudness
        """
        if audio.ndim != 3 or audio.shape[1] != 2:
            raise ValueError(f"Expected stereo audio with shape (batch, 2, n_samples). "
                             f"Got shape: {audio.shape}")
        
        batch_size = audio.shape[0]
        
        stages = {} if self.return_stages else None
        
        # Stage 1: Outer/Middle Ear Filtering
        audio_left = audio[:, 0, :]   # (batch, n_samples)
        audio_right = audio[:, 1, :]  # (batch, n_samples)
        
        filtered_left = self.outer_middle_ear(audio_left)    # (batch, n_samples_filtered)
        filtered_right = self.outer_middle_ear(audio_right)  # (batch, n_samples_filtered)
        
        # Reconstruct stereo for spectrum
        filtered_stereo = torch.stack([filtered_left, filtered_right], dim=1)  # (batch, 2, n_samples_filtered)
        
        if self.return_stages:
            stages['filtered_left'] = filtered_left
            stages['filtered_right'] = filtered_right
        
        # Stage 2-4: Monaural Instantaneous Specific Loudness
        # Stage 2: Spectrum (processes stereo, returns left and right separately)
        freqs_left, levels_left, freqs_right, levels_right = self.spectrum(filtered_stereo)
        # (batch, n_segments, max_components) for each output
        
        if self.return_stages:
            stages['freqs_left'] = freqs_left
            stages['levels_left'] = levels_left
            stages['freqs_right'] = freqs_right
            stages['levels_right'] = levels_right
        
        # Stage 3: Excitation Pattern (processes sparse spectrum)
        # Excitation pattern operates on (batch, n_components), so process each frame
        n_frames = freqs_left.shape[1]
        excitation_left_list = []
        excitation_right_list = []
        
        for t in range(n_frames):
            exc_left_t = self.excitation(freqs_left[:, t, :], levels_left[:, t, :])  # (batch, 150)
            exc_right_t = self.excitation(freqs_right[:, t, :], levels_right[:, t, :])  # (batch, 150)
            excitation_left_list.append(exc_left_t)
            excitation_right_list.append(exc_right_t)
        
        excitation_left = torch.stack(excitation_left_list, dim=1)   # (batch, n_frames, 150)
        excitation_right = torch.stack(excitation_right_list, dim=1)
        
        if self.return_stages:
            stages['excitation_left'] = excitation_left
            stages['excitation_right'] = excitation_right
        
        # Stage 4: Instantaneous Specific Loudness
        inst_spec_loud_left = self.specific_loudness(excitation_left)   # (batch, n_frames, 150)
        inst_spec_loud_right = self.specific_loudness(excitation_right)
        
        # Remove NaNs (following MATLAB: instSpecLoudL(find(isnan(instSpecLoudL))) = 0)
        inst_spec_loud_left = torch.nan_to_num(inst_spec_loud_left, nan=0.0)
        inst_spec_loud_right = torch.nan_to_num(inst_spec_loud_right, nan=0.0)
        
        if self.return_stages:
            stages['inst_spec_loud_left'] = inst_spec_loud_left
            stages['inst_spec_loud_right'] = inst_spec_loud_right
        
        # ========== Stage 5: Short-Term Temporal Integration ==========
        # MATLAB: moore2016_shorttermspecloudness
        # Returns both STL specific loudness (150 channels) and STL loudness (scalar)
        # But we only use STL specific loudness for binaural inhibition
        
        # Process each sample in batch separately (temporal integration needs sequential processing)
        stl_spec_loud_left_list = []
        stl_spec_loud_right_list = []
        
        for b in range(batch_size):
            # Left ear
            inst_left_b = inst_spec_loud_left[b]  # (n_frames, 150)
            _, stl_spec_left_b, stl_left_b = self.temporal_integration(inst_left_b, return_intermediate=True)
            stl_spec_loud_left_list.append(stl_spec_left_b)
            
            # Right ear
            inst_right_b = inst_spec_loud_right[b]  # (n_frames, 150)
            _, stl_spec_right_b, stl_right_b = self.temporal_integration(inst_right_b, return_intermediate=True)
            stl_spec_loud_right_list.append(stl_spec_right_b)
        
        stl_spec_loud_left = torch.stack(stl_spec_loud_left_list, dim=0)   # (batch, n_frames, 150)
        stl_spec_loud_right = torch.stack(stl_spec_loud_right_list, dim=0)
        
        if self.return_stages:
            stages['stl_spec_loud_left'] = stl_spec_loud_left
            stages['stl_spec_loud_right'] = stl_spec_loud_right
        
        # ========== Stage 6: Binaural Inhibition (frame-by-frame) ==========
        # MATLAB: moore2016_binauralloudness in a loop over frames
        # Returns scalar loudness per ear (already integrated / 4)
        
        n_frames = stl_spec_loud_left.shape[1]
        stl_loud_left_list = []
        stl_loud_right_list = []
        
        for b in range(batch_size):
            stl_left_frames = []
            stl_right_frames = []
            
            for t in range(n_frames):
                # Get frame t for left and right
                spec_left_t = stl_spec_loud_left[b, t, :]   # (150,)
                spec_right_t = stl_spec_loud_right[b, t, :]  # (150,)
                
                # Apply binaural inhibition (returns total, left, right - all (1,))
                _, loud_left_t, loud_right_t = self.binaural_loudness(spec_left_t.unsqueeze(0),   # (1, 150)
                                                                      spec_right_t.unsqueeze(0))  # (1, 150)
                
                # Squeeze to scalar
                stl_left_frames.append(loud_left_t.squeeze())   # scalar
                stl_right_frames.append(loud_right_t.squeeze())
            
            stl_loud_left_list.append(torch.stack(stl_left_frames, dim=0))   # (n_frames,)
            stl_loud_right_list.append(torch.stack(stl_right_frames, dim=0))
        
        stl_loud_left = torch.stack(stl_loud_left_list, dim=0)   # (batch, n_frames)
        stl_loud_right = torch.stack(stl_loud_right_list, dim=0)
        
        if self.return_stages:
            stages['stl_loud_left'] = stl_loud_left
            stages['stl_loud_right'] = stl_loud_right
        
        # ========== Stage 7: Long-Term Temporal Integration ==========
        # MATLAB: moore2016_longtermloudness (separate for left and right)
        
        ltl_loud_left_list = []
        ltl_loud_right_list = []
        
        for b in range(batch_size):
            ltl_left_b = self.ltl_agc_left(stl_loud_left[b])    # (n_frames,)
            ltl_right_b = self.ltl_agc_right(stl_loud_right[b])
            
            ltl_loud_left_list.append(ltl_left_b)
            ltl_loud_right_list.append(ltl_right_b)
        
        ltl_loud_left = torch.stack(ltl_loud_left_list, dim=0)   # (batch, n_frames)
        ltl_loud_right = torch.stack(ltl_loud_right_list, dim=0)
        
        if self.return_stages:
            stages['ltl_loud_left'] = ltl_loud_left
            stages['ltl_loud_right'] = ltl_loud_right
        
        # ========== Stage 8: Output ==========
        # MATLAB: sLoud = sLoudL + sLoudR, lLoud = lLoudL + lLoudR, mLoud = max(lLoud)
        
        sLoud = stl_loud_left + stl_loud_right   # (batch, n_frames)
        lLoud = ltl_loud_left + ltl_loud_right   # (batch, n_frames)
        mLoud = lLoud.max(dim=1)[0]              # (batch,)
        
        if self.return_stages:
            return (sLoud, lLoud, mLoud), stages
        else:
            return sLoud, lLoud, mLoud
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all model parameters.
        
        Returns
        -------
        dict
            Dictionary with model parameters:
            - 'fs': Sampling rate (32000 Hz)
            - 'learnable': Whether parameters are trainable
            - 'return_stages': Whether intermediate stages are returned
        """
        return {'fs': self.fs,
                'learnable': self.learnable,
                'return_stages': self.return_stages}
    
    def extra_repr(self) -> str:
        """
        Extra representation for printing.
        
        Returns
        -------
        str
            String representation of module parameters.
        """
        return (f"fs={self.fs}, learnable={self.learnable}, "
                f"return_stages={self.return_stages}")
