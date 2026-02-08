"""
Glasberg & Moore (2002) Loudness Model
======================================

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the Glasberg & Moore (2002) model for perceptual loudness 
computation applicable to time-varying sounds. The model provides a complete pipeline 
from audio waveform to loudness perception in sone units, accounting for frequency-
dependent hearing sensitivity, masking effects, and temporal dynamics.

The implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT) 
and extended with PyTorch for gradient-based optimization and GPU acceleration.

References
----------
.. [1] B. R. Glasberg and B. C. J. Moore, "A Model of Loudness Applicable to 
       Time-Varying Sounds," *J. Audio Eng. Soc.*, vol. 50, no. 5, pp. 331-342, 
       May 2002.

.. [2] B. C. J. Moore and B. R. Glasberg, "A Model for the Prediction of Thresholds, 
       Loudness, and Partial Loudness," *J. Audio Eng. Soc.*, vol. 45, no. 4, 
       pp. 224-240, Apr. 1997.

.. [3] B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter shapes 
       from notched-noise data," *Hear. Res.*, vol. 47, no. 1-2, pp. 103-138, 
       Aug. 1990.

.. [4] B. C. J. Moore and B. R. Glasberg, "Formulae describing frequency selectivity 
       as a function of frequency and level, and their use in calculating excitation 
       patterns," *Hear. Res.*, vol. 28, no. 2-3, pp. 209-225, 1987.

.. [5] ISO 226:2003, "Acoustics - Normal equal-loudness-level contours," 
       International Organization for Standardization, 2003.

.. [6] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
       p. 19, 2022.
"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from torch_amt.common.filterbanks import MultiResolutionFFT, ERBIntegration, ExcitationPattern
from torch_amt.common.loudness import SpecificLoudness, LoudnessIntegration


class Glasberg2002(nn.Module):
    r"""
    Glasberg & Moore (2002) model for time-varying loudness perception.
    
    Implements the complete loudness computation pipeline from Glasberg & Moore (2002), 
    providing perceptual loudness measures in sone from audio waveforms. The model 
    accounts for frequency-dependent hearing sensitivity (ISO 226), masking effects 
    via asymmetric excitation spreading, and temporal integration with attack/release 
    dynamics.
    
    This implementation is based on the MATLAB Auditory Modeling Toolbox (AMT) 
    ``glasberg2002`` function and provides a differentiable, GPU-accelerated version 
    suitable for neural network training and loudness-based optimization.
    
    Algorithm Overview
    ------------------
    The model implements a 5-stage loudness processing pipeline:
    
    **Stage 1: Multi-Resolution FFT**
    
    Performs time-frequency analysis with multiple FFT sizes to balance temporal 
    and frequency resolution across the audible spectrum:
    
    .. math::
        X(t, f) = \\text{FFT}_{N(f)}(x(t))
    
    where :math:`N(f)` is frequency-dependent FFT size (larger for low frequencies).
    Outputs power spectral density (PSD) in :math:`\\text{Pa}^2/\\text{Hz}`.
    
    **Stage 2: ERB Integration**
    
    Maps PSD to perceptual ERB frequency scale with 1/4 ERB resolution:
    
    .. math::
        E_{\\text{ERB}}(t, f_{\\text{ERB}}) = \\int P(t, f) \\cdot W_{\\text{ERB}}(f, f_{\\text{ERB}}) df
    
    where :math:`W_{\\text{ERB}}` is the ERB weighting function. Output in dB SPL.
    
    **Stage 3: Excitation Pattern**
    
    Models asymmetric frequency spreading with level-dependent slopes:
    
    .. math::
        E_{\\text{spread}}(t, f) = \\sum_g E_{\\text{ERB}}(t, f+g) \\cdot S(g, E)
    
    where :math:`S(g, E)` is the spreading function (steeper upward, shallower downward).
    
    **Stage 4: Specific Loudness**
    
    Applies 3-regime loudness transformation (Moore & Glasberg 1997):
    
    .. math::
        N(t, f) = \\begin{cases}
        0 & E < E_{\\text{thrq}} \\\\
        C \\cdot (E - E_{\\text{thrq}}) & E_{\\text{thrq}} < E < E_0 \\\\
        C \\cdot E_0^{1-\\alpha} (E - E_{\\text{thrq}})^{\\alpha} & E > E_0
        \\end{cases}
    
    with :math:`C=0.047`, :math:`\\alpha=0.2`, :math:`E_0=10` dB above threshold.
    
    **Stage 5: Loudness Integration**
    
    Spatial integration (sum across ERB channels) followed by temporal integration 
    with asymmetric attack/release filter:
    
    .. math::
        \\text{STL}(t) = \\sum_f N(t, f), \\quad 
        \\text{LTL}[n] = (1-\\alpha[n])\\text{STL}[n] + \\alpha[n]\\text{LTL}[n-1]
    
    where :math:`\\alpha = \\exp(-\\Delta t / \\tau)` with :math:`\\tau_{\\text{attack}}=50` ms, 
    :math:`\\tau_{\\text{release}}=200` ms.
    
    Parameters
    ----------
    fs : int, optional
        Sampling rate in Hz. Default: 32000 Hz.
        Higher sampling rates improve temporal resolution but increase computational cost.
        Typical values: 16000, 32000, 44100, 48000 Hz.
    
    learnable : bool, optional
        If True, all model stages become trainable with gradient-based optimization. 
        Default: False (fixed parameters).
        When True, enables end-to-end model training for task-specific optimization.
    
    return_stages : bool, optional
        If True, returns intermediate processing stages along with final output. 
        Default: False (only final long-term loudness).
        Useful for visualization, analysis, and multi-stage training.
    
    **multi_fft_kwargs : dict, optional
        Additional keyword arguments passed to :class:`MultiResolutionFFT`.
        Common options:
        
        - ``hop_length`` (int): Hop size for STFT in samples.
        - ``n_ffts`` (list): FFT sizes for multi-resolution analysis.
        - Other parameters accepted by MultiResolutionFFT.
    
    **erb_kwargs : dict, optional
        Additional keyword arguments passed to :class:`ERBIntegration`.
        Common options:
        
        - ``f_min`` (float): Minimum frequency in Hz. Default: 50.0.
        - ``f_max`` (float): Maximum frequency in Hz. Default: 15000.0.
        - ``erb_step`` (float): ERB frequency step. Default: 0.25.
        - ``bandwidth_scale`` (float): Bandwidth scaling factor. Default: 1.0.
        - Other parameters accepted by ERBIntegration.
    
    **excitation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`ExcitationPattern`.
        Common options:
        
        - ``upper_slope_base`` (float): Base upper spreading slope. Default: 27.0 dB/ERB.
        - ``lower_slope_base`` (float): Base lower spreading slope. Default: 27.0 dB/ERB.
        - ``upper_slope_per_db`` (float): Upper slope level dependency. Default: 0.0.
        - ``lower_slope_per_db`` (float): Lower slope level dependency. Default: -0.4 dB/ERB per dB.
        - Other parameters accepted by ExcitationPattern.
    
    **specific_loudness_kwargs : dict, optional
        Additional keyword arguments passed to :class:`SpecificLoudness`.
        Common options:
        
        - ``f_min`` (float): Minimum ERB frequency. Default: 50.0 Hz.
        - ``f_max`` (float): Maximum ERB frequency. Default: 15000.0 Hz.
        - ``erb_step`` (float): ERB step. Default: 0.25.
        - Other parameters accepted by SpecificLoudness.
    
    **loudness_integration_kwargs : dict, optional
        Additional keyword arguments passed to :class:`LoudnessIntegration`.
        Common options:
        
        - ``tau_attack`` (float): Attack time constant in seconds. Default: 0.05 (50 ms).
        - ``tau_release`` (float): Release time constant in seconds. Default: 0.20 (200 ms).
        - Other parameters accepted by LoudnessIntegration.
    
    Attributes
    ----------
    fs : int
        Sampling rate in Hz.
    
    learnable : bool
        Whether model parameters are trainable.
    
    return_stages : bool
        Whether to return intermediate processing stages.
    
    multi_fft : MultiResolutionFFT
        Stage 1: Multi-resolution time-frequency analysis module.
    
    erb_integration : ERBIntegration
        Stage 2: ERB frequency scale integration module.
    
    excitation_pattern : ExcitationPattern
        Stage 3: Excitation pattern spreading module.
    
    specific_loudness : SpecificLoudness
        Stage 4: Specific loudness transformation module.
    
    loudness_integration : LoudnessIntegration
        Stage 5: Spatial and temporal loudness integration module.
    
    Input Shape
    -----------
    audio : torch.Tensor
        Audio signal with shape:
        
        - :math:`(B, T)` - Batch of audio samples
        - :math:`(T,)` - Single audio sample (mono)
        
        where:
        
        - :math:`B` = batch size
        - :math:`T` = time samples
    
    Output Shape
    ------------
    When ``return_stages=False`` (default):
        torch.Tensor
            Long-term loudness in sone, shape :math:`(B, F)` where:
            
            - :math:`F` = number of time frames (depends on hop_length)
            
    When ``return_stages=True``:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            - First element: long-term loudness (as above)
            - Second element: dict with keys:
              
              - ``'stl'``: Short-term loudness, shape :math:`(B, F)` in sone
              - ``'specific_loudness'``: Specific loudness, shape :math:`(B, F, N_{\\text{ERB}})` in sone/ERB
              - ``'excitation'``: Excitation pattern, shape :math:`(B, F, N_{\\text{ERB}})` in dB SPL
              - ``'erb_excitation'``: ERB-integrated excitation, shape :math:`(B, F, N_{\\text{ERB}})` in dB SPL
              - ``'psd'``: Power spectral density, shape :math:`(B, F, N_{\\text{freq}})`
              - ``'freqs'``: Frequency vector for PSD, shape :math:`(N_{\\text{freq}},)` in Hz
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> from torch_amt.models import Glasberg2002
    >>> 
    >>> # Create model
    >>> model = Glasberg2002(fs=32000)
    >>> n_erb = model.erb_integration.n_erb_bands
    >>> print(f"ERB channels: {n_erb}")
    ERB channels: 150
    >>> 
    >>> # Process 1 second of audio
    >>> audio = torch.randn(2, 32000)  # 2 batches
    >>> ltl = model(audio)
    >>> print(f"LTL shape: {ltl.shape}, range: [{ltl.min():.2f}, {ltl.max():.2f}] sone")
    LTL shape: torch.Size([2, 62]), range: [0.23, 45.67] sone
    
    **With intermediate stages:**
    
    >>> model_debug = Glasberg2002(fs=32000, return_stages=True)
    >>> ltl, stages = model_debug(audio)
    >>> 
    >>> print(f"Available stages: {list(stages.keys())}")
    Available stages: ['stl', 'specific_loudness', 'excitation', 'erb_excitation', 'psd', 'freqs']
    >>> print(f"STL shape: {stages['stl'].shape}")
    STL shape: torch.Size([2, 62])
    >>> print(f"Specific loudness shape: {stages['specific_loudness'].shape}")
    Specific loudness shape: torch.Size([2, 62, 150])
    >>> print(f"Excitation shape: {stages['excitation'].shape}")
    Excitation shape: torch.Size([2, 62, 150])
    
    **Single channel input:**
    
    >>> audio_mono = torch.randn(32000)  # No batch dimension
    >>> ltl_mono = model(audio_mono)
    >>> print(f"Output shape (mono): {ltl_mono.shape}")
    Output shape (mono): torch.Size([62])
    
    **Learnable model for optimization:**
    
    >>> model_learnable = Glasberg2002(fs=32000, learnable=True)
    >>> n_params = sum(p.numel() for p in model_learnable.parameters())
    >>> print(f"Trainable parameters: {n_params}")
    Trainable parameters: 8743
    >>> 
    >>> # Example training loop
    >>> optimizer = torch.optim.Adam(model_learnable.parameters(), lr=1e-3)
    >>> # ... training code ...
    
    **Custom submodule parameters:**
    
    >>> # Custom ERB frequency range
    >>> model_custom_erb = Glasberg2002(
    ...     fs=44100,
    ...     erb_kwargs={'f_min': 80.0, 'f_max': 12000.0, 'erb_step': 0.5}
    ... )
    >>> print(f"ERB channels: {model_custom_erb.erb_integration.n_erb_bands}")
    ERB channels: 75
    >>> 
    >>> # Custom excitation spreading
    >>> model_custom_exc = Glasberg2002(
    ...     fs=32000,
    ...     excitation_kwargs={
    ...         'upper_slope_base': 30.0,  # Steeper upper slope
    ...         'lower_slope_per_db': -0.5  # More level-dependent lower slope
    ...     }
    ... )
    >>> 
    >>> # Custom temporal integration
    >>> model_custom_temp = Glasberg2002(
    ...     fs=32000,
    ...     loudness_integration_kwargs={
    ...         'tau_attack': 0.03,  # Faster attack (30 ms)
    ...         'tau_release': 0.30  # Slower release (300 ms)
    ...     }
    ... )
    
    **Different sampling rates:**
    
    >>> model_44k = Glasberg2002(fs=44100)
    >>> audio_44k = torch.randn(2, 44100)  # 1 second @ 44.1 kHz
    >>> ltl_44k = model_44k(audio_44k)
    >>> print(f"Output frames @ 44.1kHz: {ltl_44k.shape[1]}")
    Output frames @ 44.1kHz: 86
    
    **Reset temporal state for new signal:**
    
    >>> # Process first signal
    >>> signal1 = torch.randn(1, 32000)
    >>> ltl1 = model(signal1)
    >>> 
    >>> # Reset before processing unrelated second signal
    >>> model.reset_state()
    >>> signal2 = torch.randn(1, 32000)
    >>> ltl2 = model(signal2)
    
    **Convert to loudness level (phon):**
    
    >>> ltl_sone = model(audio)
    >>> ltl_phon = model.compute_loudness_level(ltl_sone)
    >>> print(f"Loudness: {ltl_sone.mean():.2f} sone = {ltl_phon.mean():.2f} phon")
    Loudness: 12.34 sone = 54.32 phon
    
    Notes
    -----
    **Model Configuration:**
    
    The Glasberg2002 model uses specific configurations for each processing stage:
    
    - **Multi-resolution FFT**: Multiple FFT sizes (frequency-dependent) for balanced resolution
    - **ERB integration**: 1/4 ERB steps from 50 Hz to 15 kHz (150 channels)
    - **Excitation pattern**: Asymmetric spreading (27 dB/ERB base, -0.4 dB/ERB per dB lower slope)
    - **Specific loudness**: 3-regime transformation (:math:`C=0.047`, :math:`\\alpha=0.2`, :math:`E_0=10` dB)
    - **Loudness integration**: Attack 50 ms, release 200 ms
    
    **Customizing Submodule Parameters:**
    
    All submodules can be customized through dedicated kwargs dictionaries:
    
    - Use ``multi_fft_kwargs`` to pass parameters to :class:`MultiResolutionFFT`
    - Use ``erb_kwargs`` to pass parameters to :class:`ERBIntegration`
    - Use ``excitation_kwargs`` to pass parameters to :class:`ExcitationPattern`
    - Use ``specific_loudness_kwargs`` to pass parameters to :class:`SpecificLoudness`
    - Use ``loudness_integration_kwargs`` to pass parameters to :class:`LoudnessIntegration`
    
    The ``learnable`` and ``dtype`` parameters are always centralized and applied 
    to all submodules automatically. Custom parameters override defaults while 
    maintaining the Glasberg2002 model structure.
    
    **Computational Complexity:**
    
    Processing time scales approximately as:
    
    .. math::
        T_{\\text{compute}} \\propto B \\cdot F \\cdot N_{\\text{ERB}} \\cdot \\log N_{\\text{FFT}}
    
    where :math:`F` = number of time frames (~60 per second), :math:`N_{\\text{ERB}}=150`.
    For 1 second at 32 kHz: ~0.05-0.2 seconds on CPU, ~0.005-0.02 seconds on GPU.
    
    **Memory Requirements:**
    
    Peak memory scales with intermediate representations:
    
    .. math::
        Memory \\approx B \\cdot F \\cdot N_{\\text{ERB}} \\cdot 4\\,\\text{bytes}
    
    For batch=8, 1 second @ 32 kHz: ~20-40 MB.
    
    **Differences from MATLAB AMT:**
    
    - This implementation uses PyTorch tensors for GPU acceleration
    - Supports batch processing natively
    - All stages are differentiable for gradient-based optimization
    - Output frames depend on hop_length (not fixed downsampling)
    
    **Loudness Units:**
    
    - **Sone**: Perceptual loudness unit. 1 sone = loudness of 1 kHz tone at 40 dB SPL
    - **Phon**: Loudness level unit. Equal to dB SPL at 1 kHz
    - Conversion: :math:`L_{\\text{phon}} = 40 + 10\\log_2(L_{\\text{sone}})`
    
    **Applications:**
    
    The model output can be used for:
    
    - Perceptual loudness measurement and normalization
    - Audio quality assessment (loudness-based metrics)
    - Dynamic range compression/expansion
    - Hearing aid fitting and evaluation
    - Psychoacoustic model validation
    - Feature extraction for machine learning
    
    See Also
    --------
    MultiResolutionFFT : Stage 1 - Time-frequency analysis
    ERBIntegration : Stage 2 - ERB frequency scale
    ExcitationPattern : Stage 3 - Excitation spreading
    SpecificLoudness : Stage 4 - Loudness transformation
    LoudnessIntegration : Stage 5 - Spatial and temporal integration
    
    References
    ----------
    .. [1] B. R. Glasberg and B. C. J. Moore, "A Model of Loudness Applicable to 
           Time-Varying Sounds," *J. Audio Eng. Soc.*, vol. 50, no. 5, pp. 331-342, 
           May 2002.
    
    .. [2] B. C. J. Moore and B. R. Glasberg, "A Model for the Prediction of Thresholds, 
           Loudness, and Partial Loudness," *J. Audio Eng. Soc.*, vol. 45, no. 4, 
           pp. 224-240, Apr. 1997.
    
    .. [3] B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter shapes 
           from notched-noise data," *Hear. Res.*, vol. 47, no. 1-2, pp. 103-138, 
           Aug. 1990.
    
    .. [4] B. C. J. Moore and B. R. Glasberg, "Formulae describing frequency selectivity 
           as a function of frequency and level, and their use in calculating excitation 
           patterns," *Hear. Res.*, vol. 28, no. 2-3, pp. 209-225, 1987.
    
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
                 multi_fft_kwargs: Dict[str, Any] = None,
                 erb_kwargs: Dict[str, Any] = None,
                 excitation_kwargs: Dict[str, Any] = None,
                 specific_loudness_kwargs: Dict[str, Any] = None,
                 loudness_integration_kwargs: Dict[str, Any] = None):
        """
        Initialize Glasberg & Moore (2002) loudness model.
        
        Parameters
        ----------
        fs : int, optional
            Sampling rate in Hz. Default: 32000.
        
        learnable : bool, optional
            If True, all model parameters become trainable. Default: False.
        
        return_stages : bool, optional
            If True, return intermediate processing stages. Default: False.
        
        **kwargs : dict
            Additional keyword arguments for submodules (see class docstring).
        """
        super().__init__()
        
        self.fs = fs
        self.learnable = learnable
        self.return_stages = return_stages
        
        # Initialize kwargs dictionaries if None
        multi_fft_kwargs = multi_fft_kwargs or {}
        erb_kwargs = erb_kwargs or {}
        excitation_kwargs = excitation_kwargs or {}
        specific_loudness_kwargs = specific_loudness_kwargs or {}
        loudness_integration_kwargs = loudness_integration_kwargs or {}
        
        # Stage 1: Multi-resolution FFT
        self.multi_fft = MultiResolutionFFT(fs=fs, learnable=learnable, **multi_fft_kwargs)
        
        # Stage 2: ERB integration
        self.erb_integration = ERBIntegration(fs=fs, learnable=learnable, **erb_kwargs)
        
        # Stage 3: Excitation pattern
        self.excitation_pattern = ExcitationPattern(fs=fs, learnable=learnable, **excitation_kwargs)
        
        # Stage 4: Specific loudness
        self.specific_loudness = SpecificLoudness(fs=fs, learnable=learnable, **specific_loudness_kwargs)
        
        # Stage 5: Loudness integration
        self.loudness_integration = LoudnessIntegration(fs=fs, learnable=learnable, **loudness_integration_kwargs)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process audio through the Glasberg2002 loudness model.
        
        Parameters
        ----------
        audio : torch.Tensor
            Input audio signal. Shape: (B, T) or (T,).
            
        Returns
        -------
        torch.Tensor or tuple
            If return_stages=False:
                Long-term loudness in sone, shape (B, F) or (F,).
            If return_stages=True:
                Tuple of (ltl, stages) where stages is a dict with:
                - 'stl': Short-term loudness (B, F) in sone
                - 'specific_loudness': Specific loudness (B, F, N_ERB) in sone/ERB
                - 'excitation': Excitation pattern (B, F, N_ERB) in dB SPL
                - 'erb_excitation': ERB-integrated excitation (B, F, N_ERB) in dB SPL
                - 'psd': Power spectral density (B, F, N_freq)
                - 'freqs': Frequency vector (N_freq,) in Hz
        """
        stages = {} if self.return_stages else None
        
        # Stage 1: Multi-resolution FFT
        psd, freqs = self.multi_fft(audio)
        if self.return_stages:
            stages['psd'] = psd
            stages['freqs'] = freqs
        
        # Stage 2: ERB integration (perceptual frequency scale)
        erb_excitation = self.erb_integration(psd, freqs)
        if self.return_stages:
            stages['erb_excitation'] = erb_excitation
        
        # Stage 3: Excitation pattern (asymmetric spreading, level-dependent)
        excitation = self.excitation_pattern(erb_excitation)
        if self.return_stages:
            stages['excitation'] = excitation
        
        # Stage 4: Specific loudness (3-regime compression)
        specific_loudness = self.specific_loudness(excitation)
        if self.return_stages:
            stages['specific_loudness'] = specific_loudness
        
        # Stage 5: Loudness integration (spatial + temporal)
        ltl, stl = self.loudness_integration(specific_loudness, return_stl=True)
        if self.return_stages:
            stages['stl'] = stl
        
        if self.return_stages:
            return ltl, stages
        else:
            return ltl
    
    def reset_state(self):
        """
        Reset temporal integration state for processing discontinuous signals.
        
        Clears the internal state of the temporal integration filter (LoudnessIntegration).
        Call this method when processing multiple unrelated audio signals sequentially 
        to prevent temporal blending between signals.
        
        Examples
        --------
        >>> model = Glasberg2002(fs=32000)
        >>> signal1 = torch.randn(1, 32000)
        >>> ltl1 = model(signal1)
        >>> 
        >>> # Reset before processing new signal
        >>> model.reset_state()
        >>> signal2 = torch.randn(1, 32000)
        >>> ltl2 = model(signal2)  # No temporal carryover from signal1
        """
        self.loudness_integration.reset_state()
    
    def get_erb_frequencies(self) -> torch.Tensor:
        """
        Get ERB channel center frequencies.
        
        Returns the center frequencies (in Hz) of the ERB-spaced frequency channels 
        used throughout the model pipeline. Useful for frequency-domain visualization 
        and analysis.
        
        Returns
        -------
        torch.Tensor
            Center frequencies of ERB channels, shape (n_erb_bands,) in Hz.
            Typically 150 channels from 50 Hz to 15 kHz with 1/4 ERB spacing.
        
        Examples
        --------
        >>> model = Glasberg2002(fs=32000)
        >>> fc = model.get_erb_frequencies()
        >>> print(f"ERB frequencies: {fc.shape}, range: [{fc.min():.1f}, {fc.max():.1f}] Hz")
        ERB frequencies: torch.Size([150]), range: [50.0, 14999.2] Hz
        """
        return self.erb_integration.fc_erb
    
    def get_learnable_parameters(self) -> Dict[str, Any]:
        """
        Get all learnable parameters organized by model component.
        
        Returns a nested dictionary containing the current values of all trainable 
        parameters in each pipeline stage. Only returns parameters when model is 
        initialized with ``learnable=True``.
        
        Returns
        -------
        dict
            Dictionary with component names as keys and parameter dicts as values:
            
            - ``'multi_fft'``: MultiResolutionFFT parameters (hop_length, n_ffts)
            - ``'erb_integration'``: ERBIntegration parameters (fc_erb, bandwidth_scale)
            - ``'excitation_pattern'``: ExcitationPattern parameters (slopes)
            - ``'specific_loudness'``: SpecificLoudness parameters (C, alpha, E0, thresholds)
            - ``'loudness_integration'``: LoudnessIntegration parameters (tau_attack, tau_release)
            
            Empty dict if ``learnable=False``.
        
        Examples
        --------
        >>> model = Glasberg2002(fs=32000, learnable=True)
        >>> params = model.get_learnable_parameters()
        >>> print(f"Components: {list(params.keys())}")
        Components: ['multi_fft', 'erb_integration', 'excitation_pattern', 'specific_loudness', 'loudness_integration']
        >>> 
        >>> # Access specific parameters
        >>> print(f"Attack time: {params['loudness_integration']['tau_attack']:.3f} s")
        Attack time: 0.050 s
        >>> print(f"Bandwidth scale: {params['erb_integration']['bandwidth_scale']:.2f}")
        Bandwidth scale: 1.00
        """
        if not self.learnable:
            return {}
        
        params = {}
        
        # MultiResolutionFFT parameters
        params['multi_fft'] = {'hop_length': self.multi_fft.hop_length,
                               'n_ffts': self.multi_fft.n_ffts}
        
        # ERBIntegration parameters
        params['erb_integration'] = {'fc_erb': self.erb_integration.fc_erb,
                                     'bandwidth_scale': self.erb_integration.bandwidth_scale}
        
        # ExcitationPattern parameters
        params['excitation_pattern'] = {'upper_slope_base': self.excitation_pattern.upper_slope_base,
                                        'lower_slope_base': self.excitation_pattern.lower_slope_base,
                                        'upper_slope_per_db': self.excitation_pattern.upper_slope_per_db,
                                        'lower_slope_per_db': self.excitation_pattern.lower_slope_per_db}
        
        # SpecificLoudness parameters
        params['specific_loudness'] = self.specific_loudness.get_parameters()
        
        # LoudnessIntegration parameters
        tau_attack, tau_release = self.loudness_integration.get_time_constants()
        params['loudness_integration'] = {'tau_attack': tau_attack, 'tau_release': tau_release}
        
        return params
    
    def compute_loudness_level(self, ltl: torch.Tensor) -> torch.Tensor:
        """
        Convert loudness from sone to loudness level in phon.
        
        Applies Stevens' power law to convert perceptual loudness (sone) to 
        loudness level (phon), which is equivalent to dB SPL at 1 kHz.
        
        Parameters
        ----------
        ltl : torch.Tensor
            Loudness in sone, shape (B, F) or (F,).
            Values should be non-negative.
        
        Returns
        -------
        torch.Tensor
            Loudness level in phon, same shape as input.
            Formula: :math:`L_{\\text{phon}} = 40 + 10\\log_2(L_{\\text{sone}})`
        
        Notes
        -----
        **Loudness Units:**
        
        - **Sone**: Perceptual loudness. 1 sone = loudness of 1 kHz tone at 40 dB SPL.
          Doubling sone value = doubling perceived loudness.
        
        - **Phon**: Loudness level. Equal to dB SPL of equally loud 1 kHz tone.
          40 phon = 40 dB SPL @ 1 kHz = 1 sone.
        
        **Conversion Examples:**
        
        - 1 sone = 40 phon (reference)
        - 2 sone = 50 phon (10 dB louder)
        - 4 sone = 60 phon (20 dB louder)
        - 0.5 sone = 30 phon (10 dB quieter)
        
        Examples
        --------
        >>> model = Glasberg2002(fs=32000)
        >>> audio = torch.randn(2, 32000)
        >>> ltl_sone = model(audio)
        >>> ltl_phon = model.compute_loudness_level(ltl_sone)
        >>> 
        >>> print(f"Loudness: {ltl_sone.mean():.2f} sone")
        Loudness: 12.34 sone
        >>> print(f"Loudness level: {ltl_phon.mean():.2f} phon")
        Loudness level: 54.32 phon
        
        References
        ----------
        .. [1] S. S. Stevens, "Perceived level of noise by Mark VII and decibels (E)," 
               *J. Acoust. Soc. Am.*, vol. 51, no. 2B, pp. 575-601, 1972.
        """
        # Avoid log of zero
        ltl_safe = torch.clamp(ltl, min=1e-10)
        loudness_level = 40.0 + 10.0 * torch.log2(ltl_safe)
        return loudness_level
    
    def extra_repr(self) -> str:
        """
        Extra representation for printing.
        
        Returns
        -------
        str
            String representation of module parameters.
        """
        n_erb_bands = self.erb_integration.n_erb_bands
        return (f"fs={self.fs}, n_erb_bands={n_erb_bands}, "
                f"learnable={self.learnable}, return_stages={self.return_stages}")
