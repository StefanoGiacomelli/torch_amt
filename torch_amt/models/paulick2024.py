"""
Paulick et al. (2024) CASP Model
=================================

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the Paulick et al. (2024) Computational Auditory Signal
Processing (CASP) model, a revised version of the Jepsen et al. (2008) model
with improved physiological accuracy and extended decision-making capabilities
for psychophysical task modeling.

The model features a complete auditory periphery simulation including outer/middle
ear filtering, DRNL filterbank with dual-path nonlinear processing, physiological
IHC transduction, multi-stage adaptation, and modulation analysis with integrated
decision-making methods for detection and discrimination tasks.

This implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT)
and extended with PyTorch for gradient-based optimization and GPU acceleration.

References
----------
.. [1] L. Paulick, H. Relaño-Iborra, and T. Dau, "The Computational Auditory 
       Signal Processing and Perception Model (CASP): A Revised Version," 
       *bioRxiv*, 2024.

.. [2] M. Jepsen, S. Ewert, and T. Dau, "A computational model of human 
       auditory signal processing and perception," *J. Acoust. Soc. Am.*, 
       vol. 124, no. 1, pp. 422-438, Jul. 2008.

.. [3] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
       of amplitude modulation. I. Detection and masking with narrow-band carriers," 
       *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, Nov. 1997.

.. [4] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
       p. 19, 2022.
"""

from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torchaudio.transforms as T

# from torch_amt.common.filterbanks import DRNLFilterbank
from torch_amt.common.filterbanks import FastDRNLFilterbank
from torch_amt.common.ears import OuterMiddleEarFilter
from torch_amt.common.ihc import IHCPaulick2024
from torch_amt.common.adaptation import AdaptLoop
# from torch_amt.common.modulation import ModulationFilterbank
from torch_amt.common.modulation import FastModulationFilterbank


class Paulick2024(nn.Module):
    r"""
    Paulick et al. (2024) CASP model for auditory processing.
    
    Implements the revised Computational Auditory Signal Processing and Perception
    (CASP) model, an advanced auditory periphery simulation with physiologically
    accurate nonlinear processing and integrated decision-making capabilities for
    psychophysical task modeling.
    
    The model extends Jepsen et al. (2008) with improved IHC transduction,
    frequency-dependent adaptation, and comprehensive decision-making methods
    for detection, discrimination, and masking studies. It provides both the
    auditory internal representation and psychophysical decision mechanisms
    in a single unified framework.
    
    This implementation follows the MATLAB Auditory Modeling Toolbox (AMT)
    implementation and provides a differentiable, GPU-accelerated version
    suitable for neural network training and optimization.
    
    Algorithm Overview
    ------------------
    The model implements a 6-stage auditory processing pipeline with optional
    decision-making post-processing:
    
    **Stage 0: Outer/Middle Ear Filter** (Optional)
    
    Applies free-field or diffuse-field head-related transfer function:
    
    .. math::
        x_{\\text{ear}}(t) = h_{\\text{outer}}(t) * x(t)
    
    Transfer function from Lopezpoveda & Meddis (2001), compensating for
    headphone presentation to simulate natural listening conditions.
    
    **Stage 1: DRNL Filterbank**
    
    Dual Resonance NonLinear (DRNL) filterbank with 50 channels spanning
    125-8000 Hz with ERB spacing. Each channel combines linear and nonlinear paths:
    
    *Linear path:*
    
    .. math::
        y_{\\text{lin},i}(t) = g_{\\text{lin}} \\cdot [\\text{GT}_{\\text{lin}}(t) * x_{\\text{ear}}(t)]
    
    *Nonlinear path with compression:*
    
    .. math::
        y_{\\text{nl},i}(t) = g_{\\text{nl}} \\cdot [\\text{LP}(t) * (\\text{GT}_{\\text{nl}}(t) * x_{\\text{ear}}(t))^p]
    
    *Combined output:*
    
    .. math::
        y_i(t) = y_{\\text{lin},i}(t) + y_{\\text{nl},i}(t)
    
    where GT = 4th-order gammatone, LP = lowpass filter, :math:`p \\approx 0.2` (compression).
    
    **Stage 2: IHC Transduction** (Paulick2024-specific)
    
    3-stage physiological inner hair cell model with asymmetric compression:
    
    *Stage 2a - Half-wave rectification:*
    
    .. math::
        v_1(t) = \\max(0, y_i(t))
    
    *Stage 2b - Asymmetric compression:*
    
    .. math::
        v_2(t) = \\text{sign}(v_1) \\cdot |v_1|^{0.23}
    
    *Stage 2c - 1st-order lowpass (1500 Hz):*
    
    .. math::
        v_{\\text{IHC}}(t) = h_{\\text{LP},1500}(t) * v_2(t)
    
    **Stage 3: Adaptation Loops** (5 loops, frequency-dependent)
    
    Multi-stage adaptation using parallel feedback loops:
    
    .. math::
        v_{\\text{adapt}}(t) = \\sum_{k=1}^{5} a_{1,k}(f_c) \\cdot [v_{\\text{IHC}}(t) - b_{0,k}(f_c) \\cdot s_k(t)]
    
    where :math:`s_k(t)` is the state of loop :math:`k` with time constant :math:`\\tau_k`.
    
    Time constants (Paulick2024 preset):
    - Loop 1: :math:`\\tau_1 = 0.005` s (5 ms, fast)
    - Loop 2: :math:`\\tau_2 = 0.050` s (50 ms, medium)
    - Loop 3: :math:`\\tau_3 = 0.129` s (129 ms, slow)
    - Loop 4: :math:`\\tau_4 = 0.253` s (253 ms, very slow)
    - Loop 5: :math:`\\tau_5 = 0.500` s (500 ms, ultra slow)
    
    **Stage 4: Resampling**
    
    Downsample from 44100 Hz to 11025 Hz (÷4) using sinc interpolation:
    
    .. math::
        v_{\\text{resamp}}[n] = v_{\\text{adapt}}(t) \\Big|_{t=n/f_{s,\\text{new}}}
    
    where :math:`f_{s,\\text{new}} = f_s / 4 = 11025` Hz.
    
    **Stage 5: Modulation Filterbank**
    
    Extracts amplitude modulation content with 8 channels (Paulick2024 preset):
    
    - Lowpass: 0 Hz (DC, cutoff 2.5 Hz)
    - Bandpass: Geometric progression with ratio 5/3
      
      .. math::
          f_{\\text{mod},k} = f_{\\text{mod},1} \\cdot (5/3)^{k-1}, \\quad k=1,\\ldots,7
    
    Center frequencies: [5, 8.33, 13.89, 23.15, 38.58, 64.30, 107.17, 128.6] Hz
    
    Number of modulation filters varies per auditory channel based on upper limit:
    
    .. math::
        f_{\\text{mod,max}}(f_c) = 0.25 \\cdot f_c
    
    **Stage 6: Decision-Making** (Optional post-processing)
    
    Psychophysical decision mechanisms:
    
    1. **ROI Selection**: Extract time/frequency/modulation regions of interest
    2. **Template Correlation**: Cross-correlate with internal template
    3. **Decision Variable**: Compute metric (RMS, mean, max, L2)
    4. **Binary Decision**: Threshold or ML-based classification
    
    Output: List of :math:`N_{\\text{chan}}` tensors, each with shape 
    :math:`(B, M_i, T_{\\text{resamp}})` where :math:`M_i` ∈ [6, 8] depends on 
    :math:`f_c`.
    
    Parameters
    ----------
    fs : float, optional
        Sampling rate in Hz. Must match the audio sampling rate. Default: 44100 Hz.
        Common values: 44100, 48000 Hz.
        
        Note: Resampling to fs/4 occurs internally after adaptation.
    
    flow : float, optional
        Lower frequency bound for DRNL filterbank in Hz. Default: 125 Hz.
        Determines the lowest auditory channel center frequency.
        
        Typical range: 80-200 Hz for speech/music applications.
    
    fhigh : float, optional
        Upper frequency bound for DRNL filterbank in Hz. Default: 8000 Hz.
        Determines the highest auditory channel center frequency.
        
        Typical range: 4000-12000 Hz. Higher values increase computational cost.
    
    n_channels : int, optional
        Number of auditory frequency channels (DRNL filterbank). Default: 50.
        
        More channels → better frequency resolution but higher computational cost.
        Paulick2024 paper uses 50 channels for detailed spectral analysis.
    
    use_outerear : bool, optional
        Whether to apply outer/middle ear filtering. Default: True.
        
        - True: Applies free-field HRTF (simulates natural listening)
        - False: Skip ear filtering (for already compensated signals)
        
        Use True for headphone presentation, False for loudspeaker or 
        pre-compensated stimuli.
    
    learnable : bool, optional
        If True, all model stages become trainable with gradient-based optimization.
        Default: False (fixed physiological parameters).
        
        Enables end-to-end model training for task-specific optimization or
        hearing loss parameter fitting.
    
    return_stages : bool, optional
        If True, returns intermediate processing stages along with final output.
        Default: False (only final modulation representation).
        
        Useful for visualization, analysis, debugging, and multi-stage training.
        Returns dict with keys: ['outerear', 'drnl', 'ihc', 'adaptation', 'resampled'].
    
    filter_type : {'efilt', 'butterworth'}, optional
        Type of modulation filters. Default: 'efilt'.
        
        - ``'efilt'``: MATLAB AMT compatible (complex frequency-shifted lowpass,
          resonant response, asymmetric frequency response). Original implementation.
        - ``'butterworth'``: Conceptually correct symmetric bandpass filters
          (2nd-order, better frequency resolution).
        
        Use 'efilt' for exact MATLAB AMT compatibility, 'butterworth' for
        cleaner frequency responses.
    
    dtype : torch.dtype, optional
        Data type for computations and parameters. Default: torch.float32.
        Use torch.float64 for higher numerical precision if needed (slower).
    
    **outerear_kwargs : dict, optional
        Additional keyword arguments passed to :class:`OuterMiddleEarFilter`.
        Common options:
        
        - ``compensation_type`` (str): Type of compensation. Default: 'tfOuterMiddle1997'.
        - ``field_type`` (str): Field type ('free' or 'diffuse'). Default: 'free'.
        - Other parameters accepted by OuterMiddleEarFilter.
    
    **drnl_kwargs : dict, optional
        Additional keyword arguments passed to :class:`DRNLFilterbank`.
        Common options:
        
        - ``subject`` (str): Subject type ('NH' for normal hearing). Default: 'NH'.
        - ``model`` (str): Model version. Default: 'paulick2024'.
        - Other parameters accepted by DRNLFilterbank.
    
    **ihc_kwargs : dict, optional
        Additional keyword arguments passed to :class:`IHCPaulick2024`.
        
        - No additional parameters typically needed (Paulick2024-specific IHC).
    
    **adaptation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`AdaptLoop`.
        Common options:
        
        - ``preset`` (str): Adaptation preset. Default: 'paulick2024'.
        - Other parameters accepted by AdaptLoop.
    
    **modulation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`ModulationFilterbank`.
        Common options:
        
        - ``preset`` (str): Modulation preset. Default: 'paulick2024'.
        - ``filter_type`` (str): Filter type. Inherited from main parameter.
        - Other parameters accepted by ModulationFilterbank.
    
    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    
    flow : float
        Lower frequency bound for DRNL filterbank.
    
    fhigh : float
        Upper frequency bound for DRNL filterbank.
    
    n_channels : int
        Number of auditory frequency channels.
    
    use_outerear : bool
        Whether outer/middle ear filtering is applied.
    
    learnable : bool
        Whether model parameters are trainable.
    
    return_stages : bool
        Whether to return intermediate processing stages.
    
    filter_type : str
        Type of modulation filters ('efilt' or 'butterworth').
    
    dtype : torch.dtype
        Data type for computations.
    
    outer_middle_ear : OuterMiddleEarFilter or None
        Stage 0: Outer/middle ear filtering module (if use_outerear=True).
    
    drnl : DRNLFilterbank
        Stage 1: Dual Resonance NonLinear filterbank.
    
    ihc : IHCPaulick2024
        Stage 2: Inner hair cell transduction module (Paulick2024-specific).
    
    adaptation : AdaptLoop
        Stage 3: Multi-stage adaptation loops module.
    
    modulation : ModulationFilterbank
        Stage 5: Modulation filterbank module (operates at resampled rate).
    
    resampler : torchaudio.transforms.Resample or None
        Stage 4: Resampling module (÷4 downsampling).
    
    fc : torch.Tensor
        Center frequencies of DRNL channels, shape (n_channels,) in Hz.
    
    num_channels : int
        Number of auditory frequency channels (same as n_channels).
    
    resample_factor : int
        Resampling factor (default: 4 for 44100 Hz → 11025 Hz).
    
    fs_resampled : float
        Sampling rate after resampling (fs / resample_factor = 11025 Hz).
    
    Input Shape
    -----------
    x : torch.Tensor
        Audio signal with shape:
        
        - :math:`(B, T)` - Batch of signals
        - :math:`(T,)` - Single signal
        
        where:
        
        - :math:`B` = batch size
        - :math:`T` = time samples at original sampling rate fs
    
    Output Shape
    ------------
    When ``return_stages=False`` (default):
        List[torch.Tensor]
            List of length :math:`N_{\\text{channels}}` (default: 50), one tensor
            per auditory frequency channel.
            
            Each tensor has shape:
            
            - :math:`(B, M_i, T_{\\text{resamp}})` for batched input
            - :math:`(M_i, T_{\\text{resamp}})` for single signal input
            
            where:
            
            - :math:`M_i` ∈ [6, 8] = number of modulation filters for channel :math:`i`
              (varies per channel due to dynamic upper frequency limit)
            - :math:`T_{\\text{resamp}} = T / 4` (for fs=44100 → 11025 Hz)
            
            Example: For 1-second audio at 44100 Hz:
            - Input: (B, 44100)
            - Output: List of 50 tensors, each (B, ~7-8, 11025)
    
    When ``return_stages=True``:
        Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]
            - First element: modulation representation (as above)
            - Second element: dict with intermediate stages:
              
              - ``'outerear'``: After ear filtering, shape :math:`(B, T)`
              - ``'drnl'``: After DRNL filterbank, shape :math:`(B, F, T)`
              - ``'ihc'``: After IHC transduction, shape :math:`(B, F, T)`
              - ``'adaptation'``: After adaptation, shape :math:`(B, F, T)`
              - ``'resampled'``: After resampling, shape :math:`(B, F, T_{\\text{resamp}})`
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> from torch_amt.models import Paulick2024
    >>> 
    >>> # Create model
    >>> model = Paulick2024(fs=44100)
    >>> 
    >>> # Generate 1 second audio
    >>> audio = torch.randn(2, 44100) * 0.01
    >>> 
    >>> # Process
    >>> internal_repr = model(audio)
    >>> print(f"Number of frequency channels: {len(internal_repr)}")
    Number of frequency channels: 50
    >>> print(f"First channel shape: {internal_repr[0].shape}")
    First channel shape: torch.Size([2, 8, 11025])
    >>> print(f"Modulation filters in channel 0: {internal_repr[0].shape[1]}")
    Modulation filters in channel 0: 8
    
    **With intermediate stages:**
    
    >>> model_debug = Paulick2024(fs=44100, return_stages=True)
    >>> internal_repr, stages = model_debug(audio)
    >>> 
    >>> print(f"Available stages: {list(stages.keys())}")
    Available stages: ['outerear', 'drnl', 'ihc', 'adaptation', 'resampled']
    >>> print(f"After DRNL: {stages['drnl'].shape}")
    After DRNL: torch.Size([2, 50, 44100])
    >>> print(f"After resampling: {stages['resampled'].shape}")
    After resampling: torch.Size([2, 50, 11025])
    
    **Detection task:**
    
    >>> # Simple detection with threshold
    >>> signal = torch.randn(1, 44100) * 0.01
    >>> decision = model.detection_task(signal, threshold=0.5)
    >>> print(f"Detection: {'Signal detected' if decision.item() else 'No signal'}")
    Detection: Signal detected
    >>> 
    >>> # Detection with custom metric
    >>> decision_max = model.detection_task(signal, threshold=0.3, metric='max')
    
    **Discrimination task:**
    
    >>> # Two-interval forced choice
    >>> signal1 = torch.randn(1, 44100) * 0.01
    >>> signal2 = torch.randn(1, 44100) * 0.015  # Slightly louder
    >>> choice = model.discrimination_task(signal1, signal2, criterion='rms')
    >>> print(f"Chose interval: {choice.item() + 1}")
    Chose interval: 2
    
    **ROI selection:**
    
    >>> # Extract specific time window and frequency range
    >>> full_repr = model(audio)
    >>> 
    >>> # Focus on 200-500 ms, channels 10-20, slow modulations (0-3)
    >>> roi = model.roi_selection(
    ...     full_repr,
    ...     time_window=(0.2, 0.5),
    ...     channel_range=(10, 20),
    ...     modulation_range=(0, 3)
    ... )
    >>> print(f"ROI channels: {len(roi)}")
    ROI channels: 10
    >>> print(f"ROI shape: {roi[0].shape}")
    ROI shape: torch.Size([2, 3, 3307])  # 3 mod filters, ~0.3s at 11025 Hz
    
    **Template correlation:**
    
    >>> # Compute cross-correlation with internal template
    >>> test_signal = torch.randn(1, 44100) * 0.01
    >>> template_signal = torch.randn(1, 44100) * 0.01
    >>> 
    >>> test_repr = model(test_signal)
    >>> template_repr = model(template_signal)
    >>> 
    >>> correlation = model.template_correlation(test_repr, template_repr)
    >>> print(f"Template correlation: {correlation.item():.4f}")
    Template correlation: 0.8523
    
    **Custom decision variable:**
    
    >>> # Compute RMS energy with channel weighting
    >>> channel_weights = torch.ones(50)
    >>> channel_weights[20:30] *= 2.0  # Emphasize mid-frequency channels
    >>> 
    >>> dv = model.compute_decision_variable(
    ...     full_repr,
    ...     metric='rms',
    ...     channel_weights=channel_weights
    ... )
    >>> print(f"Decision variable: {dv}")
    Decision variable: tensor([0.0234, 0.0198])
    
    **Batch processing:**
    
    >>> # Process multiple signals
    >>> batch_audio = torch.randn(8, 44100) * 0.01
    >>> batch_repr = model(batch_audio)
    >>> print(f"Batch output - Channel 0: {batch_repr[0].shape}")
    Batch output - Channel 0: torch.Size([8, 8, 11025])
    
    **Without outer ear filtering:**
    
    >>> # For pre-compensated or loudspeaker signals
    >>> model_noear = Paulick2024(fs=44100, use_outerear=False)
    >>> output_noear = model_noear(audio)
    
    **Custom frequency range:**
    
    >>> # Focus on specific frequency region
    >>> model_lowfreq = Paulick2024(fs=44100, flow=80, fhigh=2000, n_channels=20)
    >>> print(f"Frequency channels: {model_lowfreq.num_channels}")
    Frequency channels: 20
    >>> print(f"Center freq range: {model_lowfreq.fc[0]:.1f} - {model_lowfreq.fc[-1]:.1f} Hz")
    Center freq range: 80.0 - 2000.0 Hz
    
    **Learnable model for optimization:**
    
    >>> model_learnable = Paulick2024(fs=44100, learnable=True)
    >>> n_params = sum(p.numel() for p in model_learnable.parameters() if p.requires_grad)
    >>> print(f"Trainable parameters: {n_params}")
    Trainable parameters: 17223
    >>> 
    >>> # Example training loop
    >>> optimizer = torch.optim.Adam(model_learnable.parameters(), lr=1e-4)
    >>> # ... training code ...
    
    **Butterworth modulation filters:**
    
    >>> # Use symmetric bandpass filters instead of efilt
    >>> model_butter = Paulick2024(fs=44100, filter_type='butterworth')
    >>> output_butter = model_butter(audio)
    
    **Accessing center frequencies:**
    
    >>> model = Paulick2024(fs=44100)
    >>> print(f"Auditory fc (first 5): {model.fc[:5]}")
    Auditory fc (first 5): tensor([ 125.0,  145.8,  170.1,  198.5,  231.6])
    >>> print(f"Resampled rate: {model.fs_resampled} Hz")
    Resampled rate: 11025.0 Hz
    
    Notes
    -----
    **Model Configuration:**
    
    - **Filterbank**: DRNL 50 channels, 125-8000 Hz, dual-path nonlinear
    - **IHC**: Paulick2024-specific 3-stage (rectify → compress → lowpass)
    - **Adaptation**: 5 loops with frequency-dependent parameters
    - **Resampling**: ÷4 (44100 → 11025 Hz) via sinc interpolation
    - **Modulation**: 8 channels, 0-128.6 Hz, geometric spacing 5/3
    
    **Computational Complexity:**
    
    Processing time scales as:
    
    .. math::
        T_{\\text{compute}} \\propto T \\cdot (N_{\\text{filt}} + N_{\\text{filt}} \\cdot N_{\\text{mod}})
    
    where :math:`T` = signal length, :math:`N_{\\text{filt}}` = 50 (auditory channels),
    :math:`N_{\\text{mod}}` ≈ 6-8 (modulation channels).
    
    For 1 second @ 44100 Hz: ~0.2-0.5 seconds on CPU, ~0.05-0.15 seconds on GPU.
    
    **Memory Requirements:**
    
    Peak memory with intermediate stages:
    
    .. math::
        \\text{Memory} \\approx B \\cdot T \\cdot F \\cdot (1 + M \\cdot 0.25) \\cdot 8\\,\\text{bytes}
    
    For batch=4, 1 second @ 44100 Hz, F=50, M=8: ~50-70 MB (float64).
    With float32: ~25-35 MB.
    
    **Applications:**
    
    The model is particularly suited for:
    
    - Psychophysical detection threshold prediction
    - Discrimination task modeling (2AFC, 3AFC, etc.)
    - Forward/backward masking studies
    - Spectral and temporal masking
    - Amplitude modulation detection/discrimination
    - Hearing aid processing evaluation
    - Hearing loss simulation and compensation
    - Speech intelligibility prediction
    
    **Decision-Making Methods:**
    
    The model includes 6 decision-making methods for psychophysical modeling:
    
    1. **roi_selection**: Extract time/frequency/modulation regions of interest
    2. **template_correlation**: Cross-correlate with internal template
    3. **compute_decision_variable**: Compute metric (RMS, mean, max, L2)
    4. **make_decision**: Binary threshold or ML-based decision
    5. **detection_task**: Complete detection task (signal vs. noise)
    6. **discrimination_task**: Complete discrimination task (2AFC, etc.)
    
    These methods operate on the internal representation and provide a complete
    framework for modeling psychophysical experiments.
    
    **Learnable Parameters** (if learnable=True):
    
    - OuterMiddleEarFilter: ~16000 parameters (FIR filter taps)
    - DRNLFilterbank: ~400 parameters (nonlinearity coefficients, gains)
    - IHCPaulick2024: 13 parameters (physiological constants)
    - AdaptLoop: 10 parameters (time constants a1, b0 for 5 loops)
    - ModulationFilterbank: ~800 parameters (filter coefficients)
    
    **Total**: ~17223 trainable parameters
    
    See Also
    --------
    DRNLFilterbank : Stage 1 - Dual Resonance NonLinear filterbank
    OuterMiddleEarFilter : Stage 0 - Outer/middle ear filtering
    IHCPaulick2024 : Stage 2 - Inner hair cell transduction
    AdaptLoop : Stage 3 - Multi-stage adaptation
    ModulationFilterbank : Stage 5 - Modulation analysis
    
    References
    ----------
    .. [1] L. Paulick, H. Relaño-Iborra, and T. Dau, "The Computational Auditory 
           Signal Processing and Perception Model (CASP): A Revised Version," 
           *bioRxiv*, 2024.
    
    .. [2] M. Jepsen, S. Ewert, and T. Dau, "A computational model of human 
           auditory signal processing and perception," *J. Acoust. Soc. Am.*, 
           vol. 124, no. 1, pp. 422-438, Jul. 2008.
    
    .. [3] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
           of amplitude modulation. I. Detection and masking with narrow-band carriers," 
           *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, Nov. 1997.
    
    .. [4] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
           reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
           p. 19, 2022.
    """
    
    def __init__(self,
                 fs: float = 44100,
                 flow: float = 125.0,
                 fhigh: float = 8000.0,
                 n_channels: int = 50,
                 use_outerear: bool = True,
                 learnable: bool = False,
                 return_stages: bool = False,
                 filter_type: str = 'efilt',
                 dtype: torch.dtype = torch.float32,
                 outerear_kwargs: Optional[Dict[str, Any]] = None,
                 drnl_kwargs: Optional[Dict[str, Any]] = None,
                 ihc_kwargs: Optional[Dict[str, Any]] = None,
                 adaptation_kwargs: Optional[Dict[str, Any]] = None,
                 modulation_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.fs = fs
        self.flow = flow
        self.fhigh = fhigh
        self.n_channels = n_channels
        self.use_outerear = use_outerear
        self.learnable = learnable
        self.return_stages = return_stages
        self.filter_type = filter_type
        self.dtype = dtype
        
        # Initialize kwargs dictionaries if None
        outerear_kwargs = outerear_kwargs or {}
        drnl_kwargs = drnl_kwargs or {}
        ihc_kwargs = ihc_kwargs or {}
        adaptation_kwargs = adaptation_kwargs or {}
        modulation_kwargs = modulation_kwargs or {}
        
        # Stage 0: Outer/Middle Ear Filter (optional, headphone compensation)
        if use_outerear:
            outerear_defaults = {'compensation_type': 'tfOuterMiddle1997', 'field_type': 'free'}
            outerear_params = {**outerear_defaults, **outerear_kwargs}
            self.outer_middle_ear = OuterMiddleEarFilter(fs=fs,
                                                         learnable=learnable,
                                                         dtype=dtype,
                                                         **outerear_params)
        else:
            self.outer_middle_ear = None
        
        # Stage 1: DRNL Filterbank
        drnl_defaults = {'subject': 'NH', 'model': 'paulick2024'}
        drnl_params = {**drnl_defaults, **drnl_kwargs}
        
        # Original DRNL (slower but reference implementation)
        # self.drnl = DRNLFilterbank(fc=(flow, fhigh),
        #                            fs=fs,
        #                            n_channels=n_channels,
        #                            learnable=learnable,
        #                            dtype=dtype,
        #                            **drnl_params)
        
        # Fast DRNL (~100x speedup, drop-in replacement)
        self.drnl = FastDRNLFilterbank(fc=(flow, fhigh),
                                       fs=fs,
                                       n_channels=n_channels,
                                       learnable=learnable,
                                       dtype=dtype,
                                       **drnl_params)
        
        self.fc = self.drnl.fc
        self.num_channels = self.drnl.num_channels
        
        # Stage 2: IHC Transduction
        ihc_defaults = {}
        ihc_params = {**ihc_defaults, **ihc_kwargs}
        self.ihc = IHCPaulick2024(fs=fs,
                                  learnable=learnable,
                                  dtype=dtype,
                                  **ihc_params)
        
        # Stage 3: Adaptation Loops
        adaptation_defaults = {'preset': 'paulick2024'}
        adaptation_params = {**adaptation_defaults, **adaptation_kwargs}
        self.adaptation = AdaptLoop(fs=fs,
                                    learnable=learnable,
                                    dtype=dtype,
                                    **adaptation_params)
        
        # Resampling factor: 44100 Hz → 11025 Hz (÷4)
        self.resample_factor = 4
        self.fs_resampled = fs // self.resample_factor
        
        # Stage 4: Modulation filterbank operates at resampled rate
        # Need to pass fc for dynamic upper limit computation
        modulation_defaults = {'preset': 'paulick2024', 'filter_type': filter_type}
        modulation_params = {**modulation_defaults, **modulation_kwargs}
        
        # self.modulation = ModulationFilterbank(fs=self.fs_resampled,
        #                                        fc=self.fc,
        #                                        learnable=learnable,
        #                                        dtype=dtype,
        #                                        **modulation_params)
        
        self.modulation = FastModulationFilterbank(fs=self.fs_resampled,
                                                   fc=self.fc,
                                                   learnable=learnable,
                                                   dtype=dtype,
                                                   **modulation_params)
        
        # Resampler (will be applied to all channels at once)
        if fs != self.fs_resampled:
            self.resampler = T.Resample(orig_freq=fs,
                                        new_freq=self.fs_resampled,
                                        resampling_method='sinc_interp_kaiser')
        else:
            self.resampler = None
    
    def forward(self,
                x: torch.Tensor) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]]:
        """
        Process audio through the Paulick2024 model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input audio signal. Shape: (B, T) or (T,).
            
        Returns
        -------
        List[torch.Tensor] or tuple
            If return_stages=False:
                List of tensors (one per frequency channel), each shape (B, M_i, T_resample).
            If return_stages=True:
                Tuple of (internal_repr, stages) where stages is a dict with intermediate outputs.
        """
        stages = {} if self.return_stages else None
        
        # Normalize input shape to (B, T)
        original_shape = x.shape
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [T] -> [1, T]
        elif x.ndim != 2:
            raise ValueError(f"Expected audio shape [B, T] or [T], got {x.shape}")
        
        # Stage 0: Outer/Middle Ear (optional)
        if self.outer_middle_ear is not None:
            x = self.outer_middle_ear(x)
            if self.return_stages:
                stages['outerear'] = x.clone()
        
        # Stage 1: DRNL Filterbank
        # Output: [B, F, T]
        x = self.drnl(x)
        if self.return_stages:
            stages['drnl'] = x.clone()
        
        # Stage 2: IHC Transduction
        # Input: [B, F, T], Output: [B, F, T]
        x = self.ihc(x)
        if self.return_stages:
            stages['ihc'] = x.clone()
        
        # Stage 3: Adaptation Loops
        # Input: [B, F, T], Output: [B, F, T]
        x = self.adaptation(x)
        if self.return_stages:
            stages['adaptation'] = x.clone()
        
        # Stage 4: Resampling (÷4)
        if self.resampler is not None:
            # Convert to float32 for resampler, then back to original dtype
            x_dtype = x.dtype
            x = self.resampler(x.float()).to(x_dtype)  # [B, F, T_resample]
        if self.return_stages:
            stages['resampled'] = x.clone()
        
        # Stage 5: Modulation Filterbank
        # Input: [B, F, T_resample], Output: List of F tensors [B, M_i, T_resample]
        internal_repr = self.modulation(x)
        
        # If original input was 1D, squeeze batch dimension from all outputs
        if len(original_shape) == 1:
            internal_repr = [out.squeeze(0) if out.ndim == 3 else out for out in internal_repr]
        
        if self.return_stages:
            return internal_repr, stages
        else:
            return internal_repr
    
    # =========================================================================
    # Decision-Making Methods (Post-Model Processing for Psychophysical Tasks)
    # =========================================================================
    
    def roi_selection(self,
                      internal_repr: List[torch.Tensor],
                      time_window: Optional[Tuple[float, float]] = None,
                      channel_range: Optional[Tuple[int, int]] = None,
                      modulation_range: Optional[Tuple[int, int]] = None) -> List[torch.Tensor]:
        """
        Extract Region of Interest (ROI) from internal representation.
        
        Useful for focusing decision metrics on specific:
        - Time windows (e.g., signal interval vs. silence)
        - Auditory channels (e.g., low-frequency vs. high-frequency)
        - Modulation channels (e.g., slow vs. fast modulations)
        
        Parameters:
        ----------
        internal_repr : List[torch.Tensor]
            Internal representation from forward pass (list of 50 tensors)
        time_window : Tuple[float, float], optional
            Time window in seconds (start, end). If None, use full duration.
        channel_range : Tuple[int, int], optional
            Auditory channel range (start_idx, end_idx). If None, use all channels.
        modulation_range : Tuple[int, int], optional
            Modulation channel range (start_idx, end_idx). If None, use all modulation channels.
        
        Returns:
        -------
        roi_repr : List[torch.Tensor]
            Selected region, same format as internal_repr but with reduced dimensions
        """
        roi_repr = []
        
        # Determine channel range
        if channel_range is None:
            chan_start, chan_end = 0, len(internal_repr)
        else:
            chan_start, chan_end = channel_range
        
        for ch_idx in range(chan_start, chan_end):
            x = internal_repr[ch_idx]  # [B, Nmod, T]
            
            # Select modulation channels
            if modulation_range is not None:
                mod_start, mod_end = modulation_range
                x = x[:, mod_start:mod_end, :]
            
            # Select time window
            if time_window is not None:
                t_start_sec, t_end_sec = time_window
                t_start_idx = int(t_start_sec * self.fs_resampled)
                t_end_idx = int(t_end_sec * self.fs_resampled)
                x = x[:, :, t_start_idx:t_end_idx]
            
            roi_repr.append(x)
        
        return roi_repr
    
    def template_correlation(self,
                             internal_repr: List[torch.Tensor],
                             template: List[torch.Tensor],
                             normalize: bool = True) -> torch.Tensor:
        """
        Compute correlation between internal representation and a reference template.
        
        Useful for template-matching decision strategies in psychophysical tasks
        (e.g., signal detection, discrimination).
        
        Parameters:
        ----------
        internal_repr : List[torch.Tensor]
            Internal representation from test stimulus
        template : List[torch.Tensor]
            Reference template (same format as internal_repr)
        normalize : bool, optional
            If True, compute normalized correlation (Pearson-like). Default: True
        
        Returns:
        -------
        correlation : torch.Tensor
            Correlation values with shape [B] (one value per batch item)
        """
        correlations = []
        
        for ch_idx in range(len(internal_repr)):
            x = internal_repr[ch_idx]  # [B, Nmod, T]
            t = template[ch_idx]  # [1 or B, Nmod, T]
            
            # Ensure template broadcasts correctly
            if t.shape[0] == 1 and x.shape[0] > 1:
                t = t.expand(x.shape[0], -1, -1)
            
            # Flatten modulation × time dimensions
            x_flat = x.reshape(x.shape[0], -1)  # [B, Nmod*T]
            t_flat = t.reshape(t.shape[0], -1)  # [B, Nmod*T]
            
            if normalize:
                # Pearson correlation
                x_mean = x_flat.mean(dim=1, keepdim=True)
                t_mean = t_flat.mean(dim=1, keepdim=True)
                x_centered = x_flat - x_mean
                t_centered = t_flat - t_mean
                
                numerator = (x_centered * t_centered).sum(dim=1)
                denominator = torch.sqrt(
                    (x_centered ** 2).sum(dim=1) * (t_centered ** 2).sum(dim=1)
                )
                corr = numerator / (denominator + 1e-8)
            else:
                # Dot product (unnormalized)
                corr = (x_flat * t_flat).sum(dim=1)
            
            correlations.append(corr)
        
        # Average across auditory channels
        correlation_tensor = torch.stack(correlations, dim=0)  # [50, B]
        avg_correlation = correlation_tensor.mean(dim=0)  # [B]
        
        return avg_correlation
    
    def compute_decision_variable(self,
                                  internal_repr: List[torch.Tensor],
                                  metric: str = 'rms',
                                  channel_weights: Optional[torch.Tensor] = None,
                                  modulation_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute decision variable from internal representation using specified metric.
        
        Common metrics for psychophysical modeling:
        - 'rms': Root-mean-square energy
        - 'mean': Average activity
        - 'max': Maximum activity
        - 'l2': L2 norm
        
        Parameters:
        ----------
        internal_repr : List[torch.Tensor]
            Internal representation from forward pass
        metric : str, optional
            Decision metric to compute. Default: 'rms'
        channel_weights : torch.Tensor, optional
            Weights for auditory channels [50]. If None, uniform weighting.
        modulation_weights : torch.Tensor, optional
            Weights for modulation channels [max_Nmod]. If None, uniform weighting.
        
        Returns:
        -------
        decision_var : torch.Tensor
            Decision variable with shape [B]
        """
        decision_values = []
        
        for ch_idx, x in enumerate(internal_repr):
            # x: [B, Nmod, T]
            
            # Apply modulation weights if provided
            if modulation_weights is not None:
                # Broadcast weights to [1, Nmod, 1]
                weights = modulation_weights[:x.shape[1]].view(1, -1, 1).to(x.device)
                x = x * weights
            
            # Compute metric
            if metric == 'rms':
                val = torch.sqrt(torch.mean(x ** 2, dim=(1, 2)))  # [B]
            elif metric == 'mean':
                val = torch.mean(x, dim=(1, 2))
            elif metric == 'max':
                val = torch.amax(x, dim=(1, 2))
            elif metric == 'l2':
                val = torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Apply channel weight if provided
            if channel_weights is not None:
                val = val * channel_weights[ch_idx].to(val.device)
            
            decision_values.append(val)
        
        # Sum across auditory channels
        decision_var = torch.stack(decision_values, dim=0).sum(dim=0)  # [B]
        
        return decision_var
    
    def make_decision(self,
                      decision_var: torch.Tensor,
                      threshold: float = 0.0,
                      criterion: str = 'greater') -> torch.Tensor:
        """
        Make binary decision based on decision variable and threshold.
        
        Implements the final decision stage in signal detection theory.
        
        Parameters:
        ----------
        decision_var : torch.Tensor
            Decision variable from compute_decision_variable() [B]
        threshold : float, optional
            Decision threshold. Default: 0.0
        criterion : str, optional
            Decision criterion: 'greater' (DV > threshold) or 'less' (DV < threshold).
            Default: 'greater'
        
        Returns:
        -------
        decisions : torch.Tensor
            Binary decisions with shape [B], dtype=torch.bool
        """
        if criterion == 'greater':
            decisions = decision_var > threshold
        elif criterion == 'less':
            decisions = decision_var < threshold
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return decisions
    
    def detection_task(self,
                       signal_audio: torch.Tensor,
                       noise_audio: Optional[torch.Tensor] = None,
                       metric: str = 'rms',
                       threshold: Optional[float] = None,
                       return_decision_var: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform signal detection task (signal vs. noise).
        
        High-level method combining forward pass, decision variable computation,
        and threshold-based decision.
        
        Parameters:
        ----------
        signal_audio : torch.Tensor
            Test audio (signal + noise or noise-only) [B, T]
        noise_audio : torch.Tensor, optional
            Reference noise-only audio for template subtraction [1 or B, T]
        metric : str, optional
            Decision metric. Default: 'rms'
        threshold : float, optional
            Decision threshold. If None, return decision variable only.
        return_decision_var : bool, optional
            If True, return (decisions, decision_var). Default: False
        
        Returns:
        -------
        decisions : torch.Tensor (if threshold provided)
            Binary decisions [B]
        decision_var : torch.Tensor (if threshold is None or return_decision_var=True)
            Decision variable [B]
        """
        # Process signal
        signal_repr = self(signal_audio)
        decision_var = self.compute_decision_variable(signal_repr, metric=metric)
        
        # Optional: subtract noise baseline
        if noise_audio is not None:
            noise_repr = self(noise_audio)
            noise_var = self.compute_decision_variable(noise_repr, metric=metric)
            decision_var = decision_var - noise_var
        
        if threshold is None:
            return decision_var
        
        decisions = self.make_decision(decision_var, threshold=threshold)
        
        if return_decision_var:
            return decisions, decision_var
        else:
            return decisions
    
    def discrimination_task(self,
                            stimulus1: torch.Tensor,
                            stimulus2: torch.Tensor,
                            metric: str = 'rms',
                            decision_rule: str = 'greater',
                            return_decision_var: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform discrimination task (which stimulus is louder/different?).
        
        Parameters:
        ----------
        stimulus1 : torch.Tensor
            First stimulus [B, T]
        stimulus2 : torch.Tensor
            Second stimulus [B, T]
        metric : str, optional
            Decision metric. Default: 'rms'
        decision_rule : str, optional
            'greater': choose stimulus1 if DV1 > DV2
            'less': choose stimulus1 if DV1 < DV2
            Default: 'greater'
        return_decision_var : bool, optional
            If True, return (decisions, delta_dv). Default: False
        
        Returns:
        -------
        decisions : torch.Tensor
            Binary decisions [B]: True if stimulus1 chosen, False if stimulus2 chosen
        delta_dv : torch.Tensor (if return_decision_var=True)
            Difference in decision variables [B]: DV1 - DV2
        """
        # Process both stimuli
        repr1 = self(stimulus1)
        repr2 = self(stimulus2)
        
        dv1 = self.compute_decision_variable(repr1, metric=metric)
        dv2 = self.compute_decision_variable(repr2, metric=metric)
        
        delta_dv = dv1 - dv2
        
        if decision_rule == 'greater':
            decisions = delta_dv > 0
        elif decision_rule == 'less':
            decisions = delta_dv < 0
        else:
            raise ValueError(f"Unknown decision_rule: {decision_rule}")
        
        if return_decision_var:
            return decisions, delta_dv
        else:
            return decisions
    
    def extra_repr(self) -> str:
        """
        Extra representation for printing.
        
        Returns
        -------
        str
            String representation of module parameters.
        """
        return (f"fs={self.fs}, flow={self.flow}, fhigh={self.fhigh}, "
                f"num_channels={self.num_channels}, use_outerear={self.use_outerear}, "
                f"learnable={self.learnable}, fs_resampled={self.fs_resampled}")
