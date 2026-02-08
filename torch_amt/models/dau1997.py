"""
Dau1997 Auditory Model for Monaural Masking
===========================================

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the Dau et al. (1997) model for auditory processing 
of amplitude modulation. The model consists of a 4-stage pipeline simulating 
peripheral auditory processing from the basilar membrane to modulation detection.

The implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT) 
and extended with PyTorch for gradient-based optimization and GPU acceleration.

References
----------
.. [1] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
       of amplitude modulation. I. Detection and masking with narrow-band carriers," 
       *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, 1997.

.. [2] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
       of amplitude modulation. II. Spectral and temporal integration," 
       *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2906-2919, 1997.

.. [3] T. Dau, D. Püschel, and A. Kohlrausch, "A quantitative model of the 
       'effective' signal processing in the auditory system. I. Model structure," 
       *J. Acoust. Soc. Am.*, vol. 99, no. 6, pp. 3615-3622, 1996.

.. [4] R. D. Patterson, I. Nimmo-Smith, J. Holdsworth, and P. Rice, "An efficient 
       auditory filterbank based on the gammatone function," *APU Report 2341*, 
       MRC Applied Psychology Unit, Cambridge, UK, 1988.

.. [5] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
       p. 19, 2022.
"""

from typing import Dict, Any, List

import torch
import torch.nn as nn

from torch_amt.common.filterbanks import GammatoneFilterbank
from torch_amt.common.ihc import IHCEnvelope
from torch_amt.common.adaptation import AdaptLoop
# from torch_amt.common.modulation import ModulationFilterbank
from torch_amt.common.modulation import FastModulationFilterbank


class Dau1997(nn.Module):
    r"""
    Dau et al. (1997) auditory model for amplitude modulation processing.
    
    Implements the complete auditory processing pipeline from Dau, Kollmeier, and 
    Kohlrausch (1997) for modeling detection and masking with amplitude-modulated 
    signals. The model simulates peripheral auditory processing from cochlear filtering 
    through neural adaptation to modulation detection.
    
    This implementation is based on the MATLAB Auditory Modeling Toolbox (AMT) 
    ``dau1997`` function and provides a differentiable, GPU-accelerated version 
    suitable for neural network training and optimization.
    
    Algorithm Overview
    ------------------
    The model implements a 4-stage auditory processing pipeline:
    
    **Stage 1: Gammatone Filterbank**
    
    Decomposes the input signal into frequency channels using 4th-order gammatone 
    filters with 1-ERB spacing:
    
    .. math::
        x_f(t) = \text{Gammatone}_{f_c}(x(t))
    
    where :math:`f_c` are the center frequencies from ``flow`` to ``fhigh``.
    
    **Stage 2: Inner Hair Cell (IHC) Envelope Extraction**
    
    Extracts the envelope via half-wave rectification followed by 1000 Hz lowpass 
    filtering (Dau 1996 method):
    
    .. math::
        e_f(t) = \text{Lowpass}_{1000\,\text{Hz}}(\max(0, x_f(t)))
    
    **Stage 3: Adaptation Loops**
    
    Models neural adaptation using 5 cascaded feedback loops with time constants 
    :math:`\tau = [5, 50, 129, 253, 500]` ms and overshoot limit :math:`L=10`:
    
    .. math::
        a_f(t) = \text{AdaptLoop}(e_f(t), \tau, L)
    
    **Stage 4: Modulation Filterbank**
    
    Analyzes temporal modulations using bandpass filters with :math:`Q=2` up to 
    150 Hz modulation frequency:
    
    .. math::
        m_{f,j}(t) = \text{ModFilter}_{f_{mod,j}}(a_f(t))
    
    where :math:`j` indexes modulation channels (frequency-dependent).
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz. Typical values: 16000, 32000, 44100.
        Higher sampling rates provide better temporal resolution but increase 
        computational cost.
    
    flow : float, optional
        Lower frequency bound for gammatone filterbank in Hz. Default: 80 Hz.
        Lower values extend low-frequency coverage but increase filter count.
        Typical range: [50, 200] Hz.
    
    fhigh : float, optional
        Upper frequency bound for gammatone filterbank in Hz. Default: 8000 Hz.
        Higher values extend high-frequency coverage but increase filter count.
        Must be less than Nyquist frequency (fs/2). Typical range: [4000, 16000] Hz.
    
    learnable : bool, optional
        If True, all model stages (filterbank, IHC, adaptation, modulation) become 
        trainable with gradient-based optimization. Default: False (fixed parameters).
        When True, enables end-to-end model training for task-specific optimization.
    
    return_stages : bool, optional
        If True, returns intermediate processing stages along with final output. 
        Default: False (only final modulation representation).
        Useful for visualization, analysis, and multi-stage training.
    
    dtype : torch.dtype, optional
        Data type for computations and parameters. Default: torch.float32.
        Use torch.float64 for higher precision if needed (increases memory/computation).
    
    **filterbank_kwargs : dict, optional
        Additional keyword arguments passed to :class:`GammatoneFilterbank`.
        Common options:
        
        - ``n`` (int): Filter order. Default: 4.
        - ``erb_scale`` (float): ERB scale factor. Default: 1.0.
        - Other parameters accepted by GammatoneFilterbank.
    
    **ihc_kwargs : dict, optional
        Additional keyword arguments passed to :class:`IHCEnvelope`.
        Common options:
        
        - ``method`` (str): IHC method. Default: 'dau1996'.
        - Other parameters accepted by IHCEnvelope.
    
    **adaptation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`AdaptLoop`.
        Common options:
        
        - ``tau`` (list or torch.Tensor): Time constants in seconds. 
          Default: None (uses [0.005, 0.050, 0.129, 0.253, 0.500]).
        - ``limit`` (float): Overshoot limit factor. Default: 10.0.
        - ``minspl`` (float): Minimum SPL in dB. Default: 0.0.
        - Other parameters accepted by AdaptLoop.
    
    **modulation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`ModulationFilterbank`.
        Common options:
        
        - ``Q`` (float): Quality factor for modulation filters. Default: 2.0.
        - ``max_mfc`` (float): Maximum modulation frequency in Hz. Default: 150.0.
        - ``filter_type`` (str): Filter type ('efilt' or 'butterworth'). Default: 'efilt'.
        - Other parameters accepted by ModulationFilterbank.
    
    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    
    flow : float
        Lower frequency bound in Hz.
    
    fhigh : float
        Upper frequency bound in Hz.
    
    learnable : bool
        Whether model parameters are trainable.
    
    return_stages : bool
        Whether to return intermediate processing stages.
    
    dtype : torch.dtype
        Data type for computations.
    
    filterbank : GammatoneFilterbank
        Stage 1: Gammatone auditory filterbank module.
    
    ihc : IHCEnvelope
        Stage 2: Inner hair cell envelope extraction module.
    
    adaptation : AdaptLoop
        Stage 3: Neural adaptation loops module.
    
    modulation : ModulationFilterbank
        Stage 4: Temporal modulation filterbank module.
    
    fc : torch.Tensor
        Center frequencies of auditory filterbank, shape (num_channels,) in Hz.
        Computed with 1-ERB spacing between flow and fhigh.
    
    num_channels : int
        Number of auditory frequency channels in the filterbank.
        Typically 20-40 channels for default frequency range (80-8000 Hz).
        
    Input Shape
    -----------
    x : torch.Tensor
        Audio signal with shape:
        
        - :math:`(B, T)` - Batch of audio samples
        - :math:`(T,)` - Single audio sample (mono)
        
        where:
        
        - :math:`B` = batch size
        - :math:`T` = time samples
    
    Output Shape
    ------------
    When ``return_stages=False`` (default):
        List[torch.Tensor]
            List of length ``num_channels`` (one per frequency channel).
            Each element has shape :math:`(B, M_i, T)` where:
            
            - :math:`M_i` = number of modulation channels for frequency channel :math:`i`
            - Varies per channel: low frequencies have fewer modulation channels
            
    When ``return_stages=True``:
        Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]
            - First element: internal representation (as above)
            - Second element: dict with keys ['filterbank', 'ihc', 'adaptation']
              containing intermediate outputs, each shape :math:`(B, F, T)` where 
              :math:`F` = ``num_channels``
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> from torch_amt.models import Dau1997
    >>> 
    >>> # Create model
    >>> model = Dau1997(fs=44100, flow=80, fhigh=8000)
    >>> print(f"Frequency channels: {model.num_channels}")
    Frequency channels: 31
    >>> 
    >>> # Process audio (1 second, 44.1 kHz)
    >>> audio = torch.randn(2, 44100)  # 2 batches
    >>> output = model(audio)
    >>> print(f"Output: {len(output)} frequency channels")
    Output: 31 frequency channels
    >>> print(f"First channel shape: {output[0].shape}")
    First channel shape: torch.Size([2, 13, 44100])
    >>> print(f"Last channel shape: {output[-1].shape}")
    Last channel shape: torch.Size([2, 8, 44100])
    
    **With intermediate stages for visualization:**
    
    >>> model_debug = Dau1997(fs=44100, return_stages=True)
    >>> output, stages = model_debug(audio)
    >>> 
    >>> print(f"Available stages: {list(stages.keys())}")
    Available stages: ['filterbank', 'ihc', 'adaptation']
    >>> print(f"Filterbank output: {stages['filterbank'].shape}")
    Filterbank output: torch.Size([2, 31, 44100])
    >>> print(f"IHC output: {stages['ihc'].shape}")
    IHC output: torch.Size([2, 31, 44100])
    >>> print(f"Adaptation output: {stages['adaptation'].shape}")
    Adaptation output: torch.Size([2, 31, 44100])
    
    **Single channel input (mono):**
    
    >>> audio_mono = torch.randn(44100)  # No batch dimension
    >>> output_mono = model(audio_mono)
    >>> print(f"Output shape (mono): {output_mono[0].shape}")
    Output shape (mono): torch.Size([13, 44100])
    
    **Learnable model for optimization:**
    
    >>> model_learnable = Dau1997(fs=44100, learnable=True)
    >>> n_params = sum(p.numel() for p in model_learnable.parameters())
    >>> print(f"Trainable parameters: {n_params}")
    Trainable parameters: 15234
    >>> 
    >>> # Example training loop
    >>> optimizer = torch.optim.Adam(model_learnable.parameters(), lr=1e-3)
    >>> # ... training code ...
    
    **Different frequency ranges:**
    
    >>> # Extended low frequency
    >>> model_low = Dau1997(fs=44100, flow=50, fhigh=8000)
    >>> print(f"Channels with flow=50Hz: {model_low.num_channels}")
    Channels with flow=50Hz: 35
    >>> 
    >>> # Extended high frequency
    >>> model_high = Dau1997(fs=44100, flow=80, fhigh=16000)
    >>> print(f"Channels with fhigh=16kHz: {model_high.num_channels}")
    Channels with fhigh=16kHz: 39
    
    **Custom submodule parameters via kwargs:**
    
    >>> # Custom filterbank: 6th-order filters instead of default 4th-order
    >>> model_custom_fb = Dau1997(
    ...     fs=44100, 
    ...     filterbank_kwargs={'n': 6}
    ... )
    >>> 
    >>> # Custom adaptation: different overshoot limit and time constants
    >>> model_custom_adapt = Dau1997(
    ...     fs=44100,
    ...     adaptation_kwargs={
    ...         'limit': 5.0,  # Lower overshoot limit
    ...         'tau': [0.01, 0.1, 0.2, 0.4, 0.8]  # Custom time constants
    ...     }
    ... )
    >>> 
    >>> # Custom modulation: higher Q and different max frequency
    >>> model_custom_mod = Dau1997(
    ...     fs=44100,
    ...     modulation_kwargs={
    ...         'Q': 1.0,  # Lower Q (broader filters)
    ...         'max_mfc': 200.0,  # Extend to 200 Hz
    ...         'filter_type': 'butterworth'  # Use Butterworth instead of efilt
    ...     }
    ... )
    >>> 
    >>> # Combine multiple custom parameters
    >>> model_fully_custom = Dau1997(
    ...     fs=44100,
    ...     flow=50,
    ...     fhigh=12000,
    ...     filterbank_kwargs={'n': 5, 'erb_scale': 1.2},
    ...     ihc_kwargs={'method': 'breebaart2001'},
    ...     adaptation_kwargs={'limit': 8.0, 'minspl': -10.0},
    ...     modulation_kwargs={'Q': 1.5, 'max_mfc': 180.0}
    ... )
    >>> print(f"Fully custom model channels: {model_fully_custom.num_channels}")
    Fully custom model channels: 43
    
    Notes
    -----
    **Model Configuration:**
    
    The Dau1997 model uses specific configurations for each processing stage:
    
    - **Gammatone filterbank**: 4th-order filters with 1-ERB spacing
    - **IHC envelope**: Dau1996 method (half-wave rectification + 1 kHz lowpass)
    - **Adaptation**: 5 loops with :math:`\\tau = [5, 50, 129, 253, 500]` ms, 
      overshoot limit :math:`L=10`, :math:`\\text{minspl}=0` dB
    - **Modulation filterbank**: :math:`Q=2`, max modulation frequency 150 Hz
    
    **Customizing Submodule Parameters:**
    
    All submodules can be customized through dedicated kwargs dictionaries:
    
    - Use ``filterbank_kwargs`` to pass parameters to :class:`GammatoneFilterbank`
    - Use ``ihc_kwargs`` to pass parameters to :class:`IHCEnvelope`
    - Use ``adaptation_kwargs`` to pass parameters to :class:`AdaptLoop`
    - Use ``modulation_kwargs`` to pass parameters to :class:`ModulationFilterbank`
    
    The ``learnable`` and ``dtype`` parameters are always centralized and applied 
    to all submodules automatically. Custom parameters override defaults while 
    maintaining the Dau1997 model structure.
    
    Example: To use 6th-order gammatone filters with a different adaptation limit:
    
    .. code-block:: python
    
        model = Dau1997(
            fs=44100,
            filterbank_kwargs={'n': 6},
            adaptation_kwargs={'limit': 5.0}
        )
    
    **Computational Complexity:**
    
    Processing time scales approximately as:
    
    .. math::
        T_{compute} \\propto B \\cdot F \\cdot T \\cdot (1 + M)
    
    where :math:`F` = num_channels (20-40), :math:`M` = avg modulation channels (~10).
    For 1 second at 44.1 kHz: ~0.1-0.5 seconds on CPU, ~0.01-0.05 seconds on GPU.
    
    **Memory Requirements:**
    
    Peak memory scales with output representation size:
    
    .. math::
        Memory \\approx B \\cdot F \\cdot M \\cdot T \\cdot 4\\,\\text{bytes}
    
    For batch=8, 1 second @ 44.1 kHz: ~50-100 MB.
    
    **Applications:**
    
    The model internal representation can be used for:
    
    - Amplitude modulation detection and discrimination
    - Monaural masking predictions
    - Temporal modulation transfer function (TMTF) modeling
    - Feature extraction for machine learning tasks
    - Psychoacoustic model evaluation and fitting
    
    See Also
    --------
    GammatoneFilterbank : Stage 1 - Cochlear filtering
    IHCEnvelope : Stage 2 - Inner hair cell transduction
    AdaptLoop : Stage 3 - Neural adaptation
    ModulationFilterbank : Stage 4 - Modulation detection
    
    References
    ----------
    .. [1] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
           of amplitude modulation. I. Detection and masking with narrow-band carriers," 
           *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, Nov. 1997.
    
    .. [2] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
           of amplitude modulation. II. Spectral and temporal integration," 
           *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2906-2919, Nov. 1997.
    
    .. [3] T. Dau, D. Püschel, and A. Kohlrausch, "A quantitative model of the 
           'effective' signal processing in the auditory system. I. Model structure," 
           *J. Acoust. Soc. Am.*, vol. 99, no. 6, pp. 3615-3622, June 1996.
    
    .. [4] R. D. Patterson, I. Nimmo-Smith, J. Holdsworth, and P. Rice, "An efficient 
           auditory filterbank based on the gammatone function," *APU Report 2341*, 
           MRC Applied Psychology Unit, Cambridge, UK, 1988.
    
    .. [5] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
           reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
           p. 19, 2022.
    """
    
    def __init__(self,
                 fs: float,
                 flow: float = 80.0,
                 fhigh: float = 8000.0,
                 learnable: bool = False,
                 return_stages: bool = False,
                 dtype: torch.dtype = torch.float32,
                 filterbank_kwargs: Dict[str, Any] = None,
                 ihc_kwargs: Dict[str, Any] = None,
                 adaptation_kwargs: Dict[str, Any] = None,
                 modulation_kwargs: Dict[str, Any] = None):
        super().__init__()
        
        self.fs = fs
        self.flow = flow
        self.fhigh = fhigh
        self.learnable = learnable
        self.return_stages = return_stages
        self.dtype = dtype
        
        # Initialize kwargs dictionaries if None
        filterbank_kwargs = filterbank_kwargs or {}
        ihc_kwargs = ihc_kwargs or {}
        adaptation_kwargs = adaptation_kwargs or {}
        modulation_kwargs = modulation_kwargs or {}
        
        # Stage 1: Gammatone filterbank
        # Default: 4th-order filters with 1-ERB spacing
        filterbank_defaults = {'n': 4}
        filterbank_params = {**filterbank_defaults, **filterbank_kwargs}
        self.filterbank = GammatoneFilterbank(fc=(flow, fhigh), 
                                              fs=fs, 
                                              learnable=learnable, 
                                              dtype=dtype,
                                              **filterbank_params)
        self.fc = self.filterbank.fc
        self.num_channels = self.filterbank.num_channels
        
        # Stage 2: Inner hair cell envelope extraction
        # Default: Dau 1996 method
        ihc_defaults = {'method': 'dau1996'}
        ihc_params = {**ihc_defaults, **ihc_kwargs}
        self.ihc = IHCEnvelope(fs=fs, 
                               learnable=learnable, 
                               dtype=dtype,
                               **ihc_params)
        
        # Stage 3: Adaptation loops
        # Default: Dau 1997 configuration (5 loops, limit=10, minspl=0)
        adaptation_defaults = {'tau': None, 'limit': 10.0, 'minspl': 0.0}
        adaptation_params = {**adaptation_defaults, **adaptation_kwargs}
        self.adaptation = AdaptLoop(fs=fs, 
                                    learnable=learnable, 
                                    dtype=dtype,
                                    **adaptation_params)
        
        # Stage 4: Modulation filterbank
        # Default: Q=2, max modulation frequency 150 Hz
        modulation_defaults = {'Q': 2.0, 'max_mfc': 150.0}
        modulation_params = {**modulation_defaults, **modulation_kwargs}
        # self.modulation = ModulationFilterbank(fs=fs, 
        #                                        fc=self.fc, 
        #                                        learnable=learnable, 
        #                                        dtype=dtype,
        #                                        **modulation_params)
        self.modulation = FastModulationFilterbank(fs=fs, 
                                                   fc=self.fc, 
                                                   learnable=learnable, 
                                                   dtype=dtype,
                                                   **modulation_params)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor] | tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process audio through the Dau1997 model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input audio signal. Shape: (B, T), (C, T), or (T,).
            
        Returns
        -------
        List[torch.Tensor] or tuple
            If return_stages=False:
                List of tensors (one per frequency channel), each shape (B, M, T).
            If return_stages=True:
                Tuple of (output, stages) where stages is a dict with intermediate outputs.
        """
        stages = {} if self.return_stages else None
        
        # Normalize input shape to (B, T)
        original_shape = x.shape
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [T] -> [1, T]
        
        # Stage 1: Gammatone filterbank
        # Output: [B, F, T]
        x = self.filterbank(x)
        if self.return_stages:
            stages['filterbank'] = x.clone()
        
        # Stage 2: IHC envelope extraction
        # Input: [B, F, T], Output: [B, F, T]
        x = self.ihc(x)
        if self.return_stages:
            stages['ihc'] = x.clone()
        
        # Stage 3: Adaptation
        # Input: [B, F, T], Output: [B, F, T]
        x = self.adaptation(x)
        if self.return_stages:
            stages['adaptation'] = x.clone()
        
        # Stage 4: Modulation filterbank
        # Input: [B, F, T], Output: List of [B, M_i, T]
        output = self.modulation(x)
        
        # If original input was 1D, squeeze batch dimension from all outputs
        if len(original_shape) == 1:
            output = [out.squeeze(0) if out.ndim == 3 else out for out in output]
        
        if self.return_stages:
            return output, stages
        else:
            return output
    
    def extra_repr(self) -> str:
        """
        Extra representation for printing.
        
        Returns
        -------
        str
            String representation of module parameters.
        """
        return (f"fs={self.fs}, flow={self.flow}, fhigh={self.fhigh}, "
                f"num_channels={self.num_channels}, learnable={self.learnable}")
    
    def distribute_gradients(self):
        """
        Distribute gradients to grouped filter coefficients in FastModulationFilterbank.
        
        Call this method after ``loss.backward()`` to ensure all filter coefficients
        in the modulation filterbank receive gradient updates. This is necessary when
        using ``FastModulationFilterbank`` with ``learnable=True``, as filters are
        grouped for efficiency and gradients need to be shared across group members.
        
        Notes
        -----
        This method should be called in the training loop:
        
        >>> model = Dau1997(fs=44100, learnable=True)
        >>> output = model(input_signal)
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> model.distribute_gradients()  # ← Important for FastModulationFilterbank!
        >>> optimizer.step()
        
        If the modulation filterbank doesn't have a ``distribute_gradients`` method
        (e.g., using standard ModulationFilterbank), this is a no-op.
        """
        if hasattr(self.modulation, 'distribute_gradients'):
            self.modulation.distribute_gradients()
