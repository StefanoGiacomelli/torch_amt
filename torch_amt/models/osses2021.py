"""
Osses2021 Auditory Model w. Peripheral Filtering
================================================

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the Osses et al. (2021) auditory model with realistic
peripheral filtering for fluctuation strength prediction. The model extends
the Dau1997 framework by incorporating headphone and middle ear transfer
functions, providing more accurate peripheral modeling for psychoacoustic
predictions.

The implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT)
and extended with PyTorch for gradient-based optimization and GPU acceleration.

References
----------
.. [1] A. Osses Vecchi, L. E. García, and A. Kohlrausch, "Modelling the 
       sensation of fluctuation strength," in *Proc. Forum Acusticum 2020*, 
       Lyon, France, 2020, pp. 367-371.

.. [2] D. Pralong and S. Carlile, "The role of individualized headphone 
       calibration for the generation of high fidelity virtual auditory space," 
       *J. Acoust. Soc. Am.*, vol. 100, no. 6, pp. 3785-3793, Dec. 1996.

.. [3] M. L. Jepsen, S. D. Ewert, and T. Dau, "A computational model of 
       human auditory signal processing and perception," *J. Acoust. Soc. Am.*, 
       vol. 124, no. 1, pp. 422-438, Jul. 2008.

.. [4] J. Breebaart, S. van de Par, and A. Kohlrausch, "Binaural processing 
       model based on contralateral inhibition. I. Model structure," 
       *J. Acoust. Soc. Am.*, vol. 110, no. 2, pp. 1074-1088, Aug. 2001.

.. [5] T. Dau, D. Püschel, and A. Kohlrausch, "A quantitative model of the 
       'effective' signal processing in the auditory system. I. Model structure," 
       *J. Acoust. Soc. Am.*, vol. 99, no. 6, pp. 3615-3622, Jun. 1996.

.. [6] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
       p. 19, 2022.
"""

from typing import Dict, Any, List

import torch
import torch.nn as nn

from torch_amt.common.ears import HeadphoneFilter, MiddleEarFilter
from torch_amt.common.filterbanks import GammatoneFilterbank
from torch_amt.common.ihc import IHCEnvelope
from torch_amt.common.adaptation import AdaptLoop
# from torch_amt.common.modulation import ModulationFilterbank 
from torch_amt.common.modulation import FastModulationFilterbank


class Osses2021(nn.Module):
    r"""
    Osses et al. (2021) auditory model with realistic peripheral filtering.
    
    Implements a computational model of the auditory periphery designed for
    fluctuation strength prediction and other psychoacoustic tasks. The model
    extends the Dau1997 framework by incorporating headphone and middle ear
    transfer functions, providing more accurate peripheral modeling suitable
    for headphone-presented stimuli and perceptual predictions.
    
    This implementation follows the MATLAB Auditory Modeling Toolbox (AMT)
    osses2021 configuration and provides a differentiable, GPU-accelerated
    version suitable for neural network training and optimization.
    
    Algorithm Overview
    ------------------
    The model implements a 6-stage auditory processing pipeline:
    
    **Stage 0a: Headphone Filter**
    
    Compensates for headphone and outer ear characteristics using Pralong & Carlile (1996):
    
    .. math::
        x_{hp}(t) = h_{hp}(t) * x(t)
    
    where :math:`h_{hp}` models outer ear + headphone frequency response.
    
    **Stage 0b: Middle Ear Filter**
    
    Simulates middle ear transmission using Jepsen et al. (2008) model:
    
    .. math::
        x_{me}(t) = h_{me}(t) * x_{hp}(t)
    
    Bandpass characteristic approximating middle ear transfer function.
    
    **Stage 1: Gammatone Filterbank**
    
    Decomposes signal into :math:`N` frequency channels with 1-ERB spacing:
    
    .. math::
        y_i(t) = g_i(t) * x_{me}(t), \\quad i=1,\\ldots,N
    
    where :math:`g_i(t) = t^{n-1} e^{-2\\pi b t} \\cos(2\\pi f_c t + \\phi)`,
    typically :math:`n=4` (4th-order).
    
    **Stage 2: Inner Hair Cell (IHC) Envelope**
    
    Extracts envelope via Breebaart et al. (2001) method:
    
    .. math::
        e_i(t) = |y_i(t)|^2 * h_{lp}(t)
    
    where :math:`h_{lp}` is a lowpass filter extracting the envelope.
    
    **Stage 3: Adaptation**
    
    Five parallel adaptation loops with time constants :math:`\\tau_j`:
    
    .. math::
        v_i(t) = \\sum_{j=1}^5 a_j v_{i,j}(t)
    
    where each :math:`v_{i,j}` follows:
    
    .. math::
        \\frac{dv_{i,j}}{dt} = \\frac{e_i(t) - v_{i,j}(t)}{\\tau_j}
    
    Osses2021 preset uses :math:`\\text{limit}=5.0` (vs 10.0 in Dau1997).
    
    **Stage 4: Modulation Filterbank**
    
    Extracts modulation content using jepsen2008 preset:
    
    .. math::
        m_{i,k}(t) = h_{mod,k}(t) * v_i(t)
    
    Configuration: 150 Hz lowpass, attenuation factor :math:`1/\\sqrt{2}`.
    
    Output: List of :math:`N` tensors, tensor :math:`i` has shape :math:`(B, M_i, T)`
    where :math:`M_i` varies per frequency channel.
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz. Must match the audio sampling rate.
        Common values: 44100, 48000, 32000 Hz.
    
    flow : float, optional
        Lower frequency bound for gammatone filterbank in Hz. Default: 80 Hz.
        Determines the lowest auditory channel center frequency.
    
    fhigh : float, optional
        Upper frequency bound for gammatone filterbank in Hz. Default: 8000 Hz.
        Determines the highest auditory channel center frequency.
    
    phase_type : str, optional
        Filter phase characteristic for peripheral filters. Default: 'minimum'.
        Options:
        
        - ``'minimum'``: Minimum-phase FIR (causal, introduces group delay)
        - ``'zero'``: Zero-phase via filtfilt (non-causal, no phase distortion)
        
        Use 'minimum' for real-time/causal processing, 'zero' for offline analysis.
    
    learnable : bool, optional
        If True, all model stages become trainable with gradient-based optimization.
        Default: False (fixed parameters).
        When True, enables end-to-end model training for task-specific optimization.
    
    return_stages : bool, optional
        If True, returns intermediate processing stages along with final output.
        Default: False (only final modulation output).
        Useful for visualization, analysis, and multi-stage training.
    
    dtype : torch.dtype, optional
        Data type for computations and parameters. Default: torch.float32.
        Use torch.float64 for higher precision if needed (increases memory/computation).
    
    **headphone_kwargs : dict, optional
        Additional keyword arguments passed to :class:`HeadphoneFilter`.
        Common options:
        
        - ``filter_type`` (str): Filter characteristic. Default: 'pralong1996'.
        - Other parameters accepted by HeadphoneFilter.
    
    **middleear_kwargs : dict, optional
        Additional keyword arguments passed to :class:`MiddleEarFilter`.
        Common options:
        
        - ``filter_type`` (str): Filter model. Default: 'jepsen2008'.
        - ``normalize`` (bool): Normalize filter response. Default: True.
        - Other parameters accepted by MiddleEarFilter.
    
    **filterbank_kwargs : dict, optional
        Additional keyword arguments passed to :class:`GammatoneFilterbank`.
        Common options:
        
        - ``n`` (int): Filter order. Default: 4 (4th-order gammatone).
        - ``bandwidth_factor`` (float): Bandwidth scaling. Default: 1.0 (1-ERB).
        - Other parameters accepted by GammatoneFilterbank.
    
    **ihc_kwargs : dict, optional
        Additional keyword arguments passed to :class:`IHCEnvelope`.
        Common options:
        
        - ``method`` (str): Extraction method. Default: 'breebaart2001'.
        - ``cutoff`` (float): Lowpass cutoff frequency in Hz.
        - Other parameters accepted by IHCEnvelope.
    
    **adaptation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`AdaptLoop`.
        Common options:
        
        - ``preset`` (str): Configuration preset. Default: 'osses2021'.
        - ``limit`` (float): Adaptation limit. Default: 5.0.
        - ``num_loops`` (int): Number of adaptation loops. Default: 5.
        - Other parameters accepted by AdaptLoop.
    
    **modulation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`ModulationFilterbank`.
        Common options:
        
        - ``preset`` (str): Configuration preset. Default: 'jepsen2008'.
        - ``lowpass_cutoff`` (float): Lowpass frequency in Hz. Default: 150.
        - ``att_factor`` (float): Attenuation factor. Default: 1/√2.
        - Other parameters accepted by ModulationFilterbank.
        - Other parameters accepted by ModulationFilterbank.
        
    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    
    flow : float
        Lower frequency bound for filterbank.
    
    fhigh : float
        Upper frequency bound for filterbank.
    
    phase_type : str
        Filter phase characteristic ('minimum' or 'zero').
    
    learnable : bool
        Whether model parameters are trainable.
    
    return_stages : bool
        Whether to return intermediate processing stages.
    
    dtype : torch.dtype
        Data type for computations.
    
    headphone : HeadphoneFilter
        Stage 0a: Headphone and outer ear filter.
    
    middleear : MiddleEarFilter
        Stage 0b: Middle ear transmission filter.
    
    filterbank : GammatoneFilterbank
        Stage 1: Gammatone auditory filterbank.
    
    ihc : IHCEnvelope
        Stage 2: Inner hair cell envelope extraction.
    
    adaptation : AdaptLoop
        Stage 3: Adaptation loops module.
    
    modulation : ModulationFilterbank
        Stage 4: Modulation filterbank.
    
    fc : torch.Tensor
        Center frequencies of auditory channels, shape (num_channels,) in Hz.
    
    num_channels : int
        Number of auditory frequency channels (depends on flow, fhigh, ERB spacing).
    
    Input Shape
    -----------
    x : torch.Tensor
        Audio signal with shape:
        
        - :math:`(B, T)` - Batch of signals
        - :math:`(C, T)` - Multi-channel input
        - :math:`(T,)` - Single signal
        
        where:
        
        - :math:`B` = batch size
        - :math:`C` = channels
        - :math:`T` = time samples
    
    Output Shape
    ------------
    When ``return_stages=False`` (default):
        List[torch.Tensor]
            List of :math:`N` tensors (one per frequency channel), where tensor :math:`i`
            has shape :math:`(B, M_i, T)`:
            
            - :math:`N` = num_channels (number of auditory frequency channels)
            - :math:`M_i` = number of modulation channels for frequency channel :math:`i`
            - :math:`T` = time samples
            
            Note: :math:`M_i` varies across frequency channels due to jepsen2008 preset.
            
    When ``return_stages=True``:
        Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]
            - First element: modulation output (List as above)
            - Second element: dict with keys:
              
              - ``'headphone'``: After headphone filter, shape :math:`(B, T)`
              - ``'middleear'``: After middle ear filter, shape :math:`(B, T)`
              - ``'filterbank'``: After gammatone, shape :math:`(B, N, T)`
              - ``'ihc'``: After IHC envelope, shape :math:`(B, N, T)`
              - ``'adaptation'``: After adaptation, shape :math:`(B, N, T)`
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> from torch_amt.models import Osses2021
    >>> 
    >>> # Create model
    >>> model = Osses2021(fs=44100)
    >>> 
    >>> # Generate 1 second tone
    >>> audio = torch.randn(1, 44100) * 0.01
    >>> 
    >>> # Process
    >>> output = model(audio)
    >>> print(f"Number of frequency channels: {len(output)}")
    Number of frequency channels: 31
    >>> print(f"First channel shape (B, M, T): {output[0].shape}")
    First channel shape (B, M, T): torch.Size([1, 13, 44100])
    >>> print(f"Last channel shape: {output[-1].shape}")
    Last channel shape: torch.Size([1, 8, 44100])
    
    **With intermediate stages:**
    
    >>> model_debug = Osses2021(fs=44100, return_stages=True)
    >>> output, stages = model_debug(audio)
    >>> 
    >>> print(f"Available stages: {list(stages.keys())}")
    Available stages: ['headphone', 'middleear', 'filterbank', 'ihc', 'adaptation']
    >>> print(f"Filterbank output: {stages['filterbank'].shape}")
    Filterbank output: torch.Size([1, 31, 44100])
    >>> print(f"After adaptation: {stages['adaptation'].shape}")
    After adaptation: torch.Size([1, 31, 44100])
    
    **Batch processing:**
    
    >>> # Process multiple signals
    >>> batch_audio = torch.randn(4, 44100) * 0.01
    >>> output_batch = model(batch_audio)
    >>> print(f"Batch output: {output_batch[0].shape}")
    Batch output: torch.Size([4, 13, 44100])
    
    **Zero-phase filters for offline analysis:**
    
    >>> model_zero = Osses2021(fs=44100, phase_type='zero')
    >>> output_zero = model_zero(audio)
    >>> # No phase distortion, suitable for offline processing
    
    **Custom frequency range:**
    
    >>> # Narrow frequency range
    >>> model_narrow = Osses2021(fs=44100, flow=500, fhigh=4000)
    >>> print(f"Channels: {model_narrow.num_channels}")
    Channels: 15
    >>> print(f"Center frequencies: {model_narrow.fc}")
    Center frequencies: tensor([ 500.,  631., ..., 3175., 4000.])
    
    **Custom submodule parameters:**
    
    >>> # Custom adaptation limit
    >>> model_custom = Osses2021(
    ...     fs=44100,
    ...     adaptation_kwargs={'limit': 10.0, 'num_loops': 7}
    ... )
    >>> 
    >>> # Custom modulation filterbank
    >>> model_mod = Osses2021(
    ...     fs=44100,
    ...     modulation_kwargs={'lowpass_cutoff': 200.0}
    ... )
    
    **Learnable model for optimization:**
    
    >>> model_learnable = Osses2021(fs=44100, learnable=True)
    >>> n_params = sum(p.numel() for p in model_learnable.parameters())
    >>> print(f"Trainable parameters: {n_params}")
    Trainable parameters: 2648
    >>> 
    >>> # Example training loop
    >>> optimizer = torch.optim.Adam(model_learnable.parameters(), lr=1e-3)
    >>> # ... training code ...
    
    **Accessing center frequencies:**
    
    >>> model = Osses2021(fs=44100)
    >>> print(f"Center frequencies (Hz): {model.fc[:5]}")
    Center frequencies (Hz): tensor([  80.00,  100.79,  126.96,  159.91,  201.42])
    >>> print(f"Frequency range: {model.fc[0]:.1f} - {model.fc[-1]:.1f} Hz")
    Frequency range: 80.0 - 8000.0 Hz
    
    Notes
    -----
    **Model Configuration:**
    
    - **Headphone**: Pralong & Carlile (1996) outer ear + headphone compensation
    - **Middle Ear**: Jepsen et al. (2008) bandpass characteristic
    - **Filterbank**: Gammatone 4th-order, 1-ERB spacing
    - **IHC**: Breebaart et al. (2001) envelope extraction method
    - **Adaptation**: 5 loops, limit=5.0 (osses2021 preset)
    - **Modulation**: jepsen2008 preset (150 Hz lowpass, att_factor=1/√2)
    
    **Customizing Submodule Parameters:**
    
    All submodules can be customized through dedicated kwargs dictionaries:
    
    - Use ``headphone_kwargs`` to pass parameters to :class:`HeadphoneFilter`
    - Use ``middleear_kwargs`` to pass parameters to :class:`MiddleEarFilter`
    - Use ``filterbank_kwargs`` to pass parameters to :class:`GammatoneFilterbank`
    - Use ``ihc_kwargs`` to pass parameters to :class:`IHCEnvelope`
    - Use ``adaptation_kwargs`` to pass parameters to :class:`AdaptLoop`
    - Use ``modulation_kwargs`` to pass parameters to :class:`ModulationFilterbank`
    
    The ``learnable``, ``dtype``, and ``phase_type`` parameters are always 
    centralized and applied to all submodules automatically. Custom parameters 
    override defaults while maintaining the Osses2021 model structure.
    
    **Phase Type Selection:**
    
    The ``phase_type`` parameter controls peripheral filter characteristics:
    
    - **'minimum'** (default): Causal minimum-phase FIR filters
      
      - Introduces frequency-dependent group delay
      - Suitable for real-time or causal processing
      - Matches physiological phase response
    
    - **'zero'**: Zero-phase filtering via filtfilt
      
      - No phase distortion (symmetric impulse response)
      - Non-causal (requires future samples)
      - Better for offline analysis and visualization
    
    Choose based on application: real-time → 'minimum', offline → 'zero'.
    
    **Output Format (List of Tensors):**
    
    Unlike other models that return a single tensor, Osses2021 returns a
    **List[torch.Tensor]** because each frequency channel has a different
    number of modulation channels:
    
    .. code-block:: python
    
        output = model(audio)  # List of N tensors
        for i, channel_output in enumerate(output):
            print(f"Channel {i}: {channel_output.shape}")
            # Channel 0: torch.Size([B, 13, T])
            # Channel 1: torch.Size([B, 12, T])
            # ...
    
    This reflects the jepsen2008 modulation filterbank configuration where
    higher frequency channels have fewer modulation channels. To work with
    a single tensor, you can concatenate or pad as needed for your application.
    
    **Computational Complexity:**
    
    Processing time scales as:
    
    .. math::
        T_{compute} \\propto T \\cdot (N_{filter} + N_{filt} \\cdot N_{mod})
    
    where :math:`T` = signal length, :math:`N_{filter}` = peripheral filter taps,
    :math:`N_{filt}` = number of frequency channels (~31), 
    :math:`N_{mod}` = modulation channels per frequency (~8-13).
    
    For 1 second @ 44.1 kHz: ~0.1-0.5 seconds on CPU, ~0.01-0.05 seconds on GPU.
    
    **Memory Requirements:**
    
    Peak memory with intermediate stages:
    
    .. math::
        Memory \\approx B \\cdot T \\cdot (N + \\sum_i M_i) \\cdot 4\\,\\text{bytes}
    
    For batch=8, 1 second @ 44.1 kHz: ~40-80 MB.
    
    **Applications:**
    
    The model is particularly suited for:
    
    - Fluctuation strength prediction (original application)
    - Roughness and amplitude modulation detection
    - Headphone-presented stimuli modeling
    - Psychoacoustic feature extraction
    - Perceptual quality assessment
    - Machine learning auditory features
    
    See Also
    --------
    HeadphoneFilter : Stage 0a - Headphone and outer ear filtering
    MiddleEarFilter : Stage 0b - Middle ear transmission
    GammatoneFilterbank : Stage 1 - Auditory filterbank
    IHCEnvelope : Stage 2 - Inner hair cell envelope
    AdaptLoop : Stage 3 - Adaptation loops
    ModulationFilterbank : Stage 4 - Modulation analysis
    
    References
    ----------
    .. [1] A. Osses Vecchi, L. E. García, and A. Kohlrausch, "Modelling the 
           sensation of fluctuation strength," in *Proc. Forum Acusticum 2020*, 
           Lyon, France, 2020, pp. 367-371.
    
    .. [2] D. Pralong and S. Carlile, "The role of individualized headphone 
           calibration for the generation of high fidelity virtual auditory space," 
           *J. Acoust. Soc. Am.*, vol. 100, no. 6, pp. 3785-3793, Dec. 1996.
    
    .. [3] M. L. Jepsen, S. D. Ewert, and T. Dau, "A computational model of 
           human auditory signal processing and perception," *J. Acoust. Soc. Am.*, 
           vol. 124, no. 1, pp. 422-438, Jul. 2008.
    
    .. [4] J. Breebaart, S. van de Par, and A. Kohlrausch, "Binaural processing 
           model based on contralateral inhibition. I. Model structure," 
           *J. Acoust. Soc. Am.*, vol. 110, no. 2, pp. 1074-1088, Aug. 2001.
    
    .. [5] T. Dau, D. Püschel, and A. Kohlrausch, "A quantitative model of the 
           'effective' signal processing in the auditory system. I. Model structure," 
           *J. Acoust. Soc. Am.*, vol. 99, no. 6, pp. 3615-3622, Jun. 1996.
    
    .. [6] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
           reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
           p. 19, 2022.
    """
    
    def __init__(self,
                 fs: float,
                 flow: float = 80.0,
                 fhigh: float = 8000.0,
                 phase_type: str = 'minimum',
                 learnable: bool = False,
                 return_stages: bool = False,
                 dtype: torch.dtype = torch.float32,
                 headphone_kwargs: Dict[str, Any] = None,
                 middleear_kwargs: Dict[str, Any] = None,
                 filterbank_kwargs: Dict[str, Any] = None,
                 ihc_kwargs: Dict[str, Any] = None,
                 adaptation_kwargs: Dict[str, Any] = None,
                 modulation_kwargs: Dict[str, Any] = None):
        super().__init__()
        
        self.fs = fs
        self.flow = flow
        self.fhigh = fhigh
        self.phase_type = phase_type
        self.learnable = learnable
        self.return_stages = return_stages
        self.dtype = dtype
        
        # Initialize kwargs dictionaries
        headphone_kwargs = headphone_kwargs or {}
        middleear_kwargs = middleear_kwargs or {}
        filterbank_kwargs = filterbank_kwargs or {}
        ihc_kwargs = ihc_kwargs or {}
        adaptation_kwargs = adaptation_kwargs or {}
        modulation_kwargs = modulation_kwargs or {}
        
        # Stage 0a: Headphone filter (outer ear + headphone compensation)
        self.headphone = HeadphoneFilter(fs=fs, 
                                         phase_type=phase_type, 
                                         learnable=learnable, 
                                         dtype=dtype,
                                         **headphone_kwargs)
        
        # Stage 0b: Middle ear filter (Jepsen 2008 variant)
        middleear_defaults = {'filter_type': 'jepsen2008'}
        middleear_params = {**middleear_defaults, **middleear_kwargs}
        self.middleear = MiddleEarFilter(fs=fs,
                                         phase_type=phase_type,
                                         learnable=learnable,
                                         dtype=dtype,
                                         **middleear_params)
        
        # Stage 1: Gammatone filterbank
        filterbank_defaults = {'n': 4}
        filterbank_params = {**filterbank_defaults, **filterbank_kwargs}
        self.filterbank = GammatoneFilterbank(fc=(flow, fhigh), 
                                              fs=fs, 
                                              learnable=learnable, 
                                              dtype=dtype,
                                              **filterbank_params)
        self.fc = self.filterbank.fc
        self.num_channels = self.filterbank.num_channels
        
        # Stage 2: Inner hair cell envelope extraction (Breebaart 2001 method)
        ihc_defaults = {'method': 'breebaart2001'}
        ihc_params = {**ihc_defaults, **ihc_kwargs}
        self.ihc = IHCEnvelope(fs=fs, learnable=learnable, dtype=dtype, **ihc_params)
        
        # Stage 3: Adaptation loops (osses2021 preset: limit=5.0)
        adaptation_defaults = {'preset': 'osses2021'}
        adaptation_params = {**adaptation_defaults, **adaptation_kwargs}
        self.adaptation = AdaptLoop(fs=fs, learnable=learnable, dtype=dtype, **adaptation_params)
        
        # Stage 4: Modulation filterbank (jepsen2008 preset)
        modulation_defaults = {'preset': 'jepsen2008'}
        modulation_params = {**modulation_defaults, **modulation_kwargs}
        # self.modulation = ModulationFilterbank(fs=fs, fc=self.fc, learnable=learnable, dtype=dtype, **modulation_params) 
        self.modulation = FastModulationFilterbank(fs=fs, fc=self.fc, learnable=learnable, dtype=dtype, **modulation_params) 
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor] | tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process audio through the Osses2021 model.
        
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
        
        # Stage 0a: Headphone filter
        # Input: [B, T], Output: [B, T]
        x = self.headphone(x)
        if self.return_stages:
            stages['headphone'] = x.clone()
        
        # Stage 0b: Middle ear filter
        # Input: [B, T], Output: [B, T]
        x = self.middleear(x)
        if self.return_stages:
            stages['middleear'] = x.clone()
        
        # Stage 1: Gammatone filterbank
        # Input: [B, T], Output: [B, F, T]
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
                f"phase_type={self.phase_type}, num_channels={self.num_channels}, "
                f"learnable={self.learnable}")
    
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
        
        >>> model = Osses2021(fs=44100, learnable=True)
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
