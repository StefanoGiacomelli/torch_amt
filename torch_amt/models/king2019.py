"""
King2019 Auditory Model w. Non-Linear Filtering & Compression
=============================================================

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the King et al. (2019) auditory model for studying
masking of frequency modulation by amplitude modulation. The model features
a broken-stick or power-law compression stage and a modulation filterbank
with logarithmic spacing designed for modulation analysis.

The implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT)
and extended with PyTorch for gradient-based optimization and GPU acceleration.

References
----------
.. [1] A. King, L. Varnet, and C. Lorenzi, "Accounting for masking of 
       frequency modulation by amplitude modulation with the modulation 
       filter-bank concept," *J. Acoust. Soc. Am.*, vol. 145, no. 4, 
       pp. 2277-2293, Apr. 2019.

.. [2] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
       of amplitude modulation. I. Detection and masking with narrow-band carriers," 
       *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, Nov. 1997.

.. [3] R. D. Patterson, I. Nimmo-Smith, J. Holdsworth, and P. Rice, "An 
       efficient auditory filterbank based on the gammatone function," 
       in *Proc. Meet. IOC Speech Group Auditory Modelling*, 1988.

.. [4] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
       p. 19, 2022.
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from torch_amt.common.filterbanks import fc2erb, erb2fc
from torch_amt.common.filterbanks import GammatoneFilterbank
from torch_amt.common.loudness import BrokenStickCompression, PowerCompression
from torch_amt.common.ihc import IHCEnvelope
from torch_amt.common.filters import ButterworthFilter
# from torch_amt.common.modulation import King2019ModulationFilterbank
from torch_amt.common.modulation import FastKing2019ModulationFilterbank


class King2019(nn.Module):
    r"""
    King et al. (2019) auditory model with nonlinear compression.
    
    Implements a computational model of the auditory periphery designed for
    studying masking of frequency modulation (FM) by amplitude modulation (AM).
    The model features explicit nonlinear compression (broken-stick or power-law)
    and adaptation stages, with a modulation filterbank for extracting temporal
    modulation content.
    
    This implementation follows the MATLAB Auditory Modeling Toolbox (AMT)
    king2019 configuration and provides a differentiable, GPU-accelerated
    version suitable for neural network training and optimization.
    
    Algorithm Overview
    ------------------
    The model implements a 6-stage auditory processing pipeline:
    
    **Stage 1: Gammatone Filterbank**
    
    Decomposes signal into :math:`N` frequency channels with 1-ERB spacing:
    
    .. math::
        y_i(t) = g_i(t) * x(t), \\quad i=1,\\ldots,N
    
    where :math:`g_i(t) = t^{3} e^{-2\\pi b_i t} \\cos(2\\pi f_i t)` is a
    4th-order gammatone impulse response.
    
    **Stage 2: Nonlinear Compression**
    
    Two compression types are available:
    
    *Broken-stick compression* (default):
    
    .. math::
        c_i(t) = \\begin{cases}
            |y_i(t)|, & \\text{if } L_i(t) < L_{\\text{knee}} \\\\
            10^{(L_{\\text{knee}}/20)} \\cdot \\left(\\frac{|y_i(t)|}{10^{(L_{\\text{knee}}/20)}}\\right)^n, 
            & \\text{if } L_i(t) \\geq L_{\\text{knee}}
        \\end{cases}
    
    where :math:`L_i(t) = 20\\log_{10}(|y_i(t)|/p_{\\text{ref}}) + \\text{dboffset}`,
    :math:`L_{\\text{knee}}` is the knee point (default 30 dB), and :math:`n`
    is the compression exponent (default 0.3).
    
    *Power-law compression*:
    
    .. math::
        c_i(t) = |y_i(t)|^n
    
    **Stage 3: Inner Hair Cell (IHC) Envelope**
    
    Extracts envelope via half-wave rectification and lowpass filtering:
    
    .. math::
        e_i(t) = h_{\\text{lp}}(t) * \\max(0, c_i(t))
    
    King2019 uses 1000 Hz cutoff (1st-order Butterworth lowpass).
    
    **Stage 4: Adaptation**
    
    High-pass filtering removes DC and slow drifts:
    
    .. math::
        a_i(t) = h_{\\text{hp}}(t) * e_i(t)
    
    Default: 3 Hz cutoff, 1st-order Butterworth highpass.
    
    **Stage 5: Optional 150 Hz Lowpass**
    
    Pre-filtering before modulation analysis (if ``lp_150hz=True``):
    
    .. math::
        a'_i(t) = h_{150}(t) * a_i(t)
    
    **Stage 6: Modulation Filterbank**
    
    Extracts modulation frequencies using bandpass filters:
    
    .. math::
        m_{i,k}(t) = h_{\\text{mod},k}(t) * a'_i(t)
    
    Modulation center frequencies :math:`f_{\\text{mod},k}` are logarithmically
    spaced from :math:`f_{\\text{low}}` (default 2 Hz) to :math:`f_{\\text{high}}`
    (default 150 Hz) based on Q-factor. Each filter is a 2nd-order Butterworth
    bandpass with bandwidth :math:`\\text{BW} = f_{\\text{mod},k} / Q`.
    
    Output: Tensor of shape :math:`(B, T, F, M)` where :math:`M` is the number
    of modulation channels.
    
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
    
    basef : float, optional
        Base frequency in Hz for centered analysis. If provided, ``flow`` and 
        ``fhigh`` are automatically computed as ``basef ± 2 ERB``, creating
        a narrow frequency range centered on ``basef``. Default: None (use
        ``flow``/``fhigh``).
    
    compression_type : {'brokenstick', 'power'}, optional
        Type of nonlinear compression. Default: 'brokenstick'.
        
        - ``'brokenstick'``: Two-segment compression with linear below knee,
          power-law above knee (physiologically realistic)
        - ``'power'``: Simple power-law compression :math:`y = x^n`
    
    compression_n : float, optional
        Compression exponent for power-law segment. Default: 0.3.
        Typical physiological values: 0.2-0.4.
    
    compression_knee_db : float, optional
        Knee point in dB SPL for broken-stick compression. Default: 30 dB.
        Signals below this level are uncompressed; above are compressed.
    
    dboffset : float, optional
        dB full scale convention (calibration). Default: 100 dB SPL.
        
        - 100 dB: MATLAB AMT convention (0 dBFS = 100 dB SPL)
        - 94 dB: Alternative convention for specific AMT signals
        
        Must match the signal's reference level for correct compression.
    
    adt_hp_fc : float, optional
        Adaptation highpass cutoff frequency in Hz. Default: 3.0 Hz.
        Removes DC offset and very slow drifts.
    
    adt_hp_order : int, optional
        Adaptation highpass filter order. Default: 1 (1st-order Butterworth).
        Higher orders provide steeper roll-off but may introduce artifacts.
    
    mflow : float, optional
        Minimum modulation frequency for modulation filterbank in Hz.
        Default: 2.0 Hz. Lowest temporal modulation captured.
    
    mfhigh : float, optional
        Maximum modulation frequency for modulation filterbank in Hz.
        Default: 150.0 Hz. Highest temporal modulation captured.
    
    modbank_nmod : int, optional
        Number of modulation filters. If None, automatically determined by
        Q-factor spacing (typically 5-10 filters). Default: None (automatic).
    
    modbank_qfactor : float, optional
        Q-factor for modulation filters, controlling bandwidth:
        :math:`\\text{BW} = f_{\\text{mod}} / Q`. Default: 1.0.
        Higher Q → narrower filters, better frequency resolution.
    
    lp_150hz : bool, optional
        Apply 150 Hz lowpass filter before modulation filterbank. Default: False.
        Useful for limiting analysis to lower modulation frequencies.
    
    subfs : float, optional
        Target sampling rate in Hz for downsampling output. If None, no
        downsampling is applied. Default: None (keep original fs).
        Reduces computational cost for downstream processing.
    
    learnable : bool, optional
        If True, all model stages become trainable with gradient-based optimization.
        Default: False (fixed parameters).
        Enables end-to-end model training for task-specific optimization.
    
    return_stages : bool, optional
        If True, returns intermediate processing stages along with final output.
        Default: False (only final modulation output).
        Useful for visualization, analysis, and multi-stage training.
    
    dtype : torch.dtype, optional
        Data type for computations and parameters. Default: torch.float32.
        Use torch.float64 for higher precision if needed.
    
    **filterbank_kwargs : dict, optional
        Additional keyword arguments passed to :class:`GammatoneFilterbank`.
        Common options:
        
        - ``n`` (int): Filter order. Default: 4.
        - ``betamul`` (float): Beta multiplier. Default: 1.0186.
        - Other parameters accepted by GammatoneFilterbank.
    
    **compression_kwargs : dict, optional
        Additional keyword arguments passed to :class:`BrokenStickCompression` 
        or :class:`PowerCompression` depending on ``compression_type``.
        
        For BrokenStickCompression:
        
        - No additional parameters typically needed beyond those in main signature.
        
        For PowerCompression:
        
        - No additional parameters typically needed beyond those in main signature.
    
    **ihc_kwargs : dict, optional
        Additional keyword arguments passed to :class:`IHCEnvelope`.
        Common options:
        
        - ``method`` (str): IHC method. Fixed to 'king2019' for this model.
        - Other parameters accepted by IHCEnvelope.
    
    **adaptation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`ButterworthFilter` 
        (adaptation highpass filter).
        
        - No additional parameters typically needed beyond fc and order.
    
    **modulation_kwargs : dict, optional
        Additional keyword arguments passed to :class:`King2019ModulationFilterbank`.
        
        - No additional parameters typically needed beyond those in main signature.
    
    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    
    flow : float
        Lower frequency bound for filterbank.
    
    fhigh : float
        Upper frequency bound for filterbank.
    
    dboffset : float
        dB full scale calibration.
    
    compression_type : str
        Type of compression ('brokenstick' or 'power').
    
    learnable : bool
        Whether model parameters are trainable.
    
    return_stages : bool
        Whether to return intermediate processing stages.
    
    dtype : torch.dtype
        Data type for computations.
    
    filterbank : GammatoneFilterbank
        Stage 1: Gammatone auditory filterbank.
    
    compression : BrokenStickCompression or PowerCompression
        Stage 2: Nonlinear compression module.
    
    ihc : IHCEnvelope
        Stage 3: Inner hair cell envelope extraction.
    
    adaptation_filter : ButterworthFilter
        Stage 4: Adaptation highpass filter.
    
    lp150_filter : ButterworthFilter or None
        Stage 5: Optional 150 Hz lowpass filter.
    
    modulation_filterbank : King2019ModulationFilterbank
        Stage 6: Modulation filterbank module.
    
    fc : torch.Tensor
        Center frequencies of auditory channels, shape (num_channels,) in Hz.
    
    mfc : torch.Tensor
        Center frequencies of modulation channels, shape (num_mod_filters,) in Hz.
    
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
        torch.Tensor
            Output tensor with shape :math:`(B, T', F, M)`:
            
            - :math:`B` = batch size
            - :math:`T'` = time samples (possibly downsampled if subfs specified)
            - :math:`F` = num_channels (number of auditory frequency channels)
            - :math:`M` = number of modulation channels (depends on mflow, mfhigh, qfactor)
            
    When ``return_stages=True``:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            - First element: modulation output (shape as above)
            - Second element: dict with keys:
              
              - ``'gtone_response'``: After gammatone, shape :math:`(B, F, T)`
              - ``'compressed_response'``: After compression, shape :math:`(B, F, T)`
              - ``'ihc'``: After IHC envelope, shape :math:`(B, F, T)`
              - ``'adapted_response'``: After adaptation, shape :math:`(B, F, T)`
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> from torch_amt.models import King2019
    >>> 
    >>> # Create model
    >>> model = King2019(fs=48000, basef=5000)
    >>> 
    >>> # Generate 0.5 second tone
    >>> audio = torch.randn(2, 24000) * 0.01
    >>> 
    >>> # Process
    >>> output = model(audio)
    >>> print(f"Input: {audio.shape}, Output: {output.shape}")
    Input: torch.Size([2, 24000]), Output: torch.Size([2, 24000, 5, 5])
    >>> print(f"Frequency channels: {model.num_channels}")
    Frequency channels: 5
    >>> print(f"Modulation channels: {len(model.mfc)}")
    Modulation channels: 5
    
    **With intermediate stages:**
    
    >>> model_debug = King2019(fs=48000, basef=5000, return_stages=True)
    >>> output, stages = model_debug(audio)
    >>> 
    >>> print(f"Available stages: {list(stages.keys())}")
    Available stages: ['gtone_response', 'compressed_response', 'ihc', 'adapted_response']
    >>> print(f"After gammatone: {stages['gtone_response'].shape}")
    After gammatone: torch.Size([2, 5, 24000])
    >>> print(f"After compression: {stages['compressed_response'].shape}")
    After compression: torch.Size([2, 5, 24000])
    
    **Batch processing:**
    
    >>> # Process multiple signals
    >>> batch_audio = torch.randn(8, 48000) * 0.01
    >>> output_batch = model(batch_audio)
    >>> print(f"Batch output: {output_batch.shape}")
    Batch output: torch.Size([8, 48000, 5, 5])
    
    **Using basef for frequency-specific analysis:**
    
    >>> # Analyze around 1 kHz (±2 ERB)
    >>> model_1k = King2019(fs=48000, basef=1000)
    >>> print(f"Channels: {model_1k.num_channels}")
    Channels: 5
    >>> print(f"Center frequencies: {model_1k.fc}")
    Center frequencies: tensor([ 794.3, 1000.0, 1259.2, ...])
    
    **Power-law compression:**
    
    >>> # Simple power-law instead of broken-stick
    >>> model_power = King2019(fs=48000, compression_type='power', compression_n=0.4)
    >>> output_power = model_power(audio)
    
    **Custom modulation filterbank:**
    
    >>> # Wider modulation range with higher Q
    >>> model_mod = King2019(
    ...     fs=48000,
    ...     mflow=1.0,
    ...     mfhigh=200.0,
    ...     modbank_qfactor=2.0,
    ...     lp_150hz=True
    ... )
    >>> print(f"Modulation frequencies: {model_mod.mfc}")
    Modulation frequencies: tensor([  1.00,   3.00,   9.00,  27.00, ...])
    
    **With downsampling:**
    
    >>> # Downsample output to 1000 Hz for efficiency
    >>> model_ds = King2019(fs=48000, subfs=1000)
    >>> audio_long = torch.randn(1, 480000) * 0.01  # 10 seconds
    >>> output_ds = model_ds(audio_long)
    >>> print(f"Downsampled output: {output_ds.shape}")
    Downsampled output: torch.Size([1, 10000, 31, 10])  # T reduced from 480k to 10k
    
    **Learnable model for optimization:**
    
    >>> model_learnable = King2019(fs=48000, learnable=True)
    >>> n_params = sum(p.numel() for p in model_learnable.parameters())
    >>> print(f"Trainable parameters: {n_params}")
    Trainable parameters: 2156
    >>> 
    >>> # Example training loop
    >>> optimizer = torch.optim.Adam(model_learnable.parameters(), lr=1e-3)
    >>> # ... training code ...
    
    **Accessing center frequencies:**
    
    >>> model = King2019(fs=48000)
    >>> print(f"Auditory fc (Hz): {model.fc[:5]}")
    Auditory fc (Hz): tensor([  80.0,  100.8,  126.9,  159.9,  201.4])
    >>> print(f"Modulation mfc (Hz): {model.mfc}")
    Modulation mfc (Hz): tensor([  2.00,   5.24,  13.71,  35.89,  93.96])
    
    Notes
    -----
    **Model Configuration:**
    
    - **Filterbank**: Gammatone 4th-order, 1-ERB spacing
    - **Compression**: Broken-stick (knee=30dB, n=0.3) or power-law
    - **IHC**: Half-wave rectification + 1000 Hz lowpass (king2019 method)
    - **Adaptation**: 3 Hz highpass, 1st-order Butterworth
    - **Modulation**: 2-150 Hz, Q=1.0, 2nd-order Butterworth bandpass
    
    **Compression Calibration:**
    
    The compression stage requires correct calibration via ``dboffset``:
    
    - **100 dB** (default): MATLAB AMT convention (0 dBFS = 100 dB SPL)
    - **94 dB**: Alternative convention for specific AMT signals
    
    The ``dboffset`` must match your signal's reference level. Incorrect
    calibration leads to improper compression behavior and inaccurate
    modulation representations.
    
    Example calibration:
    
    .. code-block:: python
    
        # For signals calibrated to 100 dB SPL at 0 dBFS
        model = King2019(fs=48000, dboffset=100.0)
        
        # For AMT-specific signals at 94 dB SPL
        model = King2019(fs=48000, dboffset=94.0)
    
    **basef Parameter:**
    
    When ``basef`` is specified, ``flow`` and ``fhigh`` are automatically
    computed to create a narrow frequency range:
    
    .. math::
        \\text{ERB}_{\\text{base}} = \\text{fc2erb}(\\text{basef})
    
    .. math::
        f_{\\text{low}} = \\text{erb2fc}(\\text{ERB}_{\\text{base}} - 2)
    
    .. math::
        f_{\\text{high}} = \\text{erb2fc}(\\text{ERB}_{\\text{base}} + 2)
    
    This creates ~5 channels centered on ``basef``, useful for frequency-specific
    analysis (e.g., studying FM masking at a particular carrier frequency).
    
    **Computational Complexity:**
    
    Processing time scales as:
    
    .. math::
        T_{\\text{compute}} \\propto T \\cdot (N_{\\text{filt}} + N_{\\text{filt}} \\cdot N_{\\text{mod}})
    
    where :math:`T` = signal length, :math:`N_{\\text{filt}}` = auditory channels (~5-31),
    :math:`N_{\\text{mod}}` = modulation channels (~5-10).
    
    For 1 second @ 48 kHz: ~0.05-0.2 seconds on CPU, ~0.01-0.05 seconds on GPU.
    
    **Memory Requirements:**
    
    Peak memory with intermediate stages:
    
    .. math::
        \\text{Memory} \\approx B \\cdot T \\cdot (F + F \\cdot M) \\cdot 4\\,\\text{bytes}
    
    For batch=4, 1 second @ 48 kHz, F=5, M=5: ~5-10 MB.
    
    **Applications:**
    
    The model is particularly suited for:
    
    - Frequency modulation (FM) masking studies
    - Amplitude modulation (AM) masking studies
    - FM-AM interaction analysis
    - Temporal modulation transfer functions (TMTF)
    - Psychoacoustic feature extraction
    - Auditory scene analysis
    
    See Also
    --------
    GammatoneFilterbank : Stage 1 - Auditory filterbank
    BrokenStickCompression : Stage 2 - Nonlinear compression
    PowerCompression : Stage 2 - Alternative compression
    IHCEnvelope : Stage 3 - Inner hair cell envelope
    ButterworthFilter : Stages 4-5 - Filtering
    King2019ModulationFilterbank : Stage 6 - Modulation analysis
    
    References
    ----------
    .. [1] A. King, L. Varnet, and C. Lorenzi, "Accounting for masking of 
           frequency modulation by amplitude modulation with the modulation 
           filter-bank concept," *J. Acoust. Soc. Am.*, vol. 145, no. 4, 
           pp. 2277-2293, Apr. 2019.
    
    .. [2] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
           of amplitude modulation. I. Detection and masking with narrow-band carriers," 
           *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, Nov. 1997.
    
    .. [3] R. D. Patterson, I. Nimmo-Smith, J. Holdsworth, and P. Rice, "An 
           efficient auditory filterbank based on the gammatone function," 
           in *Proc. Meet. IOC Speech Group Auditory Modelling*, 1988.
    
    .. [4] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
           reproducible research in auditory modeling," *Acta Acust.*, vol. 6, 
           p. 19, 2022.
    """
    
    def __init__(self,
                 fs: float,
                 flow: float = 80.0,
                 fhigh: float = 8000.0,
                 basef: Optional[float] = None,
                 compression_type: str = 'brokenstick',
                 compression_n: float = 0.3,
                 compression_knee_db: float = 30.0,
                 dboffset: float = 100.0,
                 adt_hp_fc: float = 3.0,
                 adt_hp_order: int = 1,
                 mflow: float = 2.0,
                 mfhigh: float = 150.0,
                 modbank_nmod: Optional[int] = None,
                 modbank_qfactor: float = 1.0,
                 lp_150hz: bool = False,
                 subfs: Optional[float] = None,
                 learnable: bool = False,
                 return_stages: bool = False,
                 dtype: torch.dtype = torch.float32,
                 filterbank_kwargs: Optional[Dict[str, Any]] = None,
                 compression_kwargs: Optional[Dict[str, Any]] = None,
                 ihc_kwargs: Optional[Dict[str, Any]] = None,
                 adaptation_kwargs: Optional[Dict[str, Any]] = None,
                 modulation_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Store parameters
        self.fs = fs
        self.dboffset = dboffset
        self.compression_type = compression_type
        self.compression_n = compression_n
        self.compression_knee_db = compression_knee_db
        self.adt_hp_fc = adt_hp_fc
        self.adt_hp_order = adt_hp_order
        self.mflow = mflow
        self.mfhigh = mfhigh
        self.modbank_nmod = modbank_nmod
        self.modbank_qfactor = modbank_qfactor
        self.lp_150hz = lp_150hz
        self.subfs = subfs
        self.learnable = learnable
        self.return_stages = return_stages
        self.dtype = dtype
        
        # Initialize kwargs dictionaries if None
        filterbank_kwargs = filterbank_kwargs or {}
        compression_kwargs = compression_kwargs or {}
        ihc_kwargs = ihc_kwargs or {}
        adaptation_kwargs = adaptation_kwargs or {}
        modulation_kwargs = modulation_kwargs or {}
        
        # Validate and compute flow/fhigh
        if basef is not None:
            # Compute flow/fhigh from basef (±2 ERB)
            erb_base = fc2erb(torch.tensor(basef)).item()
            self.flow = float(erb2fc(torch.tensor(erb_base - 2.0)).item())
            self.fhigh = float(erb2fc(torch.tensor(erb_base + 2.0)).item())
        else:
            self.flow = flow
            self.fhigh = fhigh
        
        # Stage 1: Gammatone filterbank (1-ERB spacing, 4th order)
        filterbank_defaults = {'n': 4}
        filterbank_params = {**filterbank_defaults, **filterbank_kwargs}
        self.filterbank = GammatoneFilterbank(fc=(self.flow, self.fhigh),
                                              fs=self.fs,
                                              learnable=self.learnable,
                                              dtype=self.dtype,
                                              **filterbank_params)
        self.fc = self.filterbank.fc
        self.num_channels = self.filterbank.num_channels
        
        # Stage 2: Compression
        compression_defaults = {}
        compression_params = {**compression_defaults, **compression_kwargs}
        if self.compression_type == 'brokenstick':
            self.compression = BrokenStickCompression(knee_db=self.compression_knee_db,
                                                      exponent=self.compression_n,
                                                      dboffset=self.dboffset,
                                                      num_channels=self.num_channels,
                                                      **compression_params)
        elif self.compression_type == 'power':
            self.compression = PowerCompression(knee_db=self.compression_knee_db,
                                                exponent=self.compression_n,
                                                dboffset=self.dboffset,
                                                num_channels=self.num_channels,
                                                **compression_params)
        else:
            raise ValueError(f"Unknown compression_type: {self.compression_type}")
        
        # Stage 3: IHC envelope extraction (King 2019 uses 1500 Hz but implements 1000 Hz)
        # Following the actual MATLAB implementation (1000 Hz)
        ihc_defaults = {}
        ihc_params = {**ihc_defaults, **ihc_kwargs}
        self.ihc = IHCEnvelope(fs=self.fs,
                               method='king2019',
                               learnable=self.learnable,
                               dtype=self.dtype,
                               **ihc_params)
        
        # Stage 4: Adaptation via high-pass filtering (using ButterworthFilter)
        adaptation_defaults = {}
        adaptation_params = {**adaptation_defaults, **adaptation_kwargs}
        self.adaptation_filter = ButterworthFilter(order=self.adt_hp_order,
                                                   cutoff=self.adt_hp_fc,
                                                   fs=self.fs,
                                                   btype='high',
                                                   learnable=self.learnable,
                                                   dtype=self.dtype,
                                                   **adaptation_params)
        
        # Stage 5: Optional 150 Hz lowpass (before modulation filterbank)
        if self.lp_150hz:
            self.lp150_filter = ButterworthFilter(order=1,
                                                  cutoff=150.0,
                                                  fs=self.fs,
                                                  btype='low',
                                                  learnable=self.learnable,
                                                  dtype=self.dtype)
        else:
            self.lp150_filter = None
        
        # Stage 6: Modulation filterbank (using King2019ModulationFilterbank)
        modulation_defaults = {}
        modulation_params = {**modulation_defaults, **modulation_kwargs}
        # Build explicit params dict, allowing kwargs to override
        modulation_explicit = {'fs': self.fs,
                               'mflow': self.mflow,
                               'mfhigh': self.mfhigh,
                               'qfactor': self.modbank_qfactor,
                               'nmod': self.modbank_nmod,
                               'learnable': self.learnable,
                               'dtype': self.dtype}
        # Merge: explicit params first, then kwargs override
        modulation_final = {**modulation_explicit, **modulation_params}
        
        # self.modulation_filterbank = King2019ModulationFilterbank(**modulation_final)
        self.modulation_filterbank = FastKing2019ModulationFilterbank(**modulation_final)
        self.mfc = self.modulation_filterbank.mfc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        """Process audio through the KING2019 model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input audio signal. Shape: (B, T), (C, T), or (T,).
            
        Returns
        -------
        torch.Tensor or tuple
            If return_stages=False:
                Output tensor of shape (B, T', F, M) where:
                    T' = time samples (possibly downsampled)
                    F = number of frequency channels
                    M = number of modulation channels
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
            stages['gtone_response'] = x.clone()
        
        # Stage 2: Compression
        # Input: [B, F, T], Output: [B, F, T]
        # Transpose to (B, T, F) for compression, then back
        x = x.permute(0, 2, 1)  # [B, F, T] -> [B, T, F]
        x = self.compression(x)
        x = x.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
        if self.return_stages:
            stages['compressed_response'] = x.clone()
        
        # Stage 3: IHC envelope extraction
        # Input: [B, F, T], Output: [B, F, T]
        x = self.ihc(x)
        if self.return_stages:
            stages['ihc'] = x.clone()
        
        # Stage 4: Adaptation (high-pass filtering)
        # Input: [B, F, T], Output: [B, F, T]
        x = self.adaptation_filter(x)
        if self.return_stages:
            stages['adapted_response'] = x.clone()
        
        # Stage 5: Optional 150 Hz lowpass
        if self.lp150_filter is not None:
            x = self.lp150_filter(x)
        
        # Stage 6: Modulation filterbank
        # Input: [B, F, T], Output: [B, F, M, T]
        x = self.modulation_filterbank(x)
        
        # Permute to (B, T, F, M)
        x = x.permute(0, 3, 1, 2)  # [B, F, M, T] -> [B, T, F, M]
        
        # Stage 7: Downsampling if requested
        if self.subfs is not None and self.subfs != self.fs:
            target_len = int(x.shape[1] / self.fs * self.subfs)
            x = torch.nn.functional.interpolate(x.permute(0, 2, 3, 1),  # [B, T, F, M] -> [B, F, M, T]
                                                size=target_len,
                                                mode='linear',
                                                align_corners=False).permute(0, 3, 1, 2)  # [B, F, M, T] -> [B, T, F, M]
        
        # If original input was 1D, squeeze batch dimension
        if len(original_shape) == 1:
            x = x.squeeze(0)
        
        if self.return_stages:
            return x, stages
        else:
            return x
    
    def extra_repr(self) -> str:
        """Extra representation for printing.
        
        Returns
        -------
        str
            String representation of key parameters.
        """
        return (f"fs={self.fs}, "
                f"flow={self.flow:.1f}, fhigh={self.fhigh:.1f}, "
                f"num_channels={self.num_channels}, "
                f"compression={self.compression_type}, "
                f"dboffset={self.dboffset}")
