"""
Auditory Nerve Adaptation Loops
================================

Author: 
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements non-linear adaptation stages that model the dynamic response 
properties of auditory nerve fibers. The adaptation loops simulate the time-varying 
gain control mechanisms observed in neural responses to sustained stimuli.

The implementations follow the computational auditory signal processing (CASP) framework, 
primarily based on the Auditory Modeling Toolbox (AMT) for MATLAB/Octave. Multiple preset 
configurations are provided to match different published models (Dau et al. 1997, 
Osses et al. 2021, Paulick et al. 2024).

References
----------

.. [1] P. Majdak, C. Hollomey, and R. Baumgartner, "AMT 1.x: A toolbox for 
       reproducible research in auditory modeling," *Acta Acustica*, vol. 6, 
       p. 19, 2022, doi: 10.1051/aacus/2022011.

.. [2] P. Søndergaard and P. Majdak, "The Auditory Modeling Toolbox," in 
       *The Technology of Binaural Listening*, J. Blauert, Ed. 
       Berlin-Heidelberg, Germany: Springer, 2013, pp. 33-56, 
       doi: 10.1007/978-3-642-37762-4_2.
       
.. [3] P. Majdak et al., "The Auditory Modeling Toolbox 1.x Full Packages," 
       SourceForge, 2022. [Online]. Available: 
       https://sourceforge.net/projects/amtoolbox/files/AMT%201.x/
"""

from typing import Optional

import torch
import torch.nn as nn

# -------------------------------------------------- Utilities ----------------------------------------------------

@torch.jit.script
def _adapt_loop_core_jit(x: torch.Tensor,
                         init_state: torch.Tensor,
                         a1: torch.Tensor,
                         b0: torch.Tensor,
                         corr: torch.Tensor,
                         mult: torch.Tensor,
                         minlvl: torch.Tensor,
                         minlvl_per_channel: torch.Tensor,
                         use_freq_specific_minlvl: bool,
                         learnable: bool,
                         limit: float,
                         factor: torch.Tensor,
                         expfac: torch.Tensor,
                         offset: torch.Tensor) -> torch.Tensor:
    r"""
    JIT-compiled core adaptation loop.
    
    This function contains the time-sequential processing that cannot be
    vectorized due to state dependencies. JIT compilation reduces Python
    overhead for ~1.4x performance improvement.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape :math:`(B, F, T)`.
    
    init_state : torch.Tensor
        Initial state values, shape :math:`(\text{num_loops},)`.
    
    a1 : torch.Tensor
        RC filter pole coefficients, shape :math:`(\text{num_loops},)`.
    
    b0 : torch.Tensor
        RC filter zero coefficients, shape :math:`(\text{num_loops},)`.
    
    corr : torch.Tensor
        Output correction offset (scalar).
    
    mult : torch.Tensor
        Output scaling multiplier (scalar).
    
    minlvl : torch.Tensor
        Minimum level threshold (scalar).
    
    minlvl_per_channel : torch.Tensor
        Per-channel minimum levels, shape :math:`(F,)`.
    
    use_freq_specific_minlvl : bool
        Whether to use frequency-specific thresholds.
    
    learnable : bool
        Whether parameters are learnable (affects clamping).
    
    limit : float
        Overshoot limit factor.
    
    factor : torch.Tensor
        Overshoot limiting factor, shape :math:`(\text{num_loops},)`.
    
    expfac : torch.Tensor
        Overshoot limiting exponential factor, shape :math:`(\text{num_loops},)`.
    
    offset : torch.Tensor
        Overshoot limiting offset, shape :math:`(\text{num_loops},)`.
    
    Returns
    -------
    torch.Tensor
        Adapted output, shape :math:`(B, F, T)`.
    """
    batch_size = x.shape[0]
    num_channels = x.shape[1]
    siglen = x.shape[2]
    num_loops = a1.shape[0]
    
    # Initialize state [B, F, num_loops]
    state = init_state.unsqueeze(0).unsqueeze(0).expand(batch_size, num_channels, -1).clone()
    
    # Pre-allocate output
    output = torch.zeros_like(x)
    
    # Process sample by sample (sequential due to state dependency)
    for t in range(siglen):
        tmp = x[:, :, t].clone()
        
        # Clamp to minimum level (frequency-specific or scalar)
        if use_freq_specific_minlvl:
            if learnable:
                minlvl_clamped = torch.clamp(minlvl_per_channel, min=1e-10)
                minlvl_expanded = minlvl_clamped.unsqueeze(0)
            else:
                minlvl_expanded = minlvl_per_channel.unsqueeze(0)
            tmp = torch.maximum(tmp, minlvl_expanded)
        else:
            if learnable:
                minlvl_clamped = torch.clamp(minlvl, min=1e-10)
                tmp = torch.clamp(tmp, min=minlvl_clamped)
            else:
                tmp = torch.clamp(tmp, min=minlvl)
        
        # Apply adaptation loops
        for loop_idx in range(num_loops):
            # Divide by state
            tmp = tmp / state[:, :, loop_idx]
            
            # Overshoot limiting
            if limit > 1.0:
                mask = tmp > 1.0
                if mask.any():
                    limited = factor[loop_idx] / (1.0 + torch.exp(expfac[loop_idx] * (tmp - 1.0))) - offset[loop_idx]
                    tmp = torch.where(mask, limited, tmp)
            
            # Update state with RC lowpass
            new_state_value = a1[loop_idx] * state[:, :, loop_idx] + b0[loop_idx] * tmp
            state = state.clone()
            state[:, :, loop_idx] = new_state_value
        
        # Scale to model units
        output[:, :, t] = (tmp - corr) * mult
    
    return output

# ---------------------------------------------------- Main -------------------------------------------------------

class AdaptLoop(nn.Module):
    r"""
    Adaptation loops for auditory nerve fiber dynamics.
    
    Implements a cascade of non-linear adaptation loops that model the dynamic 
    response properties observed in auditory nerve fibers. Each loop consists of 
    a division normalization followed by a first-order RC lowpass filter, 
    simulating the time-varying gain control in neural responses.
    
    The adaptation process effectively implements automatic gain control (AGC) 
    that adjusts sensitivity based on recent stimulus history. This captures the 
    neural adaptation to sustained sounds, where initial strong responses decay 
    to lower sustained levels.
    
    Algorithm Overview
    ------------------
    For each time sample and adaptation loop :math:`k`:
    
    1. **Clamp input**: :math:`x(t) = \max(x(t), \text{minlvl})`
    2. **Division normalization**: :math:`y_k(t) = x(t) / s_k(t)`
    3. **Overshoot limiting** (if ``limit > 1``): 
       
       .. math::
           y_k(t) = \begin{cases}
               \frac{2m_k}{1 + \exp(-2(y_k - 1)/m_k)} - m_k - 1, & \text{if } y_k > 1 \\
               y_k, & \text{otherwise}
           \end{cases}
       
       where :math:`m_k = (1 - s_k^2(0)) \cdot \text{limit} - 1`
    
    4. **State update** (RC lowpass): 
       
       .. math::
           s_k(t) = a_k \cdot s_k(t-1) + b_k \cdot y_k(t)
       
       with :math:`a_k = \exp(-1/(\tau_k f_s))` and :math:`b_k = 1 - a_k`
    
    5. **Output scaling**: :math:`\text{out}(t) = (y_K(t) - c) \cdot m`
       where :math:`c = \text{minlvl}^{1/2^K}` and :math:`m = 100/(1-c)`
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    
    tau : torch.Tensor, optional
        Time constants for adaptation loops in seconds, shape ``(n_loops,)``. 
        The number of loops equals the length of ``tau``. 
        Default: ``[0.005, 0.050, 0.129, 0.253, 0.500]`` (Dau et al. 1997).
    
    limit : float, optional
        Overshoot limit factor. Values ``> 1`` enable limiting of rapid increases.
        Higher values allow more overshoot before compression. 
        Default: 10.0 (Dau et al. 1997). For Osses et al. (2021), use 5.0.
    
    minspl : float or torch.Tensor, optional
        Minimum SPL in dB re 100 dB SPL. Can be:
        
        - **Scalar**: Same threshold for all frequency channels.
        - **Tensor**: Per-channel thresholds, shape ``(n_channels,)``. 
          Used in Paulick et al. (2024) for frequency-dependent sensitivity.
        
        Default: 0.0 dB.
    
    minlvl_per_channel : torch.Tensor, optional
        Frequency-specific minimum levels as linear amplitudes, shape ``(n_channels,)``.
        Alternative to ``minspl`` when providing pre-computed linear values.
        Used by ``preset='paulick2024'``. If provided, overrides ``minspl``.
    
    preset : {'dau1997', 'osses2021', 'paulick2024'}, optional
        Configuration preset selecting published model parameters:
        
        - **'dau1997'**: Original CASP model (Dau et al. 1997)
          
          * ``limit = 10.0``
          * ``tau = [0.005, 0.050, 0.129, 0.253, 0.500]``
          * ``minspl = 0`` dB (scalar)
        
        - **'osses2021'**: Reduced overshoot (Osses et al. 2021)
          
          * ``limit = 5.0`` (half of dau1997, less compression)
          * ``tau = [0.005, 0.050, 0.129, 0.253, 0.500]``
          * ``minspl = 0`` dB (scalar)
        
        - **'paulick2024'**: Revised CASP model (Paulick et al. 2024)
          
          * ``limit = 10.0``
          * ``tau = [0.007, 0.0318, 0.0878, 0.2143, 0.5]`` (revised constants)
          * ``minlvl_per_channel``: 50 frequency-specific values (250-8000 Hz ERB-spaced)
        
        If ``preset`` is provided, it overrides ``limit`` and ``tau`` (unless 
        explicitly provided as arguments). Default: ``None`` (use provided parameters).
    
    learnable : bool, optional
        If ``True``, make adaptation parameters trainable:
        
        - RC filter coefficients: ``a1``, ``b0`` (time constants)
        - Overshoot parameters: ``limit`` (if ``limit > 1``)
        - Scaling factors: ``corr``, ``mult``
        
        Default: ``False`` (fixed parameters from literature).
    
    dtype : torch.dtype, optional
        Data type for computations. Default: ``torch.float32``.
    
    Attributes
    ----------
    num_loops : int
        Number of adaptation loops (length of ``tau``).
    
    fs : float
        Sampling rate in Hz.
    
    limit : float or torch.Tensor
        Overshoot limit factor (learnable if ``learnable=True`` and ``limit > 1``).
    
    use_freq_specific_minlvl : bool
        Whether frequency-specific minimum levels are used (``True`` for paulick2024).
    
    minlvl : torch.Tensor or nn.Parameter
        Scalar minimum level (linear amplitude). 
        Learnable if ``learnable=True`` and ``minspl`` is scalar.
        Clamped to minimum 1e-10 during forward pass for numerical stability.
    
    minlvl_per_channel : torch.Tensor or nn.Parameter
        Per-channel minimum levels (linear amplitude), shape ``(F,)`` where F is number 
        of frequency channels. 
        Learnable if ``learnable=True`` and ``minspl`` is vector or ``minlvl_per_channel`` 
        is provided.
        Clamped to minimum 1e-10 during forward pass for numerical stability.
    
    a1 : torch.Tensor
        RC lowpass pole coefficients, shape ``(num_loops,)``. 
        Learnable if ``learnable=True``.
    
    b0 : torch.Tensor
        RC lowpass zero coefficients, shape ``(num_loops,)``. 
        Learnable if ``learnable=True``.
    
    init_state : torch.Tensor
        Initial state values for adaptation loops, shape ``(num_loops,)``.
    
    corr : torch.Tensor
        Output correction offset (scalar). Learnable if ``learnable=True``.
    
    mult : torch.Tensor
        Output scaling multiplier (scalar). Learnable if ``learnable=True``.
    
    Shape
    -----
    - Input: :math:`(B, F, T)` or :math:`(F, T)` where
        * :math:`B` = batch size
        * :math:`F` = frequency channels
        * :math:`T` = time samples
    - Output: Same shape as input
    
    Notes
    -----
    **Preset Differences:**
    
    The three presets represent evolution of the CASP (Computational Auditory 
    Signal Processing) model:
    
    1. **dau1997**: Original formulation with strong overshoot limiting 
       (``limit=10``). Provides good detection predictions but may over-compress 
       rapid transients.
    
    2. **osses2021**: Reduced overshoot limiting (``limit=5``) after comparative 
       study showing better match to physiological data for some tasks.
    
    3. **paulick2024**: Most recent revision with:
       
       - Optimized time constants (``tau``) based on newer measurements
       - Frequency-dependent minimum thresholds (50 channels, 250-8000 Hz)
       - Improved predictions for complex stimuli and impaired hearing
    
    **Frequency-Specific Minimum Levels:**
    
    The ``paulick2024`` preset uses frequency-dependent thresholds to capture 
    different sensitivity across the auditory spectrum. Low frequencies have 
    higher thresholds (less sensitive), high frequencies have lower thresholds 
    (more sensitive), matching psychophysical absolute threshold curves.
    
    **Computational Complexity:**
    
    The current implementation processes samples sequentially (loop over time). 
    For typical configurations:
    
    - Time complexity: :math:`O(B \cdot F \cdot T \cdot K)` where :math:`K` is ``num_loops``
    - Memory: :math:`O(B \cdot F \cdot K)` for state storage
    
    Processing time scales linearly with signal duration. For long signals 
    (>10 seconds), consider chunking.
    
    **Connection to CASP Models:**
    
    The cascade structure effectively creates a multi-timescale adaptation mechanism:
    
    - Fast loops (``tau ~ 5 ms``): Respond to rapid amplitude changes
    - Slow loops (``tau ~ 500 ms``): Track longer-term level variations
    
    See Also
    --------
    IHCEnvelope : Inner hair cell envelope extraction (preprocessing)
    ModulationFilterbank : Modulation filterbank (post-processing)
    GammatoneFilterbank : Gammatone auditory filterbank (frontend)
    DRNLFilterbank : Dual resonance nonlinear filterbank (frontend)
    HeadphoneFilter : Headphone/outer ear frequency response
    MiddleEarFilter : Middle ear transfer function
    
    Examples
    --------
    **Basic usage with default parameters (Dau et al. 1997):**
    
    >>> import torch
    >>> from torch_amt.common.adaptation import AdaptLoop
    >>> 
    >>> # Random filterbank output (2 batches, 31 channels, 44100 samples = 1 sec @ 44.1 kHz)
    >>> x = torch.randn(2, 31, 44100)
    >>> 
    >>> adapt = AdaptLoop(fs=44100)
    >>> y = adapt(x)
    >>> 
    >>> print(f"Input shape: {x.shape}")
    Input shape: torch.Size([2, 31, 44100])
    >>> print(f"Output shape: {y.shape}")
    Output shape: torch.Size([2, 31, 44100])
    >>> print(f"Num loops: {adapt.num_loops}")
    Num loops: 5
    
    **Using preset configurations:**
    
    >>> # Osses et al. (2021): Reduced overshoot limiting
    >>> adapt_osses = AdaptLoop(fs=16000, preset='osses2021')
    >>> print(f"Limit: {adapt_osses.limit:.1f}")
    Limit: 5.0
    >>> 
    >>> # Paulick et al. (2024): Frequency-specific thresholds (requires 50 channels)
    >>> x_50ch = torch.randn(1, 50, 16000)
    >>> adapt_paulick = AdaptLoop(fs=16000, preset='paulick2024')
    >>> y_paulick = adapt_paulick(x_50ch)
    >>> print(f"Freq-specific minlvl: {adapt_paulick.use_freq_specific_minlvl}")
    Freq-specific minlvl: True
    
    **Learnable adaptation for neural network training:**
    
    >>> # Make time constants trainable
    >>> adapt_learn = AdaptLoop(fs=16000, learnable=True)
    >>> 
    >>> # Check trainable parameters
    >>> for name, param in adapt_learn.named_parameters():
    ...     print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
    a1: shape=torch.Size([5]), requires_grad=True
    b0: shape=torch.Size([5]), requires_grad=True
    limit: shape=torch.Size([]), requires_grad=True
    corr: shape=torch.Size([]), requires_grad=True
    mult: shape=torch.Size([]), requires_grad=True
    
    References
    ----------
    .. [1] T. Dau, D. Püschel, and A. Kohlrausch, "A quantitative model of the 
       'effective' signal processing in the auditory system. I. Model structure," 
       *J. Acoust. Soc. Am.*, vol. 99, no. 6, pp. 3615-3622, 1996.

    .. [2] T. Dau, B. Kollmeier, and A. Kohlrausch, "Modeling auditory processing 
        of amplitude modulation. I. Detection and masking with narrow-band carriers," 
        *J. Acoust. Soc. Am.*, vol. 102, no. 5, pp. 2892-2905, 1997.

    .. [3] D. Püschel, "Prinzipien der zeitlichen Analyse beim Hören," 
        Ph.D. dissertation, Universität Göttingen, Germany, 1988.

    .. [4] A. Osses, L. Varnet, L. M. A. Carney, T. Dau, et al., "A comparative 
        study of eight human auditory models of monaural processing," 
        *Acta Acust.*, vol. 6, p. 17, 2022.

    .. [5] L. Paulick, H. Relaño-Iborra, and T. Dau, "The Computational Auditory 
        Signal Processing and Perception Model (CASP): A Revised Version," bioRxiv, 2024.
    """
    
    def __init__(self,
                 fs: float,
                 tau: Optional[torch.Tensor] = None,
                 limit: float = 10.0,
                 minspl: Optional[torch.Tensor] = 0.0,
                 minlvl_per_channel: Optional[torch.Tensor] = None,
                 preset: Optional[str] = None,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        self.fs = fs
        self.dtype = dtype
        self.learnable = learnable
        self.preset = preset
        
        # Apply preset configuration if specified
        if preset is not None:
            if preset == 'dau1997':
                limit = 10.0
                if tau is None:
                    tau = torch.tensor([0.005, 0.050, 0.129, 0.253, 0.500], dtype=dtype)
            elif preset == 'osses2021':
                limit = 5.0
                if tau is None:
                    tau = torch.tensor([0.005, 0.050, 0.129, 0.253, 0.500], dtype=dtype)
            elif preset == 'paulick2024':
                limit = 10.0
                if tau is None:
                    tau = torch.tensor([0.007, 0.0318, 0.0878, 0.2143, 0.5], dtype=dtype)
                # Frequency-specific minimum levels (50 channels, 250-8000 Hz ERB-spaced)
                if minlvl_per_channel is None:
                    minlvl_per_channel = torch.tensor([
                        114.6010e-6, 112.6388e-6, 110.5593e-6, 115.6323e-6, 122.8003e-6,
                        128.3473e-6, 129.4167e-6, 130.5499e-6, 134.9603e-6, 140.6641e-6,
                        146.7089e-6, 147.9811e-6, 148.9594e-6, 149.9962e-6, 142.9728e-6,
                        133.7308e-6, 123.9364e-6, 116.7957e-6, 110.2674e-6, 103.3490e-6,
                         96.4875e-6,  89.5786e-6,  82.2569e-6,  74.4976e-6,  66.1913e-6,
                         57.3802e-6,  48.0427e-6,  42.4791e-6,  40.9715e-6,  39.3738e-6,
                         37.6163e-6,  31.9073e-6,  25.8572e-6,  19.4456e-6,  14.2276e-6,
                         11.9080e-6,   9.4499e-6,   6.8449e-6,   5.6897e-6,   6.3316e-6,
                          7.0118e-6,   7.7326e-6,   7.1594e-6,   6.2346e-6,   5.2545e-6,
                          4.2159e-6,   3.5157e-6,   2.7790e-6,   1.9983e-6,   1.1709e-6
                    ], dtype=dtype)
            else:
                raise ValueError(f"Unknown preset '{preset}'. Choose from: 'dau1997', 'osses2021', 'paulick2024'")
        
        # Store limit as tensor for learnability
        limit_tensor = torch.tensor(limit, dtype=dtype) if not isinstance(limit, torch.Tensor) else limit.to(dtype=dtype)
        
        # Default time constants (Dau 1997)
        if tau is None:
            tau = torch.tensor([0.005, 0.050, 0.129, 0.253, 0.500], dtype=dtype)
        else:
            tau = tau.to(dtype=dtype)
        
        self.num_loops = len(tau)
        # Store limit as tensor for learnability
        limit_tensor = torch.tensor(limit, dtype=dtype) if not isinstance(limit, torch.Tensor) else limit.to(dtype=dtype)
        
        # Default time constants (Dau 1997)
        if tau is None:
            tau = torch.tensor([0.005, 0.050, 0.129, 0.253, 0.500], dtype=dtype)
        else:
            tau = tau.to(dtype=dtype)
        
        self.num_loops = len(tau)
        
        # Handle frequency-specific or scalar minimum level
        if minlvl_per_channel is not None:
            # Frequency-specific minlvl (paulick2024)
            if learnable:
                self.minlvl_per_channel = nn.Parameter(minlvl_per_channel.to(dtype=dtype))
            else:
                self.register_buffer('minlvl_per_channel', minlvl_per_channel.to(dtype=dtype))
            self.use_freq_specific_minlvl = True
            # Use first channel's minlvl for init_state computation
            minlvl_scalar = minlvl_per_channel[0].item()
            # Register dummy minlvl (not used but needed for forward compatibility)
            self.register_buffer('minlvl', torch.tensor(minlvl_scalar, dtype=dtype))
        else:
            # Scalar or per-channel minspl (dau1997, osses2021)
            # Convert minspl (dB re 100 dB) to linear amplitude
            # From MATLAB adaptloop.m: minlvl_lin=scaletodbspl(minspl,[],100)
            # Which with dboffset=100 gives: gaindb(1, minspl - 100) = 10^((minspl - 100)/20)
            if isinstance(minspl, (int, float)):
                # Scalar minspl
                minlvl = 10.0 ** ((minspl - 100.0) / 20.0)
                if learnable:
                    self.minlvl = nn.Parameter(torch.tensor(minlvl, dtype=dtype))
                else:
                    self.register_buffer('minlvl', torch.tensor(minlvl, dtype=dtype))
                self.use_freq_specific_minlvl = False
                minlvl_scalar = minlvl
            else:
                # Vector minspl (per-channel)
                minspl_tensor = minspl if isinstance(minspl, torch.Tensor) else torch.tensor(minspl, dtype=dtype)
                minlvl_vec = 10.0 ** ((minspl_tensor - 100.0) / 20.0)
                if learnable:
                    self.minlvl_per_channel = nn.Parameter(minlvl_vec)
                else:
                    self.register_buffer('minlvl_per_channel', minlvl_vec)
                self.use_freq_specific_minlvl = True
                minlvl_scalar = minlvl_vec[0].item()
        
        # Compute RC lowpass coefficients: y(n) = b0*x(n) + a1*y(n-1)
        a1 = torch.exp(-1.0 / (tau * fs))
        b0 = 1.0 - a1
        
        if learnable:
            self.a1 = nn.Parameter(a1)
            self.b0 = nn.Parameter(b0)
        else:
            self.register_buffer('a1', a1)
            self.register_buffer('b0', b0)
        
        # Initialize state values (sqrt of minlvl for first, then sqrt of previous)
        # Use minlvl_scalar for init_state computation
        init_state = torch.zeros(self.num_loops, dtype=dtype)
        init_state[0] = torch.sqrt(torch.tensor(minlvl_scalar, dtype=dtype))
        for i in range(1, self.num_loops):
            init_state[i] = torch.sqrt(init_state[i - 1])
        
        self.register_buffer('init_state', init_state)
        
        # Compute overshoot limiting constants (make learnable if requested)
        if limit_tensor > 1.0:
            maxvalue = (1.0 - init_state * init_state) * limit_tensor - 1.0
            factor = maxvalue * 2.0
            expfac = -2.0 / maxvalue
            offset = maxvalue - 1.0
            
            if learnable:
                self.limit = nn.Parameter(limit_tensor)
                self.register_buffer('factor_init', factor)
                self.register_buffer('expfac_init', expfac)
                self.register_buffer('offset_init', offset)
            else:
                self.register_buffer('limit', limit_tensor)
                self.register_buffer('factor', factor)
                self.register_buffer('expfac', expfac)
                self.register_buffer('offset', offset)
        else:
            if learnable:
                self.limit = nn.Parameter(limit_tensor)
            else:
                self.register_buffer('limit', limit_tensor)
        
        # Scaling constants (make learnable if requested)
        corr = init_state[-1]
        mult = 100.0 / (1.0 - corr)
        
        if learnable:
            self.corr = nn.Parameter(corr)
            self.mult = nn.Parameter(mult)
        else:
            self.register_buffer('corr', corr)
            self.register_buffer('mult', mult)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply adaptation loops to input signal.
        
        Processes the input through cascaded adaptation loops with division 
        normalization, overshoot limiting, and RC lowpass state updates.
        
        Uses JIT-compiled core loop for ~1.4x performance improvement while
        maintaining numerical accuracy and gradient flow.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape: :math:`(B, F, T)` or :math:`(F, T)` where
            
            * :math:`B` = batch size
            * :math:`F` = frequency channels
            * :math:`T` = time samples
        
        Returns
        -------
        torch.Tensor
            Adapted signal. Same shape as input. Values scaled to 0-100 model units.
        
        Notes
        -----
        The forward pass processes samples sequentially to maintain state consistency 
        across the time dimension. For batch processing, all batches and channels 
        are processed in parallel at each time step.
        """
        # Handle input shape
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Recompute overshoot params if limit is learnable and > 1
        if self.learnable and self.limit > 1.0:
            maxvalue = (1.0 - self.init_state * self.init_state) * self.limit - 1.0
            factor = maxvalue * 2.0
            expfac = -2.0 / maxvalue
            offset = maxvalue - 1.0
        elif self.limit > 1.0:
            factor = self.factor
            expfac = self.expfac
            offset = self.offset
        else:
            # Dummy values when limit <= 1
            factor = torch.zeros(self.num_loops, dtype=self.a1.dtype, device=self.a1.device)
            expfac = torch.zeros(self.num_loops, dtype=self.a1.dtype, device=self.a1.device)
            offset = torch.zeros(self.num_loops, dtype=self.a1.dtype, device=self.a1.device)
        
        # Call JIT-compiled core loop
        output = _adapt_loop_core_jit(x,
                                      self.init_state,
                                      self.a1,
                                      self.b0,
                                      self.corr,
                                      self.mult,
                                      self.minlvl,
                                      self.minlvl_per_channel if self.use_freq_specific_minlvl else torch.tensor(0.0, device=x.device),
                                      self.use_freq_specific_minlvl,
                                      self.learnable,
                                      self.limit if isinstance(self.limit, float) else self.limit.item(),
                                      factor,
                                      expfac,
                                      offset)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters in MINIMAL format.
        """
        # Compute tau range
        tau_min = self.a1.min().item() if isinstance(self.a1, nn.Parameter) else self.a1.min().item()
        tau_max = self.a1.max().item() if isinstance(self.a1, nn.Parameter) else self.a1.max().item()
        # Inverse formula: tau = -1 / (fs * log(a1))
        tau_min_sec = -1.0 / (self.fs * torch.log(torch.tensor(tau_max)).item())
        tau_max_sec = -1.0 / (self.fs * torch.log(torch.tensor(tau_min)).item())
        
        preset_str = f", preset={self.preset}" if self.preset is not None else ""
        limit_val = self.limit.item() if isinstance(self.limit, torch.Tensor) else self.limit
        
        return (f"fs={self.fs}, num_loops={self.num_loops}, "
                f"tau_range=[{tau_min_sec:.3f}, {tau_max_sec:.3f}] s, "
                f"limit={limit_val:.1f}{preset_str}, learnable={self.learnable}")
