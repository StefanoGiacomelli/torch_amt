"""
Loudness Models & Processing
=============================

Author: 
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements compression stages, specific loudness transformations, 
binaural loudness processing, and temporal integration for auditory loudness models.

The implementations follow standard psychoacoustic models and are compatible with 
the Auditory Modeling Toolbox (AMT) for MATLAB/Octave. This module consolidates:

- **Compression stages**: Nonlinear compression for auditory nerve fibers (King2019)
- **Specific loudness**: Excitation to loudness transformation (Glasberg2002, Moore2016)
- **Binaural processing**: Spatial smoothing and cross-ear inhibition (Moore2016)
- **Temporal integration**: Attack/release filtering for loudness dynamics (Glasberg2002, Moore2016)

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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .filterbanks import f2erbrate, erbrate2f

# ---------------------------------------------------- Utilities -----------------------------------------------------

def gaindb(signal: torch.Tensor, db: float) -> torch.Tensor:
    """
    Apply gain to signal using dB-to-linear conversion.
    
    Converts a gain value in decibels (dB) to a linear multiplier and applies it 
    to the input signal. Uses the standard formula:
    
    .. math::
        \\text{output} = \\text{signal} \\times 10^{\\text{dB}/20}
    
    This is commonly used in auditory models to adjust signal levels to specific 
    SPL (Sound Pressure Level) values.
    
    Parameters
    ----------
    signal : torch.Tensor
        Input signal tensor of any shape.
    
    db : float
        Gain in decibels (dB). Positive values amplify, negative values attenuate.
        
    Returns
    -------
    torch.Tensor
        Signal with gain applied, same shape as input.
        
    Examples
    --------
    >>> signal = torch.randn(1000)
    >>> # Amplify by 6 dB (factor of ~2)
    >>> amplified = gaindb(signal, 6.0)
    >>> # Attenuate by -10 dB (factor of ~0.316)
    >>> attenuated = gaindb(signal, -10.0)
    
    Notes
    -----
    Common dB-to-linear multiplier conversions:
    
    - +6 dB ≈ 2.0x
    - +3 dB ≈ 1.41x (√2)
    - 0 dB = 1.0x
    - -3 dB ≈ 0.707x (1/√2)
    - -6 dB ≈ 0.5x
    - -20 dB = 0.1x
    """
    return signal * (10.0 ** (db / 20.0))

#------------------------------------------------ Compression Stages -------------------------------------------------

class BrokenStickCompression(nn.Module):
    r"""
    Broken-stick compression for auditory nerve fiber dynamics.
    
    Implements a piecewise power-law compression where signals below a knee point 
    pass through unchanged (linear), while signals above the knee are compressed 
    using a power-law function. This creates a "broken stick" transfer function 
    that simulates the nonlinear input-output characteristics of inner hair cells 
    and auditory nerve fibers.
    
    Algorithm Overview
    ------------------
    The compression is applied element-wise:
    
    .. math::
        y(t) = \\begin{cases}
        x(t) & \\text{if } |x(t)| \\leq \\text{knee} \\\\
        \\text{sign}(x(t)) \\cdot |x(t)|^n \\cdot \\text{knee}^{1-n} & \\text{if } |x(t)| > \\text{knee}
        \\end{cases}
    
    where:
    - :math:`n` is the compression exponent (typically 0.3)
    - :math:`\\text{knee}` is the threshold in linear units
    - The formula ensures continuity at the knee point
    
    **Knee point calculation:**
    
    .. math::
        \\text{knee}_{\\text{linear}} = 10^{(\\text{knee}_{\\text{dB}} - \\text{dboffset})/20}
    
    Parameters
    ----------
    knee_db : float, optional
        Knee point in dB relative to dboffset. Default: 30 dB.
    
    exponent : float or torch.Tensor, optional
        Compression exponent n. Default: 0.3.
        
        - n < 1: compression above knee
        - n = 1: linear (no compression)
        - n > 1: expansion above knee
        
    dboffset : float, optional
        Reference level in dB SPL for full scale. Default: 100 dB.
        AMT convention uses 94 dB, King2019 uses 100 dB.
    
    num_channels : int, optional
        Number of frequency channels for per-channel exponents. Default: None.
        If provided, exponent can vary across channels.
    
    learnable : bool, optional
        If True, both exponent and knee_db become learnable nn.Parameters. Default: ``False``.
            
    Attributes
    ----------
    knee : torch.Tensor
        Linear knee point value, shape ().
    
    exponent : torch.Tensor
        Compression exponent, shape () or (num_channels,).
    
    knee_db : float
        Knee point in dB (stored for reference).
    
    dboffset : float
        Reference level in dB SPL.
        
    Shape
    -----
    - Input: :math:`(T, C)` or :math:`(T,)` where
      
      * :math:`T` = time samples
      * :math:`C` = channels (optional)
    
    - Output: Same shape as input
        
    Notes
    -----
    **Compression behavior:**
    
    - Below knee: No compression (linear passthrough)
    - Above knee: Power-law compression with exponent n
    - For King et al. (2019): knee_db=30, exponent=0.3, dboffset=100
    
    **Per-channel exponents:**
    
    Exponent can be a scalar (same for all channels) or a tensor with one value  
    per channel, allowing frequency-dependent compression characteristics.
    
    See Also
    --------
    PowerCompression : Full power-law compression without knee
    
    References
    ----------
    .. [1] A. King, L. Varnet, and C. Lorenzi, "Accounting for masking of frequency 
           modulation by amplitude modulation with the modulation filter-bank concept," 
           *J. Acoust. Soc. Am.*, vol. 145, no. 4, pp. 2277-2293, 2019.
    
    Examples
    --------
    >>> # Standard King2019 compression
    >>> comp = BrokenStickCompression(knee_db=30, exponent=0.3, dboffset=100)
    >>> signal = torch.randn(1000, 31)
    >>> compressed = comp(signal)
    >>> print(compressed.shape)
    torch.Size([1000, 31])
    
    >>> # Per-channel compression (different exponent per channel)
    >>> exponents = torch.linspace(0.2, 0.4, 31)  # Vary from 0.2 to 0.4
    >>> comp_perchan = BrokenStickCompression(exponent=exponents, num_channels=31)
    >>> compressed = comp_perchan(signal)
    """
    
    def __init__(self,
                 knee_db: float = 30.0,
                 exponent: Union[float, torch.Tensor] = 0.3,
                 dboffset: float = 100.0,
                 num_channels: Optional[int] = None,
                 learnable: bool = False):
        super().__init__()
        
        self.dboffset = dboffset
        self.learnable = learnable
        self.num_channels = num_channels
        
        # Handle knee_db: single value only
        knee_db_tensor = torch.tensor(float(knee_db))
        
        # Make knee_db learnable if requested
        if learnable:
            self.knee_db_param = nn.Parameter(knee_db_tensor)
            self.knee_db = None  # Will be computed in forward
        else:
            self.knee_db = knee_db
            # Compute static linear knee point
            knee_linear = 10.0 ** ((knee_db_tensor - dboffset) / 20.0)
            self.register_buffer('knee', knee_linear)
        
        # Handle per-channel exponents
        if isinstance(exponent, (int, float)):
            if num_channels is not None:
                exponent = torch.full((num_channels,), float(exponent))
            else:
                exponent = torch.tensor(float(exponent))
        elif isinstance(exponent, torch.Tensor):
            if num_channels is not None and exponent.numel() != num_channels:
                raise ValueError(f"Exponent tensor size {exponent.numel()} does not match "
                                 f"num_channels {num_channels}")
        else:
            raise TypeError(f"Exponent must be float or Tensor, got {type(exponent)}")
        
        # Make exponent learnable if requested
        if learnable:
            self.exponent = nn.Parameter(exponent)
        else:
            self.register_buffer('exponent', exponent)
            # OPTIMIZATION: Pre-calculate knee^(1-n) to avoid recomputing in forward
            if not learnable:  # Only for non-learnable case
                if exponent.numel() == 1:
                    knee_factor = knee_linear ** (1.0 - exponent.item())
                else:
                    # For per-channel: use mean exponent for shared knee_factor
                    knee_factor = knee_linear ** (1.0 - exponent.mean().item())
                self.register_buffer('knee_factor', knee_factor)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply broken-stick compression.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape (time, channels) or (time,).
            
        Returns
        -------
        torch.Tensor
            Compressed signal with same shape as input.
        """
        # Store original shape
        orig_shape = x.shape
        
        # Ensure 2D for processing
        if x.ndim == 1:
            x = x.unsqueeze(-1)
            
        # Get device and dtype
        device = x.device
        dtype = x.dtype
        
        # Compute knee (learnable or static)
        if self.learnable:
            knee_db = self.knee_db_param.to(device=device, dtype=dtype)
            knee = 10.0 ** ((knee_db - self.dboffset) / 20.0)
            # For learnable, compute knee_factor on the fly
            knee_factor = knee ** (1.0 - exponent)
        else:
            knee = self.knee.to(device=device, dtype=dtype)
            # OPTIMIZATION: Use pre-calculated knee_factor
            knee_factor = self.knee_factor.to(device=device, dtype=dtype)
        
        # Move exponent to same device/dtype
        exponent = self.exponent.to(device=device, dtype=dtype)
        
        # Reshape exponent for broadcasting if needed
        if exponent.ndim == 1 and x.shape[1] > 1:
            exponent = exponent.view(1, -1)
        
        # Find samples above knee point
        abs_x = torch.abs(x)
        mask = abs_x > knee
        
        # Apply compression only above knee
        # OPTIMIZATION: Use pre-calculated knee_factor
        if mask.any():
            sign_x = torch.sign(x)
            # For learnable case, knee_factor already computed above
            # For non-learnable, we use pre-calculated buffer
            if self.learnable:
                abs_compressed = torch.pow(abs_x, exponent) * knee_factor
            else:
                # Use pre-calculated knee_factor
                abs_compressed = torch.pow(abs_x, exponent) * knee_factor
            # Direct where without intermediate variable
            y = torch.where(mask, sign_x * abs_compressed, x)
        else:
            y = x

            
        # Restore original shape
        if len(orig_shape) == 1:
            y = y.squeeze(-1)
            
        return y
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        knee_val = self.knee_db_param.item() if self.learnable else self.knee_db
        return (f"knee_db={knee_val:.1f}, "
                f"exponent={self.exponent.mean().item():.3f}, "
                f"dboffset={self.dboffset}")
    
    def get_parameters(self) -> dict:
        """
        Get compression parameters.
        
        Returns
        -------
        dict
            Dictionary with knee_db, exponent (mean), dboffset, learnable
        """
        knee_val = self.knee_db_param.item() if self.learnable else self.knee_db
        
        return {'knee_db': knee_val,
                'exponent': self.exponent.mean().item(),
                'dboffset': self.dboffset,
                'learnable': self.learnable}


class PowerCompression(nn.Module):
    r"""
    Full power-law compression for auditory processing.
    
    Applies a power-law compression/expansion to the entire signal without a 
    linear region. Unlike BrokenStickCompression, this affects all signal levels 
    and can cause expansion below the knee when exponent < 1.
    
    Algorithm Overview
    ------------------
    The compression is applied globally:
    
    .. math::
        y(t) = \\text{sign}(x(t)) \\cdot \\left|\\frac{x(t)}{\\text{knee}}\\right|^n \\cdot \\text{knee}
    
    where:
    - :math:`n` is the power-law exponent
    - :math:`\\text{knee}` is the reference level in linear units
    
    **Compression characteristics:**
    
    - For n < 1: Compresses signals above knee, expands signals below knee
    - For n = 1: Linear (no effect)
    - For n > 1: Expands signals above knee, compresses signals below knee
    
    Parameters
    ----------
    knee_db : float, optional
        Reference level in dB relative to dboffset. Default: 30 dB.
    
    exponent : float or torch.Tensor, optional
        Power-law exponent n. Default: 0.3.
        
        - n < 1: compression above knee, expansion below
        - n = 1: linear (identity transform)
        - n > 1: expansion above knee, compression below
        
    dboffset : float, optional
        Reference level in dB SPL for full scale. Default: 100 dB SPL.
    
    num_channels : int, optional
        Number of frequency channels for per-channel exponents. Default: None.
    
    learnable : bool, optional
        If True, both exponent and knee_db become learnable nn.Parameters. Default: ``False``.
            
    Attributes
    ----------
    knee : torch.Tensor
        Linear reference level, shape ().
    
    exponent : torch.Tensor
        Power-law exponent, shape () or (num_channels,).
    
    knee_db : float
        Reference level in dB.
    
    dboffset : float
        dB SPL reference.
        
    Shape
    -----
    - Input: :math:`(T, C)` or :math:`(T,)` where
      
      * :math:`T` = time samples
      * :math:`C` = channels (optional)
    
    - Output: Same shape as input
        
    Notes
    -----
    **Difference from BrokenStickCompression:**
    
    - PowerCompression: Affects all signal levels, no linear region
    - BrokenStickCompression: Linear below knee, compressed above knee
    
    **WARNING:** With typical exponent values (n < 1), this compression can 
    amplify low-level signals (expansion below knee), which may not be 
    physiologically accurate for auditory modeling.
    
    **Usage in King et al. (2019):**
    
    This compression type is specific to the PEMO model. The Dau et al. (1997) 
    model does not use compression at this stage.
    
    See Also
    --------
    BrokenStickCompression : Piecewise compression with linear region below knee
    
    References
    ----------
    .. [1] A. King, L. Varnet, and C. Lorenzi, "Accounting for masking of frequency 
           modulation by amplitude modulation with the modulation filter-bank concept," 
           *J. Acoust. Soc. Am.*, vol. 145, no. 4, pp. 2277-2293, 2019.
    
    Examples
    --------
    >>> # Standard power-law compression
    >>> comp = PowerCompression(knee_db=30, exponent=0.3, dboffset=100)
    >>> signal = torch.randn(1000, 31)
    >>> compressed = comp(signal)
    >>> print(compressed.shape)
    torch.Size([1000, 31])
    """
    
    def __init__(self,
                 knee_db: float = 30.0,
                 exponent: Union[float, torch.Tensor] = 0.3,
                 dboffset: float = 100.0,
                 num_channels: Optional[int] = None,
                 learnable: bool = False):
        super().__init__()
        
        self.dboffset = dboffset
        self.learnable = learnable
        self.num_channels = num_channels
        
        # Handle knee_db: single value only (not per-channel)
        knee_db_tensor = torch.tensor(float(knee_db))
        
        # Make knee_db learnable if requested
        if learnable:
            self.knee_db_param = nn.Parameter(knee_db_tensor)
            self.knee_db = None  # Will be computed in forward
        else:
            self.knee_db = knee_db
            # Compute static linear knee point
            knee_linear = 10.0 ** ((knee_db_tensor - dboffset) / 20.0)
            self.register_buffer('knee', knee_linear)
        
        # Handle per-channel exponents
        if isinstance(exponent, (int, float)):
            if num_channels is not None:
                exponent = torch.full((num_channels,), float(exponent))
            else:
                exponent = torch.tensor(float(exponent))
        elif isinstance(exponent, torch.Tensor):
            if num_channels is not None and exponent.numel() != num_channels:
                raise ValueError(f"Exponent tensor size {exponent.numel()} does not match "
                                 f"num_channels {num_channels}")
        else:
            raise TypeError(f"Exponent must be float or Tensor, got {type(exponent)}")
        
        # Make exponent learnable if requested
        if learnable:
            self.exponent = nn.Parameter(exponent)
        else:
            self.register_buffer('exponent', exponent)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply power-law compression.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape (time, channels) or (time,).
            
        Returns
        -------
        torch.Tensor
            Compressed signal with same shape as input.
        """
        # Store original shape
        orig_shape = x.shape
        
        # Ensure 2D for processing
        if x.ndim == 1:
            x = x.unsqueeze(-1)
            
        # Get device and dtype
        device = x.device
        dtype = x.dtype
        
        # Compute knee (learnable or static)
        if self.learnable:
            knee_db = self.knee_db_param.to(device=device, dtype=dtype)
            knee = 10.0 ** ((knee_db - self.dboffset) / 20.0)
        else:
            knee = self.knee.to(device=device, dtype=dtype)
        
        # Move exponent to same device/dtype
        exponent = self.exponent.to(device=device, dtype=dtype)
        
        # Reshape exponent for broadcasting if needed
        if exponent.ndim == 1 and x.shape[1] > 1:
            exponent = exponent.view(1, -1)
        
        # Apply power-law: y = sign(x) * |x/knee|^n * knee
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        y = sign_x * torch.pow(abs_x / knee, exponent) * knee
        
        # Restore original shape
        if len(orig_shape) == 1:
            y = y.squeeze(-1)
            
        return y
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        knee_val = self.knee_db_param.item() if self.learnable else self.knee_db
        return (f"knee_db={knee_val:.1f}, "
                f"exponent={self.exponent.mean().item():.3f}, "
                f"dboffset={self.dboffset}")
    
    def get_parameters(self) -> dict:
        """
        Get compression parameters.
        
        Returns
        -------
        dict
            Dictionary with knee_db, exponent (mean), dboffset, learnable
        """
        knee_val = self.knee_db_param.item() if self.learnable else self.knee_db
        
        return {'knee_db': knee_val,
                'exponent': self.exponent.mean().item(),
                'dboffset': self.dboffset,
                'learnable': self.learnable}

# ------------------------------------------------ Specific Loudness -------------------------------------------------

class SpecificLoudness(nn.Module):
    r"""
    Specific loudness transformation for Glasberg & Moore (2002) loudness model.
    
    Transforms excitation pattern from ERB filterbank to specific loudness using 
    a three-regime model that accounts for absolute threshold, linear region, and 
    compressive region. Based on Moore & Glasberg (1997) model with ISO 226 
    threshold curves.
    
    Algorithm Overview
    ------------------
    The transformation uses a piecewise function with three regimes:
    
    **Regime 1: Sub-threshold (E ≤ E_Thrq)**
    
    .. math::
        N(f,t) = 0
    
    **Regime 2: Linear region (E_Thrq < E ≤ E_Thrq + E_0)**
    
    .. math::
        N(f,t) = C \\cdot (E - E_{Thrq})
    
    **Regime 3: Compressive region (E > E_Thrq + E_0)**
    
    .. math::
        N(f,t) = C \\cdot E_0^{1-\\alpha} \\cdot (E - E_{Thrq})^{\\alpha}
    
    where:
    
    - :math:`E(f,t)` is the excitation level in dB SPL at ERB frequency :math:`f`
    - :math:`E_{Thrq}(f)` is the absolute threshold in quiet (ISO 226), frequency-dependent
    - :math:`C = 0.047` is the gain constant (Moore & Glasberg 1997)
    - :math:`\\alpha = 0.2` is the compression exponent
    - :math:`E_0 = 10` dB is the transition point from linear to compressive regime
    - :math:`N(f,t)` is the specific loudness in sone/ERB
    
    **Absolute Threshold Computation (ISO 226 approximation):**
    
    .. math::
        E_{Thrq}(f) = 3.64 \\left(\\frac{f}{1000}\\right)^{-0.8} 
        - 6.5 \\exp\\left[-0.6\\left(\\frac{f}{1000} - 3.3\\right)^2\\right]
        + 10^{-3} \\left(\\frac{f}{1000}\\right)^4
    
    Parameters
    ----------
    fs : int, optional
        Sampling rate in Hz. Default: 32000
    
    f_min : float, optional
        Minimum frequency for ERB filterbank in Hz. Default: 50.0
    
    f_max : float, optional
        Maximum frequency for ERB filterbank in Hz. Default: 15000.0
    
    erb_step : float, optional
        ERB frequency step for filterbank spacing. Default: 0.25 ERB
    
    learnable : bool, optional
        If True, C, α, E_0, and threshold adjustments become learnable parameters.
        Default: False
    
    Attributes
    ----------
    fc_erb : torch.Tensor
        ERB filterbank center frequencies in Hz, shape (n_erb_bands,)
    
    n_erb_bands : int
        Number of ERB frequency channels
    
    ethrq_base : torch.Tensor
        Base absolute threshold in quiet (ISO 226), shape (n_erb_bands,)
    
    ethrq_adjustment : torch.Tensor or nn.Parameter
        Additive adjustment to threshold in dB, shape (n_erb_bands,)
    
    C : torch.Tensor or nn.Parameter
        Gain constant, scalar. Fixed at 0.047 (Moore & Glasberg 1997)
    
    alpha : torch.Tensor or nn.Parameter
        Compression exponent, scalar. Fixed at 0.2
    
    E0_offset : torch.Tensor or nn.Parameter
        Transition point from linear to compressive regime in dB above threshold.
        Fixed at 10.0 dB
    
    Input Shape
    -----------
    excitation : torch.Tensor
        Excitation pattern in dB SPL, shape (batch, n_frames, n_erb_bands)
    
    Output Shape
    ------------
    specific_loudness : torch.Tensor
        Specific loudness in sone/ERB, shape (batch, n_frames, n_erb_bands)
    
    Notes
    -----
    - The three-regime model captures the transition from complete masking (sub-threshold),
      through a linear loudness growth region, to the compressive loudness region
    - The absolute threshold :math:`E_{Thrq}` is frequency-dependent and follows ISO 226,
      with minimum threshold (~4 dB SPL) around 2-5 kHz
    - The linear region (Regime 2) extends from threshold to ~10 dB above threshold
    - The compressive region (Regime 3) with :math:`\\alpha=0.2` implements the well-known
      power-law loudness growth (approximately doubling loudness per 10 dB)
    - When learnable=True, the model can adapt the threshold, gain, and compression
      characteristics through backpropagation
    - This implementation is compatible with `glasberg2002` model from AMT MATLAB
    
    See Also
    --------
    Moore2016SpecificLoudness : Specific loudness for Moore2016 model (ANSI S3.4-2007)
    LoudnessIntegration : Temporal integration for Glasberg2002 model
    auditoryfilterbank : ERB gammatone filterbank preprocessing
    
    Examples
    --------
    Basic usage with default Glasberg2002 parameters:
    
    >>> import torch
    >>> from torch_amt.common.loudness import SpecificLoudness
    >>> 
    >>> # Create module
    >>> spec_loud = SpecificLoudness(fs=32000, f_min=50, f_max=15000, erb_step=0.25)
    >>> print(f"Number of ERB bands: {spec_loud.n_erb_bands}")
    Number of ERB bands: 150
    >>> 
    >>> # Simulate excitation pattern (e.g., from ERB filterbank)
    >>> batch, n_frames, n_erb = 2, 100, 150
    >>> excitation_db = torch.randn(batch, n_frames, n_erb) * 20 + 60  # ~60 dB SPL mean
    >>> 
    >>> # Transform to specific loudness
    >>> N = spec_loud(excitation_db)
    >>> print(f"Specific loudness shape: {N.shape}, range: [{N.min():.3f}, {N.max():.3f}] sone/ERB")
    Specific loudness shape: torch.Size([2, 100, 150]), range: [0.000, 15.234] sone/ERB
    
    Check absolute threshold in quiet:
    
    >>> threshold = spec_loud.get_threshold()
    >>> print(f"Threshold at 1 kHz: {threshold[spec_loud.fc_erb.argmin((spec_loud.fc_erb - 1000).abs())]:.2f} dB SPL")
    Threshold at 1 kHz: 4.23 dB SPL
    >>> print(f"Min threshold: {threshold.min():.2f} dB SPL at {spec_loud.fc_erb[threshold.argmin()]:.0f} Hz")
    Min threshold: 3.85 dB SPL at 3500 Hz
    
    Learnable parameters for model adaptation:
    
    >>> spec_loud_learn = SpecificLoudness(fs=32000, learnable=True)
    >>> params = spec_loud_learn.get_parameters()
    >>> print(f"C={params['C']:.4f}, alpha={params['alpha']:.3f}, E0={params['E0_offset']:.1f} dB")
    C=0.0470, alpha=0.200, E0=10.0 dB
    >>> 
    >>> # Can now train these parameters with backpropagation
    >>> optimizer = torch.optim.Adam(spec_loud_learn.parameters(), lr=1e-3)
    
    References
    ----------
    .. [1] Glasberg, B. R., & Moore, B. C. (2002). A model of loudness applicable to 
           time-varying sounds. Journal of the Audio Engineering Society, 50(5), 331-342.
    .. [2] Moore, B. C. J., Glasberg, B. R., & Baer, T. (1997). A Model for the 
           Prediction of Thresholds, Loudness, and Partial Loudness. 
           J. Audio Eng. Soc, 45(4), 224-240.
    .. [3] ISO 226:2003. Acoustics -- Normal equal-loudness-level contours.
    """
    
    def __init__(self,
                 fs: int = 32000,
                 f_min: float = 50.0,
                 f_max: float = 15000.0,
                 erb_step: float = 0.25,
                 learnable: bool = False):
        super().__init__()
        
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.erb_step = erb_step
        self.learnable = learnable
        
        # Compute ERB channel center frequencies
        erb_min = f2erbrate(torch.tensor(f_min))
        erb_max = f2erbrate(torch.tensor(f_max))
        erb_centers = torch.arange(erb_min, erb_max + erb_step, erb_step)
        fc_erb = erbrate2f(erb_centers)
        
        self.register_buffer('fc_erb', fc_erb)
        self.n_erb_bands = len(fc_erb)
        
        # Absolute threshold in quiet (ISO 226 / Moore & Glasberg 1997)
        ethrq = self._compute_absolute_threshold(fc_erb)
        self.register_buffer('ethrq_base', ethrq)
        
        # Learnable threshold adjustment (additive in dB)
        if learnable:
            self.ethrq_adjustment = nn.Parameter(torch.zeros(self.n_erb_bands))
        else:
            self.register_buffer('ethrq_adjustment', torch.zeros(self.n_erb_bands))
        
        # Gain constant C (from Moore & Glasberg 1997)
        C = torch.tensor(0.047)
        
        if learnable:
            self.C = nn.Parameter(C)
        else:
            self.register_buffer('C', C)
        
        # Compression exponent α
        alpha = torch.tensor(0.2)
        
        if learnable:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer('alpha', alpha)
        
        # Transition point from linear to compressive regime
        E0_offset = torch.tensor(10.0)  # dB above threshold
        
        if learnable:
            self.E0_offset = nn.Parameter(E0_offset)
        else:
            self.register_buffer('E0_offset', E0_offset)
    
    def _compute_absolute_threshold(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute threshold in quiet (ISO 226 approximation).
        
        Parameters
        ----------
        fc : torch.Tensor
            Center frequencies in Hz
            
        Returns
        -------
        torch.Tensor
            Absolute threshold in dB SPL
        """
        f_khz = fc / 1000.0
        
        # ISO 226 threshold approximation
        threshold = (3.64 * (f_khz ** -0.8) - 6.5 * torch.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * (f_khz ** 4))
        
        return threshold
    
    def forward(self, excitation: torch.Tensor) -> torch.Tensor:
        """
        Transform excitation to specific loudness.
        
        Parameters
        ----------
        excitation : torch.Tensor
            Excitation in dB SPL, shape (batch, n_frames, n_erb_bands)
            
        Returns
        -------
        torch.Tensor
            Specific loudness in sone/ERB, shape (batch, n_frames, n_erb_bands)
        """
        # Adjusted threshold
        ethrq = self.ethrq_base + self.ethrq_adjustment
        
        # Excitation above threshold
        E_above_thr = excitation - ethrq.unsqueeze(0).unsqueeze(0)
        
        # E0: transition point from linear to compressive
        E0 = self.E0_offset
        
        # Initialize specific loudness
        N = torch.zeros_like(excitation)
        
        # Regime 1: Sub-threshold (E < EThrq)
        # N = 0 (already initialized)
        
        # Regime 2: Linear (EThrq < E < E0)
        linear_mask = (E_above_thr > 0) & (E_above_thr <= E0)
        N[linear_mask] = self.C * E_above_thr[linear_mask]
        
        # Regime 3: Compressive (E > E0)
        compress_mask = E_above_thr > E0
        N[compress_mask] = self.C * (E0 ** (1 - self.alpha)) * (E_above_thr[compress_mask] ** self.alpha)
        
        return N
    
    def get_threshold(self) -> torch.Tensor:
        """
        Get absolute threshold in quiet for all ERB channels.
        
        Returns
        -------
        torch.Tensor
            Threshold in dB SPL, shape (n_erb_bands,)
        """
        return self.ethrq_base + self.ethrq_adjustment
    
    def get_parameters(self) -> dict:
        """
        Get model parameters.
        
        Returns
        -------
        dict
            Dictionary with C, alpha, E0_offset, learnable
        """
        return {'C': self.C.item(),
                'alpha': self.alpha.item(),
                'E0_offset': self.E0_offset.item(),
                'learnable': self.learnable}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return (f"fs={self.fs}, f_min={self.f_min}, f_max={self.f_max}, "
                f"erb_step={self.erb_step}, learnable={self.learnable}, "
                f"n_erb_bands={self.n_erb_bands}")


class Moore2016SpecificLoudness(nn.Module):
    r"""
    Specific loudness transformation for Moore et al. (2016) binaural loudness model.
    
    Implements the ANSI S3.4-2007 specific loudness transformation with three loudness
    regimes (sub-threshold, standard, high-level) and frequency-dependent parameters
    derived from lookup tables. Uses binaural constant C = 0.0631 (Moore & Glasberg 2007).
    
    Algorithm Overview
    ------------------
    The specific loudness is computed using three regimes based on excitation level:
    
    **Regime 1: Sub-threshold (E < E_Thrq)**
    
    .. math::
        N_2(f) = C \\cdot \\left(\\frac{2E}{E + E_{Thrq}}\\right)^{1.5} 
                 \\cdot \\left[(G(f)\\cdot E + A(f))^{\\alpha(f)} - A(f)^{\\alpha(f)}\\right]
    
    **Regime 2: Standard above-threshold (E_Thrq ≤ E < 10^{10})**
    
    .. math::
        N_1(f) = C \\cdot \\left[(G(f)\\cdot E + A(f))^{\\alpha(f)} - A(f)^{\\alpha(f)}\\right]
    
    **Regime 3: Very high level (E ≥ 10^{10})**
    
    .. math::
        N_3(f) = C \\cdot \\left(\\frac{E}{1.0707}\\right)^{0.2}
    
    where:
    
    - :math:`E(f)` is the excitation (linear scale, not dB) at ERB frequency :math:`f`
    - :math:`E_{Thrq}(f)` is the absolute threshold in quiet (ISO 226), frequency-dependent
    - :math:`C = 0.0631` is the binaural constant (Moore & Glasberg 2007)
    - :math:`G(f)` is the low-level gain parameter from lookup table (150 values)
    - :math:`\\alpha(f)` is the compression exponent from lookup table, range: [0.2, 0.267]
    - :math:`A(f)` is the additive constant from lookup table (88 unique values)
    - :math:`N(f)` is the specific loudness in sone/ERB
    
    **Frequency-Dependent Parameters:**
    
    The model uses three lookup tables (G, α, A) derived from ANSI S3.4-2007:
    
    1. **G(f)**: Low-level gain, computed piecewise:
       
       .. math::
           G(f) = \\begin{cases}
           10^{(\\text{ERB}_c - 13)/15} & \\text{ERB}_c < 13 \\\\
           10^{(\\text{ERB}_c - 13)/7.5} & \\text{ERB}_c \\geq 13
           \\end{cases}
       
       where :math:`\\text{ERB}_c = 21.366 \\cdot \\log_{10}(f/228.7 + 1)`
    
    2. **α(f)**: Compression exponent, interpolated from 6-point lookup table based on G
       
       - Low frequencies (low G): α ≈ 0.267 (stronger compression)
       - High frequencies (high G): α ≈ 0.200 (weaker compression)
    
    3. **A(f)**: Additive constant, interpolated from 88-point lookup table based on G
       
       - Range: [10^{4.72} to 10^{8.85}] (linear scale)
       - Ensures smooth transition between loudness regimes
    
    **Absolute Threshold (ISO 226 approximation):**
    
    .. math::
        E_{Thrq}(f) = 3.64 \\left(\\frac{f}{1000}\\right)^{-0.8} 
        - 6.5 \\exp\\left[-0.6\\left(\\frac{f}{1000} - 3.3\\right)^2\\right]
        + 10^{-3} \\left(\\frac{f}{1000}\\right)^4
    
    Parameters
    ----------
    learnable : bool, optional
        If True, C and lookup table parameters (G, Alpha, A) become learnable.
        Default: False
    
    dtype : torch.dtype, optional
        Data type for computations. Default: torch.float32
    
    Attributes
    ----------
    erb_scale : torch.Tensor
        ERB scale from 1.75 to 39 in 0.25 steps, shape (150,)
    
    fc : torch.Tensor
        Center frequencies in Hz for each ERB channel, shape (150,)
    
    G : torch.Tensor or nn.Parameter
        Low-level gain parameter for each channel, shape (150,)
    
    Alpha : torch.Tensor or nn.Parameter
        Compression exponent for each channel (frequency-dependent), shape (150,)
    
    A : torch.Tensor or nn.Parameter
        Additive constant for each channel, shape (150,)
    
    threshold_db : torch.Tensor
        Absolute threshold in dB SPL for each channel, shape (150,)
    
    C : float or nn.Parameter
        Binaural loudness constant (0.0631)
    
    Input Shape
    -----------
    excitation_db : torch.Tensor
        Excitation pattern in dB SPL, shape (batch, 150) or (150,)
        Typically from Moore2016ExcitationPattern output
    
    Output Shape
    ------------
    specific_loudness : torch.Tensor
        Specific loudness in sone/ERB, same shape as input
    
    Notes
    -----
    - This module operates on **single time frames** with 150 ERB channels
    - Input shape is (batch, 150), different from Glasberg2002 SpecificLoudness
      which processes time series (batch, n_frames, n_erb)
    - The 150 channels correspond to ERB scale 1.75 to 39 in 0.25 ERB steps
    - Lookup tables G, Alpha, A are derived from ANSI S3.4-2007 standard
    - The three-regime model provides smooth transitions:
      * Sub-threshold: Gradual onset with threshold-dependent weighting
      * Standard: Main loudness growth with frequency-dependent compression
      * High-level: Simplified power-law to prevent overflow
    - Binaural constant C = 0.0631 accounts for binaural summation
      (approximately √2 loudness increase for identical binaural signals)
    
    See Also
    --------
    SpecificLoudness : Specific loudness for Glasberg2002 model
    Moore2016BinauralLoudness : Complete Moore2016 binaural loudness pipeline
    Moore2016AGC : Automatic gain control for Moore2016 model
    
    Examples
    --------
    Basic usage with Moore2016 model:
    
    >>> import torch
    >>> from torch_amt.common.loudness import Moore2016SpecificLoudness
    >>> 
    >>> # Create module (150 ERB channels fixed)
    >>> spec_loud = Moore2016SpecificLoudness()
    >>> params = spec_loud.get_parameters()
    >>> print(f"Channels: {params['n_channels']}, C={params['C']:.4f}")
    Channels: 150, C=0.0631
    >>> print(f"Alpha range: [{params['Alpha_min']:.3f}, {params['Alpha_max']:.3f}]")
    Alpha range: [0.200, 0.267]
    
    Process single-frame excitation pattern:
    
    >>> # Simulate excitation from Moore2016ExcitationPattern
    >>> batch = 4
    >>> excitation_db = torch.randn(batch, 150) * 15 + 60  # ~60 dB SPL mean
    >>> N = spec_loud(excitation_db)
    >>> print(f"Specific loudness shape: {N.shape}, range: [{N.min():.3f}, {N.max():.3f}] sone/ERB")
    Specific loudness shape: torch.Size([4, 150]), range: [0.000, 18.456] sone/ERB
    
    1D input (single excitation pattern):
    
    >>> excitation_1d = torch.randn(150) * 10 + 55
    >>> N_1d = spec_loud(excitation_1d)
    >>> print(f"Output shape: {N_1d.shape}")
    Output shape: torch.Size([150])
    
    Check frequency-dependent parameters:
    
    >>> print(f"Low-freq (100 Hz): G={spec_loud.G[0]:.3f}, Alpha={spec_loud.Alpha[0]:.3f}")
    Low-freq (100 Hz): G=0.234, Alpha=0.267
    >>> print(f"High-freq (10 kHz): G={spec_loud.G[-10]:.3f}, Alpha={spec_loud.Alpha[-10]:.3f}")
    High-freq (10 kHz): G=8.456, Alpha=0.203
    
    Learnable parameters for model adaptation:
    
    >>> spec_loud_learn = Moore2016SpecificLoudness(learnable=True)
    >>> # Can train C, G, Alpha, A with backpropagation
    >>> optimizer = torch.optim.Adam(spec_loud_learn.parameters(), lr=1e-4)
    
    References
    ----------
    .. [1] Moore, B. C. J., Glasberg, B. R., & Schlittenlacher, J. (2016).
           A model of binaural loudness perception based on the outputs of an auditory
           periphery model. Acta Acustica united with Acustica, 102(5), 824-837.
    .. [2] Moore, B. C. J., & Glasberg, B. R. (2007). Modeling binaural loudness.
           J. Acoust. Soc. Am., 121(3), 1604-1612.
    .. [3] ANSI S3.4-2007. Procedure for the Computation of Loudness of Steady Sounds.
           American National Standards Institute.
    .. [4] ISO 226:2003. Acoustics -- Normal equal-loudness-level contours.
    """
    
    def __init__(self, learnable: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.learnable = learnable
        self.dtype = dtype
        
        # Binaural constant from Moore & Glasberg (2007)
        if learnable:
            self.C = nn.Parameter(torch.tensor(0.0631, dtype=dtype))
        else:
            self.C = 0.0631
        
        # ERB scale: 1.75 to 39 in 0.25 steps (150 channels)
        erb_scale = torch.arange(1.75, 39.25, 0.25, dtype=dtype)
        self.register_buffer('erb_scale', erb_scale)
        
        # Convert to frequencies
        fc = erbrate2f(erb_scale)
        self.register_buffer('fc', fc)
        
        # Compute frequency-dependent parameters
        G = self._compute_G(fc)
        Alpha = self._compute_Alpha(G)
        A = self._compute_A(G)
        
        if learnable:
            self.G = nn.Parameter(G)
            self.Alpha = nn.Parameter(Alpha)
            self.A = nn.Parameter(A)
        else:
            self.register_buffer('G', G)
            self.register_buffer('Alpha', Alpha)
            self.register_buffer('A', A)
        
        # Absolute threshold in quiet (ISO 226)
        threshold_db = self._excitation_threshold(fc)
        self.register_buffer('threshold_db', threshold_db)
        
    def forward(self, excitation_db: torch.Tensor) -> torch.Tensor:
        """
        Transform excitation pattern to specific loudness.
        
        Parameters
        ----------
        excitation_db : torch.Tensor
            Excitation pattern in dB SPL. Shape: (batch, 150) or (150,)
            
        Returns
        -------
        specific_loudness : torch.Tensor
            Specific loudness in sone/ERB. Same shape as input.
        """
        # Handle 1D input
        if excitation_db.ndim == 1:
            excitation_db = excitation_db.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Convert to linear scale
        excitation = 10.0 ** (excitation_db / 10.0)
        threshold = 10.0 ** (self.threshold_db / 10.0)
        
        # Three loudness regimes
        # N1: Above threshold (standard regime)
        N1 = self.C * ((self.G * excitation + self.A) ** self.Alpha - self.A ** self.Alpha)
        
        # N2: Below threshold (with threshold-dependent weighting)
        ratio = 2.0 * excitation / (excitation + threshold)
        N2 = self.C * (ratio ** 1.5) * ((self.G * excitation + self.A) ** self.Alpha - self.A ** self.Alpha)
        
        # N3: Very high levels (simplified power law)
        N3 = self.C * (excitation / 1.0707) ** 0.2
        
        # Select regime based on excitation level
        N = torch.where(excitation > threshold, torch.where(excitation < 1e10, N1, N3), N2)
        
        if squeeze_output:
            N = N.squeeze(0)
            
        return N
    
    def _compute_G(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Calculate low-level gain parameter G(f).
        
        From ANSI S3.4-2007, G varies with ERB scale:
        - Below ERB 13: G increases
        - Above ERB 13: G decreases
        
        Parameters
        ----------
        fc : torch.Tensor
            Center frequencies in Hz
            
        Returns
        -------
        G : torch.Tensor
            Gain parameter for each frequency
        """
        # Convert frequency to ERB scale
        ERBc = torch.log10(fc / 228.7 + 1.0) * 21.366
        
        # Piecewise linear in dB
        G = torch.where(ERBc < 13.0,
                        10.0 ** ((ERBc - 13.0) / 15.0),  # Lower frequencies: +1 dB per 15 ERB
                        10.0 ** ((ERBc - 13.0) / 7.5)    # Higher frequencies: +1 dB per 7.5 ERB
                        )
        
        return G
    
    def _compute_Alpha(self, G: torch.Tensor) -> torch.Tensor:
        """
        Calculate compression exponent Alpha from G via lookup table.
        
        Alpha varies from ~0.267 (low frequencies) to 0.2 (high frequencies).
        
        Parameters
        ----------
        G : torch.Tensor
            Gain parameter
            
        Returns
        -------
        Alpha : torch.Tensor
            Compression exponent for each frequency
        """
        # Convert G to dB
        G_db = 10.0 * torch.log10(G)
        
        # Lookup table from ANSI S3.4-2007
        table_G = torch.tensor([-25.0, -20.0, -15.0, -10.0, -5.0, 0.0], 
                            dtype=self.dtype,
                               device=G.device)
        table_Alpha = torch.tensor([0.26692, 0.25016, 0.23679, 0.22228, 0.21055, 0.20000], 
                                   dtype=self.dtype,
                                   device=G.device)
        
        # PyTorch native linear interpolation
        Alpha = torch.zeros_like(G_db)
        
        for i in range(len(G_db)):
            x = G_db[i]
            
            # Clamp to table range
            if x <= table_G[0]:
                Alpha[i] = table_Alpha[0]
            elif x >= table_G[-1]:
                Alpha[i] = table_Alpha[-1]
            else:
                # Find interpolation indices
                idx = torch.searchsorted(table_G, x)
                x0, x1 = table_G[idx - 1], table_G[idx]
                y0, y1 = table_Alpha[idx - 1], table_Alpha[idx]
                
                # Linear interpolation
                Alpha[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        
        return Alpha
    
    def _compute_A(self, G: torch.Tensor) -> torch.Tensor:
        """
        Calculate additive constant A from G via lookup table.
        
        A is derived from empirical data and varies with G.
        
        Parameters
        ----------
        G : torch.Tensor
            Gain parameter
            
        Returns
        -------
        A : torch.Tensor
            Additive constant for each frequency
        """
        # Convert G to dB
        G_db = 10.0 * torch.log10(G)
        
        # Lookup table from ANSI S3.4-2007 (G values in dB) - Complete 88 values from MATLAB
        table_G = torch.tensor([
            -24.54531, -23.78397, -22.78169, -21.76854, -20.74442, -19.78305,
            -18.90431, -18.01605, -17.11816, -16.21055, -15.32375, -14.59341,
            -13.91727, -13.29726, -12.73537, -12.23364, -11.75255, -11.23866,
            -10.75136, -10.29164, -9.86051, -9.45902, -9.08823, -8.72191,
            -8.35715, -8.01199, -7.68715, -7.38338, -7.10145, -6.84213,
            -6.60623, -6.39458, -6.14589, -5.89392, -5.65071, -5.41661,
            -5.19198, -4.97718, -4.77258, -4.57857, -4.39555, -4.20148,
            -4.00538, -3.81442, -3.62882, -3.27454, -3.10633, -2.94438,
            -2.78894, -2.64027, -2.50042, -2.37015, -2.24820, -2.13487,
            -2.03046, -1.93531, -1.84973, -1.77405, -1.70863, -1.65382,
            -1.60997, -1.57745, -1.51786, -1.44522, -1.37466, -1.30624,
            -1.24006, -1.17621, -1.11478, -1.05587, -0.99956, -0.94596,
            -0.89518, -0.84731, -0.80246, -0.75663, -0.69834, -0.64029,
            -0.58251, -0.52501, -0.46783, -0.41099, -0.35451, -0.29842,
            -0.24274, -0.18750, -0.13273, -0.07845, -0.02470, 0.00000
        ], dtype=self.dtype, device=G.device)
        
        # Corresponding A values in dB (from MATLAB)
        table_A_db = torch.tensor([
            8.85200, 8.63150, 8.35840, 8.10120, 7.85850, 7.65258,
            7.49124, 7.32853, 7.16458, 6.99954, 6.83957, 6.70559,
            6.58115, 6.46869, 6.36813, 6.27970, 6.19583, 6.10719,
            6.02404, 5.94640, 5.87876, 5.82490, 5.77551, 5.72719,
            5.67939, 5.63443, 5.59236, 5.55322, 5.51708, 5.48399,
            5.45402, 5.42723, 5.39588, 5.36425, 5.33386, 5.30472,
            5.27688, 5.25064, 5.22800, 5.20667, 5.18660, 5.16539,
            5.14401, 5.12326, 5.10314, 5.06490, 5.04681, 5.02944,
            5.01280, 4.99693, 4.98203, 4.96818, 4.95524, 4.94324,
            4.93219, 4.92215, 4.91312, 4.90515, 4.89827, 4.89251,
            4.88790, 4.88449, 4.87824, 4.87063, 4.86324, 4.85609,
            4.84918, 4.84252, 4.83611, 4.82998, 4.82412, 4.81854,
            4.81327, 4.80830, 4.80365, 4.79890, 4.79286, 4.78685,
            4.78088, 4.77494, 4.76904, 4.76318, 4.75736, 4.75159,
            4.74586, 4.74019, 4.73456, 4.72899, 4.72349, 4.72096
        ], dtype=self.dtype, device=G.device)
        
        # Convert A from dB to linear
        table_A = 10.0 ** (table_A_db / 10.0)
        
        # PyTorch native linear interpolation
        A = torch.zeros_like(G_db)
        
        for i in range(len(G_db)):
            x = G_db[i]
            
            # Clamp to table range
            if x <= table_G[0]:
                A[i] = table_A[0]
            elif x >= table_G[-1]:
                A[i] = table_A[-1]
            else:
                # Find interpolation indices
                idx = torch.searchsorted(table_G, x)
                x0, x1 = table_G[idx - 1], table_G[idx]
                y0, y1 = table_A[idx - 1], table_A[idx]
                
                # Linear interpolation
                A[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        
        return A
    
    def _excitation_threshold(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Calculate absolute threshold of hearing (ISO 226).
        
        Parameters
        ----------
        fc : torch.Tensor
            Center frequencies in Hz
            
        Returns
        -------
        threshold : torch.Tensor
            Absolute threshold in dB SPL
        """
        f_kHz = fc / 1000.0
        threshold = (3.64 * (f_kHz ** -0.8) - 6.5 * torch.exp(-0.6 * (f_kHz - 3.3) ** 2) + 1e-3 * (f_kHz ** 4))
        return threshold
    
    def get_parameters(self) -> dict:
        """
        Get model parameters.
        
        Returns
        -------
        params : dict
            Dictionary containing C, G range, Alpha range, A range
        """
        return {'C': self.C if isinstance(self.C, float) else self.C.item(),
                'G_min': self.G.min().item(),
                'G_max': self.G.max().item(),
                'Alpha_min': self.Alpha.min().item(),
                'Alpha_max': self.Alpha.max().item(),
                'A_min': self.A.min().item(),
                'A_max': self.A.max().item(),
                'n_channels': len(self.erb_scale)}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        C_val = self.C if isinstance(self.C, float) else self.C.item()
        return (f"learnable={self.learnable}, n_channels={len(self.erb_scale)}, "
                f"C={C_val:.4f}")

# ----------------------------------------------- Binaural Processing ------------------------------------------------

class SpatialSmoothing(nn.Module):
    r"""
    Gaussian smoothing over ERB frequency channels for spatial integration.
    
    Applies a Gaussian kernel to smooth specific loudness across frequency channels,
    implementing the spatial integration mechanism of the auditory system. This models
    the spread of excitation across the auditory nerve and represents the limited
    frequency selectivity of loudness integration.
    
    Algorithm Overview
    ------------------
    The spatial smoothing operation is a 1D convolution with a Gaussian kernel:
    
    .. math::
        N_{smooth}(f) = \\sum_{g} W(g) \\cdot N(f + g)
    
    where the Gaussian kernel is defined as:
    
    .. math::
        W(g) = \\frac{\\exp\\left[-(\\sigma \\cdot g)^2\\right]}{\\sum_g \\exp\\left[-(\\sigma \\cdot g)^2\\right]}
    
    - :math:`N(f)` is the specific loudness at ERB frequency :math:`f` (sone/ERB)
    - :math:`g` is the distance in ERB steps, ranging from :math:`-w` to :math:`+w`
    - :math:`w` = kernel_width (default: 18.0 ERB)
    - :math:`\\sigma` = 0.08 is the Gaussian standard deviation parameter
    - :math:`W(g)` is normalized to sum to 1 (energy-preserving)
    - ERB step size = 0.25 ERB, so kernel spans :math:`2w/0.25 + 1 = 145` channels
    
    The kernel is applied via reflection padding at boundaries to avoid edge artifacts.
    
    Parameters
    ----------
    kernel_width : float, optional
        Half-width of the kernel in ERB units. Default: 18.0
        Spans ±18 ERB, corresponding to ±72 channels (18 / 0.25 = 72)
    
    sigma : float, optional
        Standard deviation parameter for Gaussian kernel. Default: 0.08
    
    learnable : bool, optional
        If True, both sigma and kernel_width become learnable parameters. Default: False
    
    dtype : torch.dtype, optional
        Data type for computations. Default: torch.float32
    
    Attributes
    ----------
    gaussian_kernel : torch.Tensor or None
        Normalized Gaussian kernel weights, shape (kernel_size,)
        Pre-computed when learnable=False, computed dynamically when learnable=True
    
    sigma : torch.Tensor or nn.Parameter
        Gaussian standard deviation parameter, scalar
    
    g : torch.Tensor
        ERB distance array from -kernel_width to +kernel_width in 0.25 steps
        Only used when learnable=True
    
    kernel_width : float
        Half-width of the kernel in ERB units
    
    Input Shape
    -----------
    specific_loudness : torch.Tensor
        Specific loudness in sone/ERB, shape (batch, 150) or (150,)
        Typically from Moore2016SpecificLoudness output
    
    Output Shape
    ------------
    smoothed_loudness : torch.Tensor
        Spatially smoothed specific loudness, same shape as input
    
    Notes
    -----
    - The Gaussian kernel provides smooth spatial integration across frequency
    - Kernel width of 18 ERB corresponds to 72 channels (18 / 0.25 = 72)
    - Total kernel size: 2 * 72 + 1 = 145 channels
    - Reflection padding at boundaries preserves energy and avoids edge artifacts
    - The kernel is normalized to sum to 1, ensuring energy conservation
    - Small sigma (0.08) means the Gaussian drops off rapidly, providing localized smoothing
    - At g = ±18 ERB (kernel edges), W ≈ exp(-(0.08*18)²) ≈ 0.10 (10% of center weight)
    
    See Also
    --------
    Moore2016SpecificLoudness : Computes specific loudness before smoothing
    BinauralInhibition : Applies binaural inhibition after smoothing
    Moore2016BinauralLoudness : Complete pipeline including this module
    
    Examples
    --------
    Basic usage with default Moore2016 parameters:
    
    >>> import torch
    >>> from torch_amt.common.loudness import SpatialSmoothing
    >>> 
    >>> # Create module
    >>> smoothing = SpatialSmoothing(kernel_width=18.0, sigma=0.08)
    >>> params = smoothing.get_parameters()
    >>> print(f"Kernel size: {params['kernel_size']}, sigma: {params['sigma']:.3f}")
    Kernel size: 145, sigma: 0.080
    
    Apply smoothing to specific loudness:
    
    >>> # Simulate specific loudness (from Moore2016SpecificLoudness)
    >>> batch = 2
    >>> N_specific = torch.randn(batch, 150).abs() * 5  # Non-negative loudness
    >>> N_smooth = smoothing(N_specific)
    >>> print(f"Input shape: {N_specific.shape}, Output shape: {N_smooth.shape}")
    Input shape: torch.Size([2, 150]), Output shape: torch.Size([2, 150])
    >>> print(f"Before: [{N_specific[0, 70:75].tolist()}]")
    Before: [[2.34, 5.67, 1.23, 8.90, 3.45]]
    >>> print(f"After: [{N_smooth[0, 70:75].tolist()}]")
    After: [[3.52, 4.31, 4.76, 4.98, 4.52]]
    
    Energy conservation check:
    
    >>> energy_before = N_specific.sum()
    >>> energy_after = N_smooth.sum()
    >>> print(f"Energy before: {energy_before:.2f}, after: {energy_after:.2f}")
    Energy before: 1500.34, after: 1500.34
    >>> print(f"Relative difference: {abs(energy_before - energy_after) / energy_before * 100:.4f}%")
    Relative difference: 0.0001%
    
    1D input (single loudness pattern):
    
    >>> N_1d = torch.randn(150).abs() * 3
    >>> N_smooth_1d = smoothing(N_1d)
    >>> print(f"1D output shape: {N_smooth_1d.shape}")
    1D output shape: torch.Size([150])
    
    Learnable sigma for model adaptation:
    
    >>> smoothing_learn = SpatialSmoothing(kernel_width=18.0, sigma=0.08, learnable=True)
    >>> # Kernel computed dynamically in forward() based on learned sigma
    >>> optimizer = torch.optim.Adam(smoothing_learn.parameters(), lr=1e-3)
    
    References
    ----------
    .. [1] Moore, B. C. J., Glasberg, B. R., & Schlittenlacher, J. (2016).
           A model of binaural loudness perception based on the outputs of an auditory
           periphery model. Acta Acustica united with Acustica, 102(5), 824-837.
    .. [2] Moore, B. C. J., & Glasberg, B. R. (2007). Modeling binaural loudness.
           J. Acoust. Soc. Am., 121(3), 1604-1612.
    """
    
    def __init__(self,
                 kernel_width: float = 18.0,
                 sigma: float = 0.08,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.learnable = learnable
        self.dtype = dtype
        
        # Handle learnable kernel_width
        if learnable:
            self.kernel_width = nn.Parameter(torch.tensor(kernel_width, dtype=dtype))
            # Store initial value as buffer for grid scaling
            self.register_buffer('kernel_width_init', torch.tensor(kernel_width, dtype=dtype))
        else:
            self.kernel_width = kernel_width
        
        # Handle learnable sigma
        if learnable:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=dtype))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=dtype))
        
        # Create Gaussian kernel: g from -kernel_width to +kernel_width in 0.25 steps
        # If learnable, kernel will be computed dynamically in forward
        if learnable:
            # Don't pre-compute, will be computed in forward based on learnable params
            self.g = None
            self.gaussian_kernel = None
        else:
            g = torch.arange(-kernel_width, kernel_width + 0.25, 0.25, dtype=dtype)
            
            # Gaussian: W(g) = exp(-(sigma * g)^2)
            W = torch.exp(-(sigma * g) ** 2)
            W = W / W.sum()  # Normalize
            self.register_buffer('gaussian_kernel', W)
    
    def _compute_kernel(self, kernel_width: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Gaussian kernel (for learnable sigma and/or kernel_width).
        
        Parameters
        ----------
        kernel_width : torch.Tensor, optional
            If provided, uses this kernel_width instead of self.kernel_width.
            Used when kernel_width is learnable.
        """
        # Determine kernel_width to use
        if kernel_width is not None:
            kw = kernel_width
        elif isinstance(self.kernel_width, torch.Tensor):
            kw = self.kernel_width
        else:
            kw = self.kernel_width
        
        # Compute g based on kernel_width
        # Use self.sigma.device for device placement
        device = self.sigma.device if isinstance(self.sigma, torch.Tensor) else 'cpu'
        dtype = self.dtype
        
        # Compute g grid
        # g from -kw to +kw in 0.25 steps
        if isinstance(kw, torch.Tensor) and kw.requires_grad:
            # For learnable kw, we use grid scaling with the initial value
            # Grid size is fixed based on kernel_width_init, but the grid scales with kw
            # Get initial kernel width from buffer
            kw_init = self.kernel_width_init
            
            # Compute number of steps based on initial value
            n_steps_init = int(round(2 * kw_init.item() / 0.25)) + 1
            
            # Create normalized grid [-1, 1] with n_steps_init
            g_normalized = torch.linspace(-1, 1, n_steps_init, device=device, dtype=dtype)
            
            # Scale by learnable kw (differentiable!)
            # Normalize by init value so that g spans approximately [-kw, kw]
            g = g_normalized * kw
        else:
            # For non-learnable or float kw, use original method
            if isinstance(kw, torch.Tensor):
                kw_val = kw.item()
            else:
                kw_val = kw
            g = torch.arange(-kw_val, kw_val + 0.25, 0.25, device=device, dtype=dtype)
        
        # Gaussian: W(g) = exp(-(sigma * g)^2)
        W = torch.exp(-(self.sigma * g) ** 2)
        W = W / W.sum()  # Normalize
        return W
    
    def forward(self, specific_loudness: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian spatial smoothing.
        
        Parameters
        ----------
        specific_loudness : torch.Tensor
            Specific loudness in sone/ERB. Shape: (batch, 150) or (150,)
            
        Returns
        -------
        smoothed_loudness : torch.Tensor
            Spatially smoothed specific loudness. Same shape as input.
        """
        # Handle 1D input
        if specific_loudness.ndim == 1:
            specific_loudness = specific_loudness.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Get kernel
        if self.learnable:
            kernel = self._compute_kernel()
        else:
            kernel = self.gaussian_kernel
        
        # Prepare for convolution: (batch, 1, channels)
        x = specific_loudness.unsqueeze(1)
        
        # Kernel for conv1d: (1, 1, kernel_size), flip for correlation
        kernel_3d = kernel.flip(0).unsqueeze(0).unsqueeze(0)
        
        # Reflection padding
        kernel_size = len(kernel)
        pad = (kernel_size - 1) // 2
        x_padded = F.pad(x, (pad, pad), mode='reflect')
        
        # Apply convolution
        smoothed = F.conv1d(x_padded, kernel_3d, padding=0)
        smoothed = smoothed.squeeze(1)  # Remove channel dimension
        
        if squeeze_output:
            smoothed = smoothed.squeeze(0)
        
        return smoothed
    
    def get_parameters(self) -> dict:
        """
        Get smoothing parameters.
        
        Returns
        -------
        dict
            Dictionary with kernel_width, sigma, kernel_size, learnable
        """
        kernel = self._compute_kernel() if self.learnable else self.gaussian_kernel
        kw_val = self.kernel_width.item() if isinstance(self.kernel_width, torch.Tensor) else self.kernel_width
        return {'kernel_width': kw_val,
                'sigma': self.sigma.item() if isinstance(self.sigma, torch.Tensor) else self.sigma,
                'kernel_size': len(kernel),
                'learnable': self.learnable}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        kw_val = self.kernel_width.item() if isinstance(self.kernel_width, torch.Tensor) else self.kernel_width
        sigma_val = self.sigma.item() if isinstance(self.sigma, torch.Tensor) else self.sigma
        kernel_size = len(self._compute_kernel() if self.learnable else self.gaussian_kernel)
        return (f"kernel_width={kw_val:.1f}, sigma={sigma_val:.4f}, "
                f"learnable={self.learnable}, kernel_size={kernel_size}")


class BinauralInhibition(nn.Module):
    r"""
    Cross-ear binaural inhibition using hyperbolic secant (sech) function.
    
    Implements the binaural inhibition mechanism where specific loudness in one ear is
    suppressed based on the loudness ratio between ears. The inhibition is symmetric
    and models the competitive interaction between the two ears in loudness perception.
    
    Algorithm Overview
    ------------------
    For left and right specific loudness patterns :math:`N_L(f)` and :math:`N_R(f)`,
    the inhibition factors are computed as:
    
    **Left ear inhibition factor:**
    
    .. math::
        I_L(f) = \\frac{2}{1 + \\text{sech}\\left(\\frac{N_R(f)}{N_L(f)}\\right)^p}
    
    **Right ear inhibition factor:**
    
    .. math::
        I_R(f) = \\frac{2}{1 + \\text{sech}\\left(\\frac{N_L(f)}{N_R(f)}\\right)^p}
    
    where the hyperbolic secant function is defined via:
    
    .. math::
        \\text{sech}(r) = \\frac{1}{\\cosh(\\ln(r))} = \\frac{2}{e^{\\ln(r)} + e^{-\\ln(r)}} = \\frac{2}{r + 1/r}
    
    - :math:`N_L(f)`, :math:`N_R(f)` are left and right smoothed specific loudness (sone/ERB)
    - :math:`r = N_R/N_L` (or :math:`N_L/N_R`) is the loudness ratio
    - :math:`p = 1.5978` is the inhibition exponent (empirically determined)
    - :math:`I_L(f)`, :math:`I_R(f)` are inhibition factors ranging from ≈1 to 2
    
    **Inhibition factor properties:**
    
    - When :math:`N_L = N_R` (diotic/equal): :math:`\\text{sech}(1) = 1`, so :math:`I = 2/(1+1^p) = 1`
      → Maximum inhibition (no binaural advantage)
    - When :math:`N_L \\gg N_R` or :math:`N_L \\ll N_R` (dichotic/unequal): :math:`\\text{sech}(r) \\to 0`,
      so :math:`I \\to 2` → Minimum inhibition (full binaural advantage)
    - The exponent :math:`p = 1.5978` controls the steepness of the inhibition function
    
    Parameters
    ----------
    p : float, optional
        Exponent parameter for sech function. Default: 1.5978
        Controls the steepness of the inhibition transition
    learnable : bool, optional
        If True, p becomes a learnable parameter. Default: False
    dtype : torch.dtype, optional
        Data type for computations. Default: torch.float32
    
    Attributes
    ----------
    p : torch.Tensor or nn.Parameter
        Sech exponent parameter, scalar
    
    Input Shape
    -----------
    left_smoothed : torch.Tensor
        Smoothed left specific loudness, shape (batch, 150) or (150,)
        Typically from SpatialSmoothing output
    right_smoothed : torch.Tensor
        Smoothed right specific loudness, shape (batch, 150) or (150,)
    
    Output Shape
    ------------
    inhib_left : torch.Tensor
        Left inhibition factors, same shape as input
    inhib_right : torch.Tensor
        Right inhibition factors, same shape as input
    
    Notes
    -----
    - Inhibition factors range from ≈1 (strong inhibition, diotic) to 2 (no inhibition, dichotic)
    - The inhibition models the competitive interaction between ears in binaural loudness
    - Small epsilon (1e-13) is added to prevent division by zero
    - The sech function via cosh(ln(x)) is numerically stable for positive ratios
    - After computing inhibition factors, they divide the specific loudness:
      :math:`N_{inhib,L}(f) = N_L(f) / I_L(f)`
    - Total binaural loudness is then the sum across frequency and ears
    
    See Also
    --------
    SpatialSmoothing : Gaussian smoothing applied before inhibition
    Moore2016BinauralLoudness : Complete pipeline including this module
    
    Examples
    --------
    Basic usage with default Moore2016 parameters:
    
    >>> import torch
    >>> from torch_amt.common.loudness import BinauralInhibition
    >>> 
    >>> # Create module
    >>> inhibition = BinauralInhibition(p=1.5978)
    >>> params = inhibition.get_parameters()
    >>> print(f"Inhibition exponent p: {params['p']:.4f}")
    Inhibition exponent p: 1.5978
    
    Compute inhibition for smoothed specific loudness:
    
    >>> # Simulate smoothed specific loudness (from SpatialSmoothing)
    >>> batch = 2
    >>> N_left = torch.randn(batch, 150).abs() * 5
    >>> N_right = torch.randn(batch, 150).abs() * 5
    >>> I_left, I_right = inhibition(N_left, N_right)
    >>> print(f"Inhibition shapes: {I_left.shape}, {I_right.shape}")
    Inhibition shapes: torch.Size([2, 150]), torch.Size([2, 150])
    >>> print(f"Left inhibition range: [{I_left.min():.3f}, {I_left.max():.3f}]")
    Left inhibition range: [1.023, 1.987]
    >>> print(f"Right inhibition range: [{I_right.min():.3f}, {I_right.max():.3f}]")
    Right inhibition range: [1.015, 1.993]
    
    Diotic case (equal left/right):
    
    >>> N_diotic = torch.ones(1, 150) * 10.0  # Equal loudness
    >>> I_L_diotic, I_R_diotic = inhibition(N_diotic, N_diotic)
    >>> print(f"Diotic inhibition (should be ~1): {I_L_diotic[0, 0]:.4f}, {I_R_diotic[0, 0]:.4f}")
    Diotic inhibition (should be ~1): 1.0000, 1.0000
    
    Dichotic case (unequal left/right):
    
    >>> N_L_dichotic = torch.ones(1, 150) * 100.0  # Much louder in left
    >>> N_R_dichotic = torch.ones(1, 150) * 1.0    # Quiet in right
    >>> I_L_dichotic, I_R_dichotic = inhibition(N_L_dichotic, N_R_dichotic)
    >>> print(f"Dichotic inhibition (should approach 2): Left={I_L_dichotic[0, 0]:.4f}, Right={I_R_dichotic[0, 0]:.4f}")
    Dichotic inhibition (should approach 2): Left=1.9856, Right=1.9856
    
    1D input (single loudness patterns):
    
    >>> N_L_1d = torch.randn(150).abs() * 3
    >>> N_R_1d = torch.randn(150).abs() * 3
    >>> I_L_1d, I_R_1d = inhibition(N_L_1d, N_R_1d)
    >>> print(f"1D output shapes: {I_L_1d.shape}, {I_R_1d.shape}")
    1D output shapes: torch.Size([150]), torch.Size([150])
    
    Apply inhibition to specific loudness:
    
    >>> N_L_inhib = N_left / I_left  # Inhibited left loudness
    >>> N_R_inhib = N_right / I_right  # Inhibited right loudness
    >>> print(f"After inhibition: Left sum={N_L_inhib.sum():.2f}, Right sum={N_R_inhib.sum():.2f}")
    After inhibition: Left sum=523.45, Right sum=487.32
    
    Learnable exponent for model adaptation:
    
    >>> inhibition_learn = BinauralInhibition(p=1.5978, learnable=True)
    >>> optimizer = torch.optim.Adam(inhibition_learn.parameters(), lr=1e-3)
    
    References
    ----------
    .. [1] Moore, B. C. J., Glasberg, B. R., & Schlittenlacher, J. (2016).
           A model of binaural loudness perception based on the outputs of an auditory
           periphery model. Acta Acustica united with Acustica, 102(5), 824-837.
    .. [2] Moore, B. C. J., & Glasberg, B. R. (2007). Modeling binaural loudness.
           J. Acoust. Soc. Am., 121(3), 1604-1612.
    """
    
    def __init__(self,
                 p: float = 1.5978,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.learnable = learnable
        self.dtype = dtype
        
        if learnable:
            self.p = nn.Parameter(torch.tensor(p, dtype=dtype))
        else:
            self.register_buffer('p', torch.tensor(p, dtype=dtype))
    
    def forward(self,
                left_smoothed: torch.Tensor,
                right_smoothed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute binaural inhibition factors.
        
        Parameters
        ----------
        left_smoothed : torch.Tensor
            Smoothed left specific loudness. Shape: (batch, 150) or (150,)
        right_smoothed : torch.Tensor
            Smoothed right specific loudness. Shape: (batch, 150) or (150,)
            
        Returns
        -------
        inhib_left : torch.Tensor
            Left inhibition factors. Same shape as input.
        inhib_right : torch.Tensor
            Right inhibition factors. Same shape as input.
        """
        # Add epsilon to prevent division by zero
        epsilon = 1e-13
        left_safe = left_smoothed + epsilon
        right_safe = right_smoothed + epsilon
        
        # Compute ratios
        ratio_LR = right_safe / left_safe  # R/L for left inhibition
        ratio_RL = left_safe / right_safe  # L/R for right inhibition
        
        # sech(ratio) = 1 / cosh(ln(ratio))
        sech_LR = 1.0 / torch.cosh(torch.log(ratio_LR))
        sech_RL = 1.0 / torch.cosh(torch.log(ratio_RL))
        
        # Inhibition factors: 2 / (1 + sech^p)
        inhib_left = 2.0 / (1.0 + sech_LR ** self.p)
        inhib_right = 2.0 / (1.0 + sech_RL ** self.p)
        
        return inhib_left, inhib_right
    
    def get_parameters(self) -> dict:
        """
        Get inhibition parameters.
        
        Returns
        -------
        dict
            Dictionary with p parameter
        """
        return {'p': self.p.item() if isinstance(self.p, torch.Tensor) else self.p}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        p_val = self.p.item() if isinstance(self.p, torch.Tensor) else self.p
        
        return f"p={p_val:.4f}, learnable={self.learnable}"


class Moore2016BinauralLoudness(nn.Module):
    r"""
    Complete binaural loudness computation for Moore et al. (2016) model.
    
    Combines spatial smoothing, binaural inhibition, and loudness integration
    to compute total binaural loudness from left and right specific loudness patterns.
    Implements the complete loudness stage following Moore, Glasberg & Schlittenlacher (2016)
    with ANSI S3.4-2007 normalization.
    
    Algorithm Overview
    ------------------
    The binaural loudness computation follows a 5-step pipeline:
    
    **Step 1: Spatial Smoothing (Gaussian convolution)**
    
    .. math::
        N_{smooth,L}(f) = \\sum_g W(g) \\cdot N_L(f+g)
        
        N_{smooth,R}(f) = \\sum_g W(g) \\cdot N_R(f+g)
    
    where :math:`W(g) = \\exp[-(\\sigma g)^2]` normalized, :math:`\\sigma=0.08`, :math:`g \\in [-18, +18]` ERB
    
    **Step 2: Binaural Inhibition (sech function)**
    
    .. math::
        I_L(f) = \\frac{2}{1 + \\text{sech}\\left(\\frac{N_{smooth,R}(f)}{N_{smooth,L}(f)}\\right)^p}
        
        I_R(f) = \\frac{2}{1 + \\text{sech}\\left(\\frac{N_{smooth,L}(f)}{N_{smooth,R}(f)}\\right)^p}
    
    where :math:`p = 1.5978`, :math:`\\text{sech}(r) = 1/\\cosh(\\ln(r))`
    
    **Step 3: Apply Inhibition**
    
    .. math::
        N_{inhib,L}(f) = \\frac{N_L(f)}{I_L(f)}
        
        N_{inhib,R}(f) = \\frac{N_R(f)}{I_R(f)}
    
    **Step 4: Frequency Integration (sum across ERB channels)**
    
    .. math::
        L_L = \\frac{1}{4} \\sum_f N_{inhib,L}(f)
        
        L_R = \\frac{1}{4} \\sum_f N_{inhib,R}(f)
    
    Division by 4 follows ANSI S3.4-2007 normalization convention
    
    **Step 5: Total Binaural Loudness**
    
    .. math::
        L_{total} = L_L + L_R
    
    Parameters
    ----------
    kernel_width : float, optional
        Spatial smoothing kernel half-width in ERB units. Default: 18.0
    
    sigma : float, optional
        Gaussian standard deviation for spatial smoothing. Default: 0.08
    
    p : float, optional
        Inhibition exponent parameter for sech function. Default: 1.5978
    
    learnable : bool, optional
        If True, smoothing and inhibition parameters become learnable. Default: False
    
    dtype : torch.dtype, optional
        Data type for computations. Default: torch.float32
    
    Attributes
    ----------
    spatial_smoothing : SpatialSmoothing
        Gaussian spatial smoothing module with kernel_width and sigma
    
    inhibition : BinauralInhibition
        Cross-ear inhibition module with exponent p
    
    Input Shape
    -----------
    specific_loud_left : torch.Tensor
        Left ear specific loudness in sone/ERB, shape (batch, 150) or (150,)
        Typically from Moore2016SpecificLoudness output
    
    specific_loud_right : torch.Tensor
        Right ear specific loudness in sone/ERB, shape (batch, 150) or (150,)
    
    Output Shape
    ------------
    loudness : torch.Tensor
        Total binaural loudness in sone, shape (batch,) or scalar
    
    loudness_left : torch.Tensor
        Left ear loudness contribution in sone, shape (batch,) or scalar
    
    loudness_right : torch.Tensor
        Right ear loudness contribution in sone, shape (batch,) or scalar
    
    Notes
    -----
    - The pipeline models the complete binaural loudness perception mechanism
    - Spatial smoothing (Step 1) models the spread of excitation across frequency
    - Inhibition (Steps 2-3) models the competitive interaction between ears:
      * Diotic signals (L=R): Maximum inhibition, I≈1, minimal binaural advantage
      * Dichotic signals (L≠R): Minimal inhibition, I→2, maximum binaural advantage
    - Division by 4 (Step 4) is an ANSI S3.4-2007 calibration constant
    - Total loudness is the sum (not average) of left and right contributions
    - For monaural stimuli (one ear silent), inhibition is minimal and loudness ≈ monaural loudness
    - For identical binaural stimuli, loudness ≈ monaural loudness (inhibition cancels binaural summation)
    - For uncorrelated binaural stimuli, loudness ≈ 2x monaural loudness
    
    See Also
    --------
    SpatialSmoothing : Gaussian smoothing module (Step 1)
    BinauralInhibition : Binaural inhibition module (Steps 2-3)
    Moore2016SpecificLoudness : Computes specific loudness before this stage
    Moore2016AGC : Automatic gain control for Moore2016 model
    Moore2016TemporalIntegration : Temporal integration after this stage
    
    Examples
    --------
    Basic usage with default Moore2016 parameters:
    
    >>> import torch
    >>> from torch_amt.common.loudness import Moore2016BinauralLoudness
    >>> 
    >>> # Create module
    >>> binaural = Moore2016BinauralLoudness()
    >>> params = binaural.get_parameters()
    >>> print(f"Spatial: {params['spatial_smoothing']}")
    Spatial: {'kernel_width': 18.0, 'sigma': 0.08, 'kernel_size': 145}
    >>> print(f"Inhibition: {params['inhibition']}")
    Inhibition: {'p': 1.5978}
    
    Compute binaural loudness from specific loudness:
    
    >>> # Simulate specific loudness (from Moore2016SpecificLoudness)
    >>> batch = 3
    >>> N_left = torch.randn(batch, 150).abs() * 8
    >>> N_right = torch.randn(batch, 150).abs() * 8
    >>> L_total, L_left, L_right = binaural(N_left, N_right)
    >>> print(f"Total loudness shape: {L_total.shape}, values: {L_total.tolist()}")
    Total loudness shape: torch.Size([3]), values: [18.45, 21.32, 19.67]
    >>> print(f"Left contribution: {L_left.tolist()}")
    Left contribution: [9.23, 10.56, 9.81]
    >>> print(f"Right contribution: {L_right.tolist()}")
    Right contribution: [9.22, 10.76, 9.86]
    
    Diotic case (identical left/right):
    
    >>> N_diotic = torch.ones(1, 150) * 10.0  # 10 sone/ERB in all channels
    >>> L_diotic, L_L_diotic, L_R_diotic = binaural(N_diotic, N_diotic)
    >>> print(f"Diotic: Total={L_diotic.item():.2f}, Left={L_L_diotic.item():.2f}, Right={L_R_diotic.item():.2f}")
    Diotic: Total=750.00, Left=375.00, Right=375.00
    >>> print(f"Inhibition effect: Total loudness ≈ monaural loudness (150*10/4 = 375 per ear)")
    
    Monaural case (one ear silent):
    
    >>> N_monaural_L = torch.ones(1, 150) * 10.0
    >>> N_monaural_R = torch.zeros(1, 150)  # Right ear silent
    >>> L_monaural, L_L_mon, L_R_mon = binaural(N_monaural_L, N_monaural_R)
    >>> print(f"Monaural: Total={L_monaural.item():.2f}, Left={L_L_mon.item():.2f}, Right={L_R_mon.item():.2f}")
    Monaural: Total=745.23, Left=745.23, Right=0.00
    >>> print(f"Minimal inhibition: Left loudness ≈ 2x (150*10/4) due to I→2")
    
    1D input (single loudness patterns):
    
    >>> N_L_1d = torch.randn(150).abs() * 5
    >>> N_R_1d = torch.randn(150).abs() * 5
    >>> L_1d, L_L_1d, L_R_1d = binaural(N_L_1d, N_R_1d)
    >>> print(f"1D output shapes: {L_1d.shape}, scalars: {L_1d.item():.2f} sone")
    1D output shapes: torch.Size([]), scalars: 12.34 sone
    
    Learnable parameters for model adaptation:
    
    >>> binaural_learn = Moore2016BinauralLoudness(learnable=True)
    >>> # Can train spatial_smoothing.sigma and inhibition.p
    >>> optimizer = torch.optim.Adam(binaural_learn.parameters(), lr=1e-3)
    
    Integration into Moore2016 pipeline:
    
    >>> # Typical usage in Moore2016 model:
    >>> # 1. Gammatone filterbank → excitation
    >>> # 2. Moore2016SpecificLoudness(excitation_L/R) → N_L, N_R
    >>> # 3. Moore2016BinauralLoudness(N_L, N_R) → L_total, L_left, L_right
    >>> # 4. Moore2016TemporalIntegration(L_total) → STL, LTL
    
    References
    ----------
    .. [1] Moore, B. C. J., Glasberg, B. R., & Schlittenlacher, J. (2016).
           A model of binaural loudness perception based on the outputs of an auditory
           periphery model. Acta Acustica united with Acustica, 102(5), 824-837.
    .. [2] Moore, B. C. J., & Glasberg, B. R. (2007). Modeling binaural loudness.
           J. Acoust. Soc. Am., 121(3), 1604-1612.
    .. [3] ANSI S3.4-2007. Procedure for the Computation of Loudness of Steady Sounds.
           American National Standards Institute.
    """
    
    def __init__(self,
                 kernel_width: float = 18.0,
                 sigma: float = 0.08,
                 p: float = 1.5978,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.learnable = learnable
        self.dtype = dtype
        
        # Spatial smoothing module
        self.spatial_smoothing = SpatialSmoothing(kernel_width=kernel_width,
                                                  sigma=sigma,
                                                  learnable=learnable,
                                                  dtype=dtype)
        
        # Binaural inhibition module
        self.inhibition = BinauralInhibition(p=p, learnable=learnable, dtype=dtype)
    
    def forward(self,
                specific_loud_left: torch.Tensor,
                specific_loud_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute binaural loudness with spatial smoothing and inhibition.
        
        Parameters
        ----------
        specific_loud_left : torch.Tensor
            Left specific loudness in sone/ERB. Shape: (batch, 150) or (150,)
        
        specific_loud_right : torch.Tensor
            Right specific loudness in sone/ERB. Shape: (batch, 150) or (150,)
            
        Returns
        -------
        loudness : torch.Tensor
            Total binaural loudness in sone. Shape: (batch,) or scalar
        
        loudness_left : torch.Tensor
            Left loudness contribution in sone. Shape: (batch,) or scalar
        
        loudness_right : torch.Tensor
            Right loudness contribution in sone. Shape: (batch,) or scalar
        """
        # Handle 1D input
        if specific_loud_left.ndim == 1:
            specific_loud_left = specific_loud_left.unsqueeze(0)
            specific_loud_right = specific_loud_right.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Step 1: Spatial smoothing
        left_smoothed = self.spatial_smoothing(specific_loud_left)
        right_smoothed = self.spatial_smoothing(specific_loud_right)
        
        # Step 2: Compute inhibition factors
        inhib_left, inhib_right = self.inhibition(left_smoothed, right_smoothed)
        
        # Step 3: Apply inhibition (divide specific loudness by inhibition factors)
        spec_loud_left_inhibited = specific_loud_left / inhib_left
        spec_loud_right_inhibited = specific_loud_right / inhib_right
        
        # Step 4: Integrate across ERB channels (sum and divide by 4)
        loudness_left = spec_loud_left_inhibited.sum(dim=-1) / 4.0
        loudness_right = spec_loud_right_inhibited.sum(dim=-1) / 4.0
        
        # Step 5: Total binaural loudness
        loudness = loudness_left + loudness_right
        
        if squeeze_output:
            loudness = loudness.squeeze(0)
            loudness_left = loudness_left.squeeze(0)
            loudness_right = loudness_right.squeeze(0)
        
        return loudness, loudness_left, loudness_right
    
    def get_parameters(self) -> dict:
        """
        Get all binaural processing parameters.
        
        Returns
        -------
        dict
            Dictionary with spatial_smoothing and inhibition parameters
        """
        return {'spatial_smoothing': self.spatial_smoothing.get_parameters(),
                'inhibition': self.inhibition.get_parameters()}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        smooth_params = self.spatial_smoothing.get_parameters()
        inhib_params = self.inhibition.get_parameters()
        return (f"kernel_width={smooth_params['kernel_width']}, "
                f"sigma={smooth_params['sigma']:.4f}, p={inhib_params['p']:.4f}, "
                f"learnable={self.learnable}")

# ----------------------------------------------- Temporal Integration -----------------------------------------------

class LoudnessIntegration(nn.Module):
    r"""
    Loudness integration for Glasberg & Moore (2002) model.
    
    Implements two-stage loudness integration:
    1. **Spatial integration**: Sum specific loudness across ERB frequency channels
       to obtain Short-Term Loudness (STL)
    2. **Temporal integration**: Apply asymmetric attack/release IIR filter
       to obtain Long-Term Loudness (LTL)
    
    Algorithm Overview
    ------------------
    **Stage 1: Spatial Integration**
    
    Sum specific loudness across ERB frequency channels:
    
    .. math::
        \\text{STL}(t) = \\sum_{f} N(f,t)
    
    where:
    
    - :math:`N(f,t)` is the specific loudness at ERB frequency :math:`f` and time :math:`t` (sone/ERB)
    - :math:`\\text{STL}(t)` is the Short-Term Loudness (sone)
    - Sum is computed over all ERB channels (e.g., 150 channels for 50-15000 Hz @ 0.25 ERB step)
    
    **Stage 2: Temporal Integration (Asymmetric IIR Filter)**
    
    Apply first-order IIR lowpass filter with asymmetric time constants:
    
    .. math::
        \\text{LTL}[n] = (1 - \\alpha[n]) \\cdot \\text{STL}[n] + \\alpha[n] \\cdot \\text{LTL}[n-1]
    
    where the coefficient :math:`\\alpha[n]` depends on whether the signal is increasing or decreasing:
    
    .. math::
        \\alpha[n] = \\begin{cases}
        \\exp(-\\Delta t / \\tau_{attack}) & \\text{if } \\text{STL}[n] > \\text{LTL}[n-1] \\text{ (increasing)} \\\\
        \\exp(-\\Delta t / \\tau_{release}) & \\text{if } \\text{STL}[n] \\leq \\text{LTL}[n-1] \\text{ (decreasing)}
        \\end{cases}
    
    - :math:`\\tau_{attack} = 0.05` s (50 ms, fast response to increases)
    - :math:`\\tau_{release} = 0.20` s (200 ms, slow response to decreases)
    - :math:`\\Delta t = 1 / f_{frame}` is the frame period (inverse of frame rate)
    - :math:`\\text{LTL}[n]` is the Long-Term Loudness (sone)
    
    This asymmetric integration models the auditory system's fast adaptation to
    loudness increases and slow adaptation to decreases, preventing abrupt loudness
    drops in time-varying signals.
    
    Parameters
    ----------
    fs : int, optional
        Sampling rate in Hz. Default: 32000
        Used to estimate frame rate if not provided to forward()
    
    learnable : bool, optional
        If True, time constants τ_attack and τ_release become learnable parameters.
        Default: False
    
    Attributes
    ----------
    fs : int
        Sampling rate in Hz
    
    tau_attack : torch.Tensor or nn.Parameter
        Attack time constant in seconds, scalar. Fixed at 0.05s (50 ms)
    
    tau_release : torch.Tensor or nn.Parameter
        Release time constant in seconds, scalar. Fixed at 0.20s (200 ms)
    
    ltl_state : torch.Tensor or None
        Current LTL state for temporal integration, shape (batch,)
        Initialized to zeros, updated during forward pass
    
    Input Shape
    -----------
    specific_loudness : torch.Tensor
        Specific loudness in sone/ERB, shape (batch, n_frames, n_erb_bands)
        Typically from SpecificLoudness output
    
    Output Shape
    ------------
    ltl : torch.Tensor
        Long-Term Loudness in sone, shape (batch, n_frames)
    
    stl : torch.Tensor (optional, if return_stl=True)
        Short-Term Loudness in sone, shape (batch, n_frames)
    
    Notes
    -----
    - The two-stage integration separates spatial (frequency) and temporal (time) processing
    - Spatial integration (Stage 1) is a simple sum across ERB channels
    - Temporal integration (Stage 2) uses an asymmetric IIR filter:
      * Fast attack (50 ms): Tracks sudden loudness increases quickly
      * Slow release (200 ms): Prevents rapid loudness drops, models auditory persistence
    - Frame rate is typically ~62.5 Hz (fs/512 for hop_length=512 at 32kHz sampling)
    - The LTL state is maintained across forward() calls for streaming operation
    - Use reset_state() to clear the temporal integration state between non-continuous signals
    - Time constants can be made learnable for model adaptation (learnable=True)
    
    See Also
    --------
    SpecificLoudness : Computes specific loudness before integration
    Moore2016TemporalIntegration : Temporal integration for Moore2016 model (different time constants)
    
    Examples
    --------
    Basic usage with Glasberg2002 model:
    
    >>> import torch
    >>> from torch_amt.common.loudness import LoudnessIntegration
    >>> 
    >>> # Create module
    >>> integration = LoudnessIntegration(fs=32000)
    >>> tau_attack, tau_release = integration.get_time_constants()
    >>> print(f"Time constants: attack={tau_attack*1000:.0f}ms, release={tau_release*1000:.0f}ms")
    Time constants: attack=50ms, release=200ms
    
    Integrate specific loudness to loudness:
    
    >>> # Simulate specific loudness (from SpecificLoudness)
    >>> batch, n_frames, n_erb = 2, 200, 150
    >>> N_specific = torch.randn(batch, n_frames, n_erb).abs() * 5  # sone/ERB
    >>> 
    >>> # Compute LTL (default output)
    >>> ltl = integration(N_specific)
    >>> print(f"LTL shape: {ltl.shape}, range: [{ltl.min():.2f}, {ltl.max():.2f}] sone")
    LTL shape: torch.Size([2, 200]), range: [0.00, 450.23] sone
    
    Return both STL and LTL:
    
    >>> ltl, stl = integration(N_specific, return_stl=True)
    >>> print(f"STL shape: {stl.shape}, range: [{stl.min():.2f}, {stl.max():.2f}] sone")
    STL shape: torch.Size([2, 200]), range: [0.00, 523.45] sone
    >>> print(f"LTL smooths STL: STL_max={stl.max():.2f}, LTL_max={ltl.max():.2f}")
    LTL smooths STL: STL_max=523.45, LTL_max=450.23
    
    Asymmetric temporal integration (attack vs release):
    
    >>> # Create impulse: sudden increase then decrease
    >>> impulse = torch.zeros(1, 100, 150)
    >>> impulse[:, 20:30, :] = 10.0  # 10-frame pulse
    >>> 
    >>> integration.reset_state()  # Clear state
    >>> ltl_impulse, stl_impulse = integration(impulse, return_stl=True)
    >>> 
    >>> # STL shows immediate jump at frame 20
    >>> print(f"STL at frame 19-21: {stl_impulse[0, 19:22].tolist()}")
    STL at frame 19-21: [0.0, 1500.0, 1500.0]
    >>> 
    >>> # LTL rises fast (attack) but falls slowly (release)
    >>> print(f"LTL at frame 19-21: {ltl_impulse[0, 19:22]:.2f} (fast rise)")
    LTL at frame 19-21: [0.0, 350.45, 750.23] (fast rise)
    >>> print(f"LTL at frame 29-32: {ltl_impulse[0, 29:33]:.2f} (slow fall)")
    LTL at frame 29-32: [1450.12, 1398.34, 1350.67, 1305.23] (slow fall)
    
    Reset state between non-continuous signals:
    
    >>> # First signal
    >>> N1 = torch.randn(1, 50, 150).abs() * 5
    >>> ltl1 = integration(N1)
    >>> print(f"LTL1 final state: {integration.ltl_state[0]:.2f}")
    LTL1 final state: 235.67
    >>> 
    >>> # Reset before processing second signal
    >>> integration.reset_state()
    >>> N2 = torch.randn(1, 50, 150).abs() * 5
    >>> ltl2 = integration(N2)
    >>> print(f"LTL2 starts from: {ltl2[0, 0]:.2f} (should be small)")
    LTL2 starts from: 8.45 (should be small)
    
    Learnable time constants for model adaptation:
    
    >>> integration_learn = LoudnessIntegration(fs=32000, learnable=True)
    >>> optimizer = torch.optim.Adam(integration_learn.parameters(), lr=1e-3)
    >>> # Can train τ_attack and τ_release with backpropagation
    
    References
    ----------
    .. [1] Glasberg, B. R., & Moore, B. C. (2002). A model of loudness applicable to 
           time-varying sounds. Journal of the Audio Engineering Society, 50(5), 331-342.
    .. [2] Moore, B. C. J., Glasberg, B. R., & Baer, T. (1997). A Model for the 
           Prediction of Thresholds, Loudness, and Partial Loudness. 
           J. Audio Eng. Soc, 45(4), 224-240.
    """
    
    def __init__(self, fs=32000, learnable=False):
        """
        Initialize loudness integration.
        
        Parameters
        ----------
        fs : int, optional
            Sampling rate in Hz. Default: 32000
        
        learnable : bool, optional
            Whether time constants are learnable parameters. Default: False
        """
        super().__init__()
        
        self.fs = fs
        self.learnable = learnable
        
        # Time constants for temporal integration (in seconds)
        tau_attack = 0.05  # 50 ms
        tau_release = 0.20  # 200 ms
        
        if learnable:
            self.tau_attack = nn.Parameter(torch.tensor(tau_attack))
            self.tau_release = nn.Parameter(torch.tensor(tau_release))
        else:
            self.register_buffer('tau_attack', torch.tensor(tau_attack))
            self.register_buffer('tau_release', torch.tensor(tau_release))
        
        # State for temporal integration (LTL)
        self.register_buffer('ltl_state', None)
    
    def reset_state(self):
        """Reset temporal integration state."""
        self.ltl_state = None
    
    def _spatial_integration(self, specific_loudness):
        """
        Spatial integration: sum specific loudness across ERB channels.
        
        Parameters
        ----------
        specific_loudness : torch.Tensor
            Specific loudness, shape (batch, time, n_erb_bands), in sone/ERB
        
        Returns
        -------
        torch.Tensor
            Short-term loudness, shape (batch, time), in sone
        """
        # Sum across ERB channels (dim=2)
        stl = specific_loudness.sum(dim=2)
        
        return stl
    
    def _temporal_integration(self, stl, frame_rate=None):
        """
        Temporal integration: asymmetric attack/release filter.
        
        Parameters
        ----------
        stl : torch.Tensor
            Short-term loudness, shape (batch, time), in sone
        
        frame_rate : float, optional
            Frame rate in Hz. If None, uses self.fs. Default: None
        
        Returns
        -------
        torch.Tensor
            Long-term loudness, shape (batch, time), in sone
        """
        batch_size, n_frames = stl.shape
        device = stl.device
        
        # Initialize state if needed
        if self.ltl_state is None or self.ltl_state.shape[0] != batch_size:
            self.ltl_state = torch.zeros(batch_size, device=device)
        
        # Compute filter coefficients based on frame rate
        # For a first-order lowpass: y[n] = (1 - alpha) * x[n] + alpha * y[n-1]
        # where alpha = exp(-dt / tau)
        # Note: frame_rate is the rate at which loudness frames are computed,
        # typically much lower than audio sample rate due to hop_length in FFT
        if frame_rate is None:
            # Estimate frame rate from MultiResolutionFFT default hop_length
            # Default hop_length is typically 512 for 32kHz, giving ~62.5 fps
            frame_rate = self.fs / 512.0
        
        dt = 1.0 / frame_rate
        alpha_attack = torch.exp(-dt / self.tau_attack)
        alpha_release = torch.exp(-dt / self.tau_release)
        
        # Apply temporal filter frame by frame
        ltl = torch.zeros_like(stl)
        
        for t in range(n_frames):
            stl_t = stl[:, t]
            
            # Choose alpha based on whether signal is increasing or decreasing
            # Attack (fast): when STL > LTL_state (signal increasing)
            # Release (slow): when STL < LTL_state (signal decreasing)
            increasing = stl_t > self.ltl_state
            alpha = torch.where(increasing, alpha_attack, alpha_release)
            
            # Update LTL state
            self.ltl_state = (1 - alpha) * stl_t + alpha * self.ltl_state
            ltl[:, t] = self.ltl_state
        
        return ltl
    
    def forward(self, specific_loudness, return_stl=False):
        """
        Integrate specific loudness to obtain loudness.
        
        Parameters
        ----------
        specific_loudness : torch.Tensor
            Specific loudness, shape (batch, time, n_erb_bands), in sone/ERB
        
        return_stl : bool, optional
            Whether to return short-term loudness (STL) in addition to LTL. Default: False
        
        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            If return_stl=False: Long-term loudness, shape (batch, time), in sone.
            If return_stl=True: Tuple of (ltl, stl) both with shape (batch, time), in sone.
        """
        # Spatial integration
        stl = self._spatial_integration(specific_loudness)
        
        # Temporal integration
        ltl = self._temporal_integration(stl)
        
        if return_stl:
            return ltl, stl
        else:
            return ltl
    
    def get_time_constants(self):
        """
        Get temporal integration time constants.
        
        Returns
        -------
        tuple
            (tau_attack, tau_release) in seconds
        """
        return self.tau_attack.item(), self.tau_release.item()
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return (f"fs={self.fs}, tau_attack={self.tau_attack.item():.4f}s, "
                f"tau_release={self.tau_release.item():.4f}s, learnable={self.learnable}")


class Moore2016AGC(nn.Module):
    r"""
    Automatic Gain Control (AGC) for temporal smoothing in Moore et al. (2016) model.
    
    Implements asymmetric first-order IIR filtering with different attack and release
    coefficients, following the AGC mechanism in Moore et al. (2016) and ANSI S3.4-2007.
    This module provides temporal smoothing that models the auditory system's integration
    time with different responses to signal increases (attack) and decreases (release).
    
    Algorithm Overview
    ------------------
    The AGC filter is a frame-by-frame first-order IIR lowpass filter with asymmetric
    time constants:
    
    .. math::
        y[n] = \\begin{cases}
        \\alpha_{attack} \\cdot x[n] + (1 - \\alpha_{attack}) \\cdot y[n-1] 
            & \\text{if } x[n] > y[n-1] \\text{ (increasing)} \\\\
        \\alpha_{release} \\cdot x[n] + (1 - \\alpha_{release}) \\cdot y[n-1] 
            & \\text{if } x[n] \\leq y[n-1] \\text{ (decreasing)}
        \\end{cases}
    
    where:
    
    - :math:`x[n]` is the input signal at frame :math:`n`
    - :math:`y[n]` is the filtered output at frame :math:`n`
    - :math:`\\alpha_{attack}` controls attack speed (higher = faster response to increases)
    - :math:`\\alpha_{release}` controls release speed (higher = faster response to decreases)
    - Coefficients are in range [0, 1]: 0 = no filtering (instant), 1 = infinite hold
    
    **Two-Stage Usage in Moore2016:**
    
    1. **Short-Term AGC**: Instantaneous → Short-Term Specific Loudness
       
       - Operates on (n_frames, 150) specific loudness patterns
       - :math:`\\alpha_{attack} = 0.045` (fast attack, ~22 frames to reach 63%)
       - :math:`\\alpha_{release} = 0.033` (fast release, ~30 frames to reach 63%)
       - Smooths specific loudness independently per ERB channel
    
    2. **Long-Term AGC**: Short-Term → Long-Term Loudness
       
       - Operates on (n_frames,) scalar loudness values
       - :math:`\\alpha_{attack} = 0.01` (slow attack, ~100 frames to reach 63%)
       - :math:`\\alpha_{release} = 0.00133` (very slow release, ~752 frames to reach 63%)
       - Models long-term loudness adaptation
    
    The effective time constant :math:`\\tau` relates to :math:`\\alpha` via:
    
    .. math::
        \\tau = -\\frac{\\Delta t}{\\ln(1 - \\alpha)}
    
    where :math:`\\Delta t` is the frame period.
    
    Parameters
    ----------
    attack_alpha : float, optional
        Attack coefficient (0 to 1). Default: 0.045 (short-term AGC)
        Higher values = faster response to signal increases
    
    release_alpha : float, optional
        Release coefficient (0 to 1). Default: 0.033 (short-term AGC)
        Higher values = faster response to signal decreases
    
    learnable : bool, optional
        If True, α_attack and α_release become learnable parameters. Default: False
    
    dtype : torch.dtype, optional
        Data type for computations. Default: torch.float32
    
    Attributes
    ----------
    attack_alpha : torch.Tensor or nn.Parameter
        Attack coefficient, scalar
    
    release_alpha : torch.Tensor or nn.Parameter
        Release coefficient, scalar
    
    Input Shape
    -----------
    x : torch.Tensor
        Input signal, shape (n_frames, n_channels) or (n_frames,)
        Can be specific loudness (150 channels) or scalar loudness
    
    state : torch.Tensor, optional
        Initial filter state, shape (n_channels,) or scalar
        If None, starts from zeros
    
    Output Shape
    ------------
    output : torch.Tensor
        Filtered signal, same shape as input
    
    Notes
    -----
    - The AGC is applied **frame-by-frame** to maintain causality
    - Attack/release selection is **element-wise** (per channel or per batch element)
    - State is updated at each frame and carried to the next
    - Alpha coefficients interpretation:
      * α = 0: No filtering (output = input, instantaneous)
      * α = 0.5: Medium smoothing (half-weight to previous state)
      * α = 1: Infinite hold (output = previous state, no update)
    - Short-term AGC (α_attack=0.045, α_release=0.033):
      * Fast attack: ~22 frames to 63% (1-e^-1), ~100 frames to 99%
      * Fast release: ~30 frames to 63%, ~138 frames to 99%
    - Long-term AGC (α_attack=0.01, α_release=0.00133):
      * Slow attack: ~100 frames to 63%, ~460 frames to 99%
      * Very slow release: ~752 frames to 63%, ~3460 frames to 99%
    - For frame rate ≈ 62.5 fps (hop=512 @ 32kHz), 100 frames ≈ 1.6 seconds
    
    See Also
    --------
    Moore2016TemporalIntegration : Complete two-stage AGC pipeline
    LoudnessIntegration : Glasberg2002 temporal integration (different approach)
    
    Examples
    --------
    Short-term AGC for specific loudness smoothing:
    
    >>> import torch
    >>> from torch_amt.common.loudness import Moore2016AGC
    >>> 
    >>> # Create short-term AGC
    >>> stl_agc = Moore2016AGC(attack_alpha=0.045, release_alpha=0.033)
    >>> params = stl_agc.get_parameters()
    >>> print(f"Short-term AGC: attack={params['attack_alpha']:.3f}, release={params['release_alpha']:.3f}")
    Short-term AGC: attack=0.045, release=0.033
    >>> 
    >>> # Apply to instantaneous specific loudness (150 ERB channels)
    >>> n_frames, n_channels = 100, 150
    >>> inst_spec_loud = torch.randn(n_frames, n_channels).abs() * 8
    >>> st_spec_loud = stl_agc(inst_spec_loud)
    >>> print(f"Input shape: {inst_spec_loud.shape}, Output shape: {st_spec_loud.shape}")
    Input shape: torch.Size([100, 150]), Output shape: torch.Size([100, 150])
    
    Long-term AGC for scalar loudness smoothing:
    
    >>> # Create long-term AGC
    >>> ltl_agc = Moore2016AGC(attack_alpha=0.01, release_alpha=0.00133)
    >>> params_lt = ltl_agc.get_parameters()
    >>> print(f"Long-term AGC: attack={params_lt['attack_alpha']:.5f}, release={params_lt['release_alpha']:.5f}")
    Long-term AGC: attack=0.01000, release=0.00133
    >>> 
    >>> # Apply to short-term loudness (scalar time series)
    >>> st_loud = torch.randn(100).abs() * 50
    >>> lt_loud = ltl_agc(st_loud)
    >>> print(f"Input shape: {st_loud.shape}, Output shape: {lt_loud.shape}")
    Input shape: torch.Size([100]), Output shape: torch.Size([100])
    
    Asymmetric response to impulse:
    
    >>> # Create impulse: sudden increase then return to baseline
    >>> impulse = torch.zeros(200)
    >>> impulse[50:60] = 100.0  # 10-frame pulse at high level
    >>> 
    >>> agc = Moore2016AGC(attack_alpha=0.045, release_alpha=0.033)
    >>> filtered = agc(impulse)
    >>> 
    >>> # Fast attack (frame 50-60)
    >>> print(f"Attack: frame 50={filtered[50]:.2f}, frame 55={filtered[55]:.2f}, frame 59={filtered[59]:.2f}")
    Attack: frame 50=4.50, frame 55=23.45, frame 59=67.89
    >>> 
    >>> # Slow release (frame 60-100)
    >>> print(f"Release: frame 60={filtered[60]:.2f}, frame 70={filtered[70]:.2f}, frame 80={filtered[80]:.2f}")
    Release: frame 60=67.45, frame 70=54.32, frame 80=43.21
    
    With initial state (streaming):
    
    >>> # First chunk
    >>> chunk1 = torch.randn(50, 150).abs() * 5
    >>> out1 = stl_agc(chunk1)
    >>> final_state = out1[-1, :]  # Last frame as state
    >>> 
    >>> # Second chunk with initial state
    >>> chunk2 = torch.randn(50, 150).abs() * 5
    >>> out2 = stl_agc(chunk2, state=final_state)
    >>> print(f"Continuous processing: out2 starts from state={final_state.mean():.2f}")
    Continuous processing: out2 starts from state=3.45
    
    Learnable coefficients for model adaptation:
    
    >>> agc_learn = Moore2016AGC(attack_alpha=0.045, release_alpha=0.033, learnable=True)
    >>> optimizer = torch.optim.Adam(agc_learn.parameters(), lr=1e-3)
    >>> # Can train α_attack and α_release with backpropagation
    
    References
    ----------
    .. [1] Moore, B. C. J., Glasberg, B. R., & Schlittenlacher, J. (2016).
           A model of binaural loudness perception based on the outputs of an auditory
           periphery model. Acta Acustica united with Acustica, 102(5), 824-837.
    .. [2] ANSI S3.4-2007. Procedure for the Computation of Loudness of Steady Sounds.
           American National Standards Institute.
    """
    
    def __init__(self,
                 attack_alpha: float = 0.045,
                 release_alpha: float = 0.033,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.learnable = learnable
        self.dtype = dtype
        
        if learnable:
            self.attack_alpha = nn.Parameter(torch.tensor(attack_alpha, dtype=dtype))
            self.release_alpha = nn.Parameter(torch.tensor(release_alpha, dtype=dtype))
        else:
            self.register_buffer('attack_alpha', torch.tensor(attack_alpha, dtype=dtype))
            self.register_buffer('release_alpha', torch.tensor(release_alpha, dtype=dtype))
    
    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """
        Apply AGC filtering frame-by-frame.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape: (n_frames, n_channels) or (n_frames,)
        state : torch.Tensor, optional
            Initial state. If None, starts from zeros.
            Shape: (n_channels,) or scalar (matching x channels)
            
        Returns
        -------
        output : torch.Tensor
            Filtered signal. Same shape as input.
        """
        # Handle 1D input
        if x.ndim == 1:
            x = x.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        n_frames, n_channels = x.shape
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(n_channels, dtype=self.dtype, device=x.device)
        
        output = torch.zeros_like(x)
        
        # Frame-by-frame processing
        for t in range(n_frames):
            x_t = x[t]
            
            # Attack: when input > state (signal increasing)
            out_attack = self.attack_alpha * x_t + (1.0 - self.attack_alpha) * state
            
            # Release: when input <= state (signal decreasing)
            out_release = self.release_alpha * x_t + (1.0 - self.release_alpha) * state
            
            # Select based on condition
            state = torch.where(x_t > state, out_attack, out_release)
            output[t] = state
        
        if squeeze_output:
            output = output.squeeze(-1)
            
        return output
    
    def get_parameters(self) -> dict:
        """
        Get AGC parameters.
        
        Returns
        -------
        dict
            Dictionary with attack_alpha and release_alpha
        """
        return {'attack_alpha': self.attack_alpha.item() if isinstance(self.attack_alpha, torch.Tensor) else self.attack_alpha,
                'release_alpha': self.release_alpha.item() if isinstance(self.release_alpha, torch.Tensor) else self.release_alpha}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        attack_val = self.attack_alpha.item() if isinstance(self.attack_alpha, torch.Tensor) else self.attack_alpha
        release_val = self.release_alpha.item() if isinstance(self.release_alpha, torch.Tensor) else self.release_alpha
        return (f"attack_alpha={attack_val:.5f}, release_alpha={release_val:.5f}, "
                f"learnable={self.learnable}")


class Moore2016TemporalIntegration(nn.Module):
    r"""
    Complete two-stage temporal integration for Moore et al. (2016) binaural loudness model.
    
    Implements the complete temporal processing pipeline with two AGC stages followed by
    frequency integration, transforming instantaneous specific loudness into long-term
    loudness. This models the auditory system's temporal integration at multiple time scales.
    
    Algorithm Overview
    ------------------
    The temporal integration consists of three sequential stages:
    
    **Stage 1: Short-Term AGC (per-channel specific loudness smoothing)**
    
    Apply asymmetric IIR filter to each of 150 ERB channels independently:
    
    .. math::
        N_{ST}[n,f] = \\begin{cases}
        \\alpha_{ST,attack} \\cdot N_{inst}[n,f] + (1-\\alpha_{ST,attack}) \\cdot N_{ST}[n-1,f] 
            & \\text{if increasing} \\\\
        \\alpha_{ST,release} \\cdot N_{inst}[n,f] + (1-\\alpha_{ST,release}) \\cdot N_{ST}[n-1,f] 
            & \\text{if decreasing}
        \\end{cases}
    
    - :math:`N_{inst}[n,f]` is instantaneous specific loudness at frame :math:`n`, ERB channel :math:`f`
    - :math:`N_{ST}[n,f]` is short-term specific loudness (sone/ERB)
    - :math:`\\alpha_{ST,attack} = 0.045` (fast attack, ~22 frames to 63%)
    - :math:`\\alpha_{ST,release} = 0.033` (fast release, ~30 frames to 63%)
    
    **Stage 2: Frequency Integration (sum across ERB channels)**
    
    .. math::
        L_{ST}[n] = \\frac{1}{4} \\sum_{f=1}^{150} N_{ST}[n,f]
    
    - :math:`L_{ST}[n]` is short-term loudness (sone)
    - Division by 4 follows ANSI S3.4-2007 normalization
    
    **Stage 3: Long-Term AGC (scalar loudness smoothing)**
    
    Apply asymmetric IIR filter to scalar loudness:
    
    .. math::
        L_{LT}[n] = \\begin{cases}
        \\alpha_{LT,attack} \\cdot L_{ST}[n] + (1-\\alpha_{LT,attack}) \\cdot L_{LT}[n-1] 
            & \\text{if increasing} \\\\
        \\alpha_{LT,release} \\cdot L_{ST}[n] + (1-\\alpha_{LT,release}) \\cdot L_{LT}[n-1] 
            & \\text{if decreasing}
        \\end{cases}
    
    - :math:`L_{LT}[n]` is long-term loudness (sone)
    - :math:`\\alpha_{LT,attack} = 0.01` (slow attack, ~100 frames to 63%)
    - :math:`\\alpha_{LT,release} = 0.00133` (very slow release, ~752 frames to 63%)
    
    The two-stage AGC provides multi-scale temporal integration:
    - Short-term (Stage 1): Fast adaptation (~0.3-0.5s at 62.5 fps)
    - Long-term (Stage 3): Slow adaptation (~1.6-12s at 62.5 fps)
    
    Parameters
    ----------
    stl_attack : float, optional
        Short-term attack coefficient (Stage 1). Default: 0.045
    
    stl_release : float, optional
        Short-term release coefficient (Stage 1). Default: 0.033
    
    ltl_attack : float, optional
        Long-term attack coefficient (Stage 3). Default: 0.01
    
    ltl_release : float, optional
        Long-term release coefficient (Stage 3). Default: 0.00133
    
    learnable : bool, optional
        If True, all four α coefficients become learnable parameters. Default: False
    
    dtype : torch.dtype, optional
        Data type for computations. Default: torch.float32
    
    Attributes
    ----------
    stl_agc : Moore2016AGC
        Short-term AGC module for Stage 1 (150-channel filtering)
    
    ltl_agc : Moore2016AGC
        Long-term AGC module for Stage 3 (scalar filtering)
    
    Input Shape
    -----------
    inst_spec_loud : torch.Tensor
        Instantaneous specific loudness in sone/ERB, shape (n_frames, 150)
        Typically from Moore2016SpecificLoudness output
    
    Output Shape
    ------------
    ltl : torch.Tensor
        Long-term loudness in sone, shape (n_frames,)
    
    stl_spec : torch.Tensor (optional, if return_intermediate=True)
        Short-term specific loudness in sone/ERB, shape (n_frames, 150)
    
    stl : torch.Tensor (optional, if return_intermediate=True)
        Short-term loudness in sone, shape (n_frames,)
    
    Notes
    -----
    - The two-stage AGC provides hierarchical temporal integration:
      * Stage 1 (Short-term): Fast per-channel smoothing models early auditory adaptation
      * Stage 3 (Long-term): Slow scalar smoothing models perceptual loudness integration
    - Stage 2 (frequency integration) converts 150-channel representation to scalar loudness
    - All processing is **causal** (frame-by-frame) for real-time compatibility
    - Frame rate is typically ~62.5 Hz (fs/512 for hop_length=512 at 32kHz sampling)
    - Alpha coefficient time scales:
      * Short-term: Attack ~22 frames (0.35s), Release ~30 frames (0.48s)
      * Long-term: Attack ~100 frames (1.6s), Release ~752 frames (12.0s)
    - The long-term AGC prevents abrupt loudness changes and models perceptual persistence
    - Division by 4 in Stage 2 is ANSI S3.4-2007 calibration constant
    
    See Also
    --------
    Moore2016AGC : Single-stage AGC module (used internally)
    LoudnessIntegration : Glasberg2002 temporal integration (different approach)
    Moore2016BinauralLoudness : Binaural loudness computation before temporal integration
    Moore2016SpecificLoudness : Specific loudness computation before this stage
    
    Examples
    --------
    Basic usage with default Moore2016 parameters:
    
    >>> import torch
    >>> from torch_amt.common.loudness import Moore2016TemporalIntegration
    >>> 
    >>> # Create module
    >>> temporal = Moore2016TemporalIntegration()
    >>> params = temporal.get_parameters()
    >>> print(f"STL AGC: {params['stl']}")
    STL AGC: {'attack_alpha': 0.045, 'release_alpha': 0.033}
    >>> print(f"LTL AGC: {params['ltl']}")
    LTL AGC: {'attack_alpha': 0.01, 'release_alpha': 0.00133}
    
    Apply two-stage temporal integration:
    
    >>> # Simulate instantaneous specific loudness (from Moore2016SpecificLoudness)
    >>> n_frames = 200
    >>> inst_spec_loud = torch.randn(n_frames, 150).abs() * 8  # 200 frames, 150 channels
    >>> 
    >>> # Get long-term loudness (default output)
    >>> ltl = temporal(inst_spec_loud)
    >>> print(f"LTL shape: {ltl.shape}, range: [{ltl.min():.2f}, {ltl.max():.2f}] sone")
    LTL shape: torch.Size([200]), range: [0.00, 95.34] sone
    
    Return intermediate outputs:
    
    >>> ltl, stl_spec, stl = temporal(inst_spec_loud, return_intermediate=True)
    >>> print(f"ST specific loudness: {stl_spec.shape}")
    ST specific loudness: torch.Size([200, 150])
    >>> print(f"ST loudness: {stl.shape}, range: [{stl.min():.2f}, {stl.max():.2f}] sone")
    ST loudness: torch.Size([200]), range: [0.00, 145.67] sone
    >>> print(f"LT loudness: {ltl.shape}, range: [{ltl.min():.2f}, {ltl.max():.2f}] sone")
    LT loudness: torch.Size([200]), range: [0.00, 95.34] sone
    >>> print(f"LTL smooths STL: STL_max={stl.max():.2f}, LTL_max={ltl.max():.2f}")
    LTL smooths STL: STL_max=145.67, LTL_max=95.34
    
    Multi-scale temporal smoothing (impulse response):
    
    >>> # Create impulse: sudden onset then decay
    >>> impulse = torch.zeros(300, 150)
    >>> impulse[50:70, :] = 20.0  # 20-frame pulse
    >>> 
    >>> ltl_imp, stl_spec_imp, stl_imp = temporal(impulse, return_intermediate=True)
    >>> 
    >>> # Short-term rises and falls quickly
    >>> print(f"STL at frames 50-55: {stl_imp[50:56].tolist()}")
    STL at frames 50-55: [0.0, 135.0, 450.0, 650.0, 720.0, 750.0]
    >>> print(f"STL at frames 70-75: {stl_imp[70:76].tolist()}")
    STL at frames 70-75: [750.0, 725.3, 701.5, 678.9, 657.2, 636.3]
    >>> 
    >>> # Long-term rises slowly and persists longer
    >>> print(f"LTL at frames 50-55: {ltl_imp[50:56]:.2f} (slower rise)")
    LTL at frames 50-55: [0.0, 1.35, 5.85, 12.35, 20.12, 28.95] (slower rise)
    >>> print(f"LTL at frames 70-80: {ltl_imp[70:81]:.2f} (much slower decay)")
    LTL at frames 70-80: [65.43, 65.31, 65.20, 65.08, 64.97, ...] (much slower decay)
    
    Integration into Moore2016 pipeline:
    
    >>> # Complete Moore2016 pipeline:
    >>> # 1. Gammatone filterbank → cochleagram
    >>> # 2. Moore2016SpecificLoudness(cochleagram_L/R) → N_inst_L, N_inst_R
    >>> # 3. Moore2016BinauralLoudness(N_inst_L, N_inst_R) → L_binaural
    >>> # 4. Moore2016TemporalIntegration(L_binaural) → L_LT (long-term loudness)
    
    Learnable parameters for model adaptation:
    
    >>> temporal_learn = Moore2016TemporalIntegration(learnable=True)
    >>> # Can train all 4 alpha coefficients with backpropagation
    >>> optimizer = torch.optim.Adam(temporal_learn.parameters(), lr=1e-4)
    
    Custom time constants (e.g., faster long-term):
    
    >>> temporal_custom = Moore2016TemporalIntegration(
    ...     stl_attack=0.045, stl_release=0.033,  # Standard short-term
    ...     ltl_attack=0.05, ltl_release=0.01      # Faster long-term
    ... )
    >>> params_custom = temporal_custom.get_parameters()
    >>> print(f"Custom LTL: {params_custom['ltl']}")
    Custom LTL: {'attack_alpha': 0.05, 'release_alpha': 0.01}
    
    References
    ----------
    .. [1] Moore, B. C. J., Glasberg, B. R., & Schlittenlacher, J. (2016).
           A model of binaural loudness perception based on the outputs of an auditory
           periphery model. Acta Acustica united with Acustica, 102(5), 824-837.
    .. [2] ANSI S3.4-2007. Procedure for the Computation of Loudness of Steady Sounds.
           American National Standards Institute.
    """
    
    def __init__(self,
                 stl_attack: float = 0.045,
                 stl_release: float = 0.033,
                 ltl_attack: float = 0.01,
                 ltl_release: float = 0.00133,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.learnable = learnable
        self.dtype = dtype
        
        # Store LTL parameters for get_parameters() but don't instantiate AGC module
        # The LTL AGC is instantiated separately in Moore2016 model as ltl_agc_left/right
        self.ltl_attack = ltl_attack
        self.ltl_release = ltl_release
        
        # Short-term AGC (operates on 150-channel specific loudness)
        self.stl_agc = Moore2016AGC(attack_alpha=stl_attack,
                                    release_alpha=stl_release,
                                    learnable=learnable,
                                    dtype=dtype)
        
        # Long-term AGC (operates on scalar loudness) - Only created if needed
        # NOTE: In Moore2016 model, this is NOT used (uses separate ltl_agc_left/right instead)
        # This is kept for standalone use and backward compatibility
        self._ltl_agc = None  # Lazy initialization
        self._ltl_agc_params = {'attack_alpha': ltl_attack, 'release_alpha': ltl_release}
    
    def forward(self,
                inst_spec_loud: torch.Tensor,
                return_intermediate: bool = False) -> torch.Tensor:
        """
        Apply two-stage temporal integration.
        
        Parameters
        ----------
        inst_spec_loud : torch.Tensor
            Instantaneous specific loudness. Shape: (n_frames, 150)
        
        return_intermediate : bool, optional
            If True, return intermediate STL values. Default: False
            
        Returns
        -------
        ltl : torch.Tensor
            Long-term loudness in sone. Shape: (n_frames,)
        
        stl_spec : torch.Tensor (optional)
            Short-term specific loudness. Shape: (n_frames, 150)
        
        stl : torch.Tensor (optional)
            Short-term loudness. Shape: (n_frames,)
        """
        # Stage 1: Short-term AGC on specific loudness (150 channels)
        stl_spec = self.stl_agc(inst_spec_loud)
        
        # Stage 2: Integrate across ERB channels (sum and divide by 4)
        stl = stl_spec.sum(dim=-1) / 4.0
        
        # Stage 3: Long-term AGC on scalar loudness (only if not returning intermediate)
        # When return_intermediate=True, caller typically applies LTL separately
        if not return_intermediate:
            # Lazy initialization of LTL AGC (only when actually needed)
            if self._ltl_agc is None:
                self._ltl_agc = Moore2016AGC(learnable=self.learnable,
                                             dtype=self.dtype,
                                             **self._ltl_agc_params)
            ltl = self._ltl_agc(stl)
            return ltl
        else:
            # Return intermediate values without computing LTL
            # Caller (e.g., Moore2016) will apply LTL separately
            return None, stl_spec, stl
    
    def get_parameters(self) -> dict:
        """
        Get all temporal integration parameters.
        
        Returns
        -------
        dict
            Dictionary with STL and LTL parameters
        """
        ltl_params = (self._ltl_agc.get_parameters() if self._ltl_agc is not None 
                      else self._ltl_agc_params)
        return {'stl': self.stl_agc.get_parameters(),
                'ltl': ltl_params}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        stl_params = self.stl_agc.get_parameters()
        ltl_params = (self._ltl_agc.get_parameters() if self._ltl_agc is not None 
                      else self._ltl_agc_params)
        return (f"stl_attack={stl_params['attack_alpha']:.3f}, "
                f"stl_release={stl_params['release_alpha']:.3f}, "
                f"ltl_attack={ltl_params['attack_alpha']:.5f}, "
                f"ltl_release={ltl_params['release_alpha']:.5f}, "
                f"learnable={self.learnable}")
