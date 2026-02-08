"""
Signal Processing and Filtering Utilities
==========================================

PyTorch-native implementations of filtering, interpolation, and signal processing
operations for auditory models with GPU acceleration and gradient flow support.

Author:
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

Contents
--------

**Signal Analysis & Processing:**
    - `torch_hilbert`: Hilbert transform via FFT for analytic signal computation
    - `torch_pchip_interp`: Piecewise Cubic Hermite interpolation (shape-preserving)

**IIR Filtering:**
    - `apply_iir_pytorch`: Direct Form II IIR filtering with PyTorch native operations
    - `IIRFilter`: nn.Module wrapper for learnable IIR filters
    - `_apply_iir_single`: Single-channel IIR filtering helper

**SOS (Second-Order Sections) Filtering:**
    - `apply_sos_pytorch`: Cascade of biquad filters for improved numerical stability
    - `SOSFilter`: nn.Module wrapper for learnable SOS filters
    - `_apply_sos_section_batch`: Batched SOS section processing helper

**FIR Filtering:**
    - `torch_firwin2`: FIR filter design via frequency sampling
    - `torch_minimum_phase`: Minimum-phase filter conversion via Hilbert transform
    - `torch_filtfilt`: Zero-phase forward-backward filtering

**Parametric Filters:**
    - `ButterworthFilter`: Learnable Butterworth lowpass/highpass filter

Design Philosophy
-----------------
- **GPU-Friendly**: All operations use PyTorch tensors for CUDA/MPS acceleration
- **Gradient-Safe**: No `.detach()`, `.cpu()`, or `.numpy()` in forward passes
- **Numerically Stable**: Direct Form II for IIR, SOS cascade for high-order filters
- **Learnable**: Filter parameters can be `nn.Parameter` for end-to-end training

Filter coefficient design (e.g., `scipy.signal.butter`) is performed only during
initialization using scipy for robustness. Forward passes use pure PyTorch operations
to maintain gradient flow and enable GPU acceleration.

See Also
--------
- `torch_amt.common.filterbanks`: Auditory filterbank implementations
- `torch_amt.common.ears`: Outer/middle ear filter implementations
"""

from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter

# -------------------------------------------------- Analysis -----------------------------------------------

def torch_hilbert(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Hilbert transform using FFT (PyTorch native).
    
    Equivalent to scipy.signal.hilbert but uses PyTorch operations.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape (..., N).
    
    Returns
    -------
    torch.Tensor
        Analytic signal (complex), shape (..., N).
    
    Notes
    -----
    Algorithm:
    1. FFT of input
    2. Zero out negative frequencies
    3. Double positive frequencies (except DC and Nyquist)
    4. IFFT to get analytic signal
    """
    # FFT
    X = torch.fft.fft(x, dim=-1)
    
    # Get signal length
    N = x.shape[-1]
    
    # Create mask for positive frequencies
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    
    # Apply mask
    X_analytic = X * h.to(X.dtype)
    
    # IFFT
    x_analytic = torch.fft.ifft(X_analytic, dim=-1)
    
    return x_analytic


def torch_pchip_interp(x: torch.Tensor, y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation (PyTorch native).
    
    Shape-preserving cubic interpolation that respects monotonicity of the data.
    Equivalent to scipy.interpolate.PchipInterpolator.
    
    Parameters
    ----------
    x : torch.Tensor
        X coordinates of data points, shape (N,), must be strictly increasing.
    
    y : torch.Tensor
        Y coordinates of data points, shape (N,).
    
    xi : torch.Tensor
        X coordinates for interpolation, shape (M,).
    
    Returns
    -------
    torch.Tensor
        Interpolated values, shape (M,).
    
    Notes
    -----
    Algorithm (Fritsch-Carlson method):
    
    1. Compute slopes between consecutive points:
       
       .. math::
           h_k = x_{k+1} - x_k
       
       .. math::
           \\delta_k = \\frac{y_{k+1} - y_k}{h_k}
    
    2. Compute derivatives at each point using weighted harmonic mean:
       
       .. math::
           d_k = \\frac{w_1 + w_2}{\\frac{w_1}{\\delta_{k-1}} + \\frac{w_2}{\\delta_k}}
       
       where :math:`w_1 = 2h_k + h_{k-1}`, :math:`w_2 = h_k + 2h_{k-1}`
    
    3. For each interval :math:`[x_k, x_{k+1}]`, use cubic Hermite polynomial:
       
       .. math::
           p(t) = y_k H_0(t) + y_{k+1} H_1(t) + h_k d_k H_2(t) + h_k d_{k+1} H_3(t)
       
       where :math:`t = (x - x_k)/h_k` and :math:`H_i` are Hermite basis functions.
    
    References
    ----------
    .. [1] Fritsch, F. N. and Carlson, R. E. (1980). "Monotone Piecewise Cubic Interpolation."
           SIAM Journal on Numerical Analysis, 17(2), 238-246.
    """
    device = x.device
    dtype = x.dtype
    n = len(x)
    
    # Compute slopes (delta_k = (y[k+1] - y[k]) / (x[k+1] - x[k]))
    h = x[1:] - x[:-1]  # h_k = x[k+1] - x[k]
    delta = (y[1:] - y[:-1]) / h  # delta_k
    
    # Compute derivatives at each point using PCHIP algorithm
    d = torch.zeros(n, device=device, dtype=dtype)
    
    # Endpoint derivatives (non-centered differences)
    # Left endpoint: use forward difference
    d[0] = delta[0]
    # Right endpoint: use backward difference
    d[-1] = delta[-1]
    
    # Interior points: Fritsch-Carlson weighted harmonic mean
    for k in range(1, n - 1):
        delta_km1 = delta[k - 1]
        delta_k = delta[k]
        
        # Check for sign change (non-monotonic region)
        if torch.sign(delta_km1) != torch.sign(delta_k):
            # Set derivative to zero at local extremum
            d[k] = 0.0
        else:
            # Weights based on interval lengths
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            
            # Weighted harmonic mean
            # d_k = (w1 + w2) / (w1/delta_{k-1} + w2/delta_k)
            d[k] = (w1 + w2) / (w1 / delta_km1 + w2 / delta_k)
    
    # Ensure shape preservation (monotonicity constraints)
    for k in range(n - 1):
        if delta[k] == 0:
            # Horizontal segment
            d[k] = 0.0
            d[k + 1] = 0.0
        else:
            # Check overshoot
            alpha = d[k] / delta[k]
            beta = d[k + 1] / delta[k]
            
            # Fritsch-Carlson condition: alpha^2 + beta^2 <= 9
            if alpha**2 + beta**2 > 9:
                tau = 3.0 / torch.sqrt(alpha**2 + beta**2)
                d[k] = tau * alpha * delta[k]
                d[k + 1] = tau * beta * delta[k]
    
    # Interpolate at query points
    yi = torch.zeros_like(xi)
    
    for i in range(len(xi)):
        xi_val = xi[i]
        
        # Handle extrapolation
        if xi_val <= x[0]:
            yi[i] = y[0]
        elif xi_val >= x[-1]:
            yi[i] = y[-1]
        else:
            # Find interval
            k = torch.searchsorted(x, xi_val) - 1
            k = min(k, n - 2)  # Ensure k is valid
            
            # Normalized position in interval [0, 1]
            t = (xi_val - x[k]) / h[k]
            
            # Hermite basis functions
            # H0(t) = 2t^3 - 3t^2 + 1
            # H1(t) = -2t^3 + 3t^2
            # H2(t) = t^3 - 2t^2 + t
            # H3(t) = t^3 - t^2
            t2 = t * t
            t3 = t2 * t
            
            H0 = 2 * t3 - 3 * t2 + 1
            H1 = -2 * t3 + 3 * t2
            H2 = t3 - 2 * t2 + t
            H3 = t3 - t2
            
            # Cubic Hermite interpolation
            yi[i] = y[k] * H0 + y[k + 1] * H1 + h[k] * d[k] * H2 + h[k] * d[k + 1] * H3
    
    return yi

# -------------------------------------------------- Filters ------------------------------------------------

class ButterworthFilter(nn.Module):
    """
    Butterworth IIR filter with GPU-accelerated application.
    
    Designs Butterworth filter coefficients using scipy.signal.butter (robust,
    validated implementation) in __init__, then applies filtering using PyTorch
    operations for GPU compatibility and optional gradient computation.
    
    The filter uses Second-Order Sections (SOS) representation for numerical
    stability, especially for higher-order filters and filters with extreme
    frequency characteristics.
    
    Parameters
    ----------
    order : int
        Filter order. Higher orders provide steeper roll-off but may be
        less stable. Typical values: 1-8.
    
    cutoff : float or tuple of float
        Cutoff frequency/frequencies in Hz:
        
        - For lowpass/highpass: single float (cutoff frequency)
        - For bandpass/bandstop: tuple of two floats (low, high)
    
    fs : float
        Sampling rate in Hz.
    
    btype : {'low', 'high', 'band', 'bandstop'}, optional
        Filter type:
        
        - 'low': Lowpass filter
        - 'high': Highpass filter  
        - 'band': Bandpass filter (requires cutoff as tuple)
        - 'bandstop': Bandstop filter (requires cutoff as tuple)
        
        Default: ``'low'``.
    
    learnable : bool, optional
        If True, SOS coefficients become trainable parameters (nn.Parameter).
        Allows task-specific optimization of filter characteristics.
        Default: ``False`` (fixed coefficients).
    
    dtype : torch.dtype, optional
        Data type for coefficients and computations. Default: torch.float32.
    
    Attributes
    ----------
    sos : torch.Tensor or nn.Parameter
        Second-order sections coefficients, shape (n_sections, 6).
        Each row: [b0, b1, b2, a0, a1, a2] for one biquad section.
    
    order : int
        Filter order.
    
    cutoff : float or tuple
        Cutoff frequency/frequencies.
    
    fs : float
        Sampling rate.
    
    btype : str
        Filter type.
    
    Shape
    -----
    - Input: :math:`(B, C, T)`, :math:`(C, T)`, or :math:`(T,)` where
        * :math:`B` = batch size (optional)
        * :math:`C` = channels (optional)
        * :math:`T` = time samples
    - Output: Same shape as input
    
    Examples
    --------
    >>> import torch
    >>> from torch_amt.common.filtering import ButterworthFilter
    >>> 
    >>> # Highpass filter (3 Hz, 1st order)
    >>> filt = ButterworthFilter(order=1, cutoff=3.0, fs=48000, btype='high')
    >>> signal = torch.randn(2, 5, 1000)  # (batch, channels, time)
    >>> output = filt(signal)
    >>> output.shape
    torch.Size([2, 5, 1000])
    >>> 
    >>> # Bandpass filter (10-100 Hz, 2nd order)
    >>> filt_bp = ButterworthFilter(order=2, cutoff=[10.0, 100.0], 
    ...                             fs=48000, btype='band')
    >>> output_bp = filt_bp(signal)
    >>> 
    >>> # Learnable filter for optimization
    >>> filt_learn = ButterworthFilter(order=2, cutoff=150.0, fs=48000, 
    ...                                 btype='low', learnable=True)
    >>> print(f"Trainable params: {sum(p.numel() for p in filt_learn.parameters())}")
    Trainable params: 6
    
    Notes
    -----
    **Design vs Apply:**
    
    - **Design**: scipy.signal.butter in __init__ (robust, validated)
    - **Apply**: PyTorch SOS filtering in forward (GPU compatible)
    
    **Numerical Stability:**
    
    SOS representation is more numerically stable than transfer function (ba)
    representation, especially for high-order filters. Each second-order section
    is applied sequentially, preventing accumulation of numerical errors.
    
    **Learnability:**
    
    When learnable=True, the SOS coefficients can be optimized during training.
    This allows the filter to adapt its frequency response for task-specific
    performance. However, learned coefficients may drift from valid Butterworth
    characteristics, so regularization may be needed.
    
    **GPU Acceleration:**
    
    All filtering operations are implemented in PyTorch, enabling:
    
    - GPU acceleration (CUDA/MPS)
    - Batch processing
    - Integration in differentiable pipelines
    
    See Also
    --------
    SOSFilter : Apply pre-computed SOS coefficients
    IIRFilter : Apply pre-computed ba coefficients
    """
    
    def __init__(self,
                 order: int,
                 cutoff: Union[float, Tuple[float, float], List[float]],
                 fs: float,
                 btype: str = 'low',
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        self.order = order
        self.cutoff = cutoff if isinstance(cutoff, (list, tuple)) else [cutoff]
        self.fs = fs
        self.btype = btype
        self.learnable = learnable
        self.dtype = dtype
        
        # Design filter using scipy (robust, validated)
        self._design_filter()
    
    def _design_filter(self):
        """Design Butterworth filter coefficients using scipy."""
        # Prepare cutoff for scipy
        if self.btype in ['low', 'high']:
            if len(self.cutoff) != 1:
                raise ValueError(f"'{self.btype}' filter requires single cutoff frequency")
            cutoff_scipy = self.cutoff[0]
        elif self.btype in ['band', 'bandstop']:
            if len(self.cutoff) != 2:
                raise ValueError(f"'{self.btype}' filter requires two cutoff frequencies")
            cutoff_scipy = list(self.cutoff)
        else:
            raise ValueError(f"Unknown btype: {self.btype}")
        
        # Design using scipy.signal.butter (gold standard)
        try:
            sos = butter(self.order, cutoff_scipy, btype=self.btype, 
                        fs=self.fs, output='sos')
        except Exception as e:
            raise ValueError(f"Failed to design Butterworth filter: {e}")
        
        # Convert to torch tensor
        sos_tensor = torch.tensor(sos, dtype=self.dtype)
        
        # Register as parameter or buffer
        if self.learnable:
            self.sos = nn.Parameter(sos_tensor)
        else:
            self.register_buffer('sos', sos_tensor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Butterworth filter to input signal.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape (batch, channels, time) or (channels, time) or (time,).
            
        Returns
        -------
        torch.Tensor
            Filtered signal, same shape as input.
        """
        # Use SOSFilter for actual filtering
        return apply_sos_pytorch(x, self.sos)
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        cutoff_str = f"{self.cutoff[0]:.1f}" if len(self.cutoff) == 1 else f"[{self.cutoff[0]:.1f}, {self.cutoff[1]:.1f}]"
        return (f"order={self.order}, cutoff={cutoff_str} Hz, fs={self.fs}, "
                f"btype={self.btype}, learnable={self.learnable}")


class SOSFilter(nn.Module):
    """
    Apply Second-Order Sections (SOS) filter.
    
    Applies pre-computed SOS coefficients to input signals using PyTorch
    operations for GPU compatibility. SOS representation provides better
    numerical stability than transfer function (ba) representation.
    
    Parameters
    ----------
    sos : torch.Tensor
        Second-order sections coefficients, shape (n_sections, 6).
        Each row: [b0, b1, b2, a0, a1, a2] for one biquad section.
        Can be obtained from scipy.signal.butter(..., output='sos').
    
    learnable : bool, optional
        If True, SOS coefficients become trainable. Default: ``False``.
    
    Attributes
    ----------
    sos : torch.Tensor or nn.Parameter
        SOS coefficients.
    
    n_sections : int
        Number of second-order sections.
    
    Shape
    -----
    - Input: :math:`(B, C, T)`, :math:`(C, T)`, or :math:`(T,)` where
        * :math:`B` = batch size (optional)
        * :math:`C` = channels (optional)
        * :math:`T` = time samples
    - Output: Same shape as input
    
    Examples
    --------
    >>> import torch
    >>> from scipy.signal import butter
    >>> from torch_amt.common.filtering import SOSFilter
    >>> 
    >>> # Design filter with scipy
    >>> sos_coeffs = butter(2, [10.0, 100.0], btype='band', fs=48000, output='sos')
    >>> sos_tensor = torch.tensor(sos_coeffs, dtype=torch.float32)
    >>> 
    >>> # Create filter
    >>> filt = SOSFilter(sos_tensor)
    >>> 
    >>> # Apply
    >>> signal = torch.randn(2, 5, 1000)
    >>> output = filt(signal)
    >>> output.shape
    torch.Size([2, 5, 1000])
    
    Notes
    -----
    Currently uses scipy.signal.sosfilt as backend for robustness.
    Pure PyTorch implementation coming in future version.
    
    See Also
    --------
    ButterworthFilter : Design and apply Butterworth filter
    IIRFilter : Apply b/a coefficients
    """
    
    def __init__(self, sos: torch.Tensor, learnable: bool = False):
        super().__init__()
        
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError(f"SOS must have shape (n_sections, 6), got {sos.shape}")
        
        self.n_sections = sos.shape[0]
        
        if learnable:
            self.sos = nn.Parameter(sos)
        else:
            self.register_buffer('sos', sos)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SOS filter.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape (batch, channels, time) or (channels, time) or (time,).
            
        Returns
        -------
        torch.Tensor
            Filtered signal, same shape as input.
        """
        return apply_sos_pytorch(x, self.sos)
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return f"n_sections={self.n_sections}"


class IIRFilter(nn.Module):
    """
    Apply IIR filter with ba coefficients.
    
    Applies pre-computed IIR filter coefficients (numerator b, denominator a)
    to input signals using PyTorch operations for GPU compatibility.
    
    For better numerical stability, consider using SOSFilter instead,
    especially for high-order filters.
    
    Parameters
    ----------
    b : torch.Tensor
        Numerator coefficients, shape (n_b,).
    
    a : torch.Tensor
        Denominator coefficients, shape (n_a,).
        First coefficient should be 1.0 (normalized).
    
    learnable : bool, optional
        If True, coefficients become trainable. Default: ``False``.
    
    Attributes
    ----------
    b : torch.Tensor or nn.Parameter
        Numerator coefficients.
    
    a : torch.Tensor or nn.Parameter
        Denominator coefficients.
    
    Shape
    -----
    - Input: :math:`(B, C, T)`, :math:`(C, T)`, or :math:`(T,)` where
        * :math:`B` = batch size (optional)
        * :math:`C` = channels (optional)
        * :math:`T` = time samples
    - Output: Same shape as input
    
    Examples
    --------
    >>> import torch
    >>> from scipy.signal import butter
    >>> from torch_amt.common.filtering import IIRFilter
    >>> 
    >>> # Design filter with scipy
    >>> b, a = butter(1, 3.0, btype='high', fs=48000)
    >>> b_tensor = torch.tensor(b, dtype=torch.float32)
    >>> a_tensor = torch.tensor(a, dtype=torch.float32)
    >>> 
    >>> # Create filter
    >>> filt = IIRFilter(b_tensor, a_tensor)
    >>> 
    >>> # Apply
    >>> signal = torch.randn(2, 5, 1000)
    >>> output = filt(signal)
    
    Notes
    -----
    Currently uses scipy.signal.lfilter as backend for robustness.
    Pure PyTorch implementation coming in future version.
    
    See Also
    --------
    ButterworthFilter : Design and apply Butterworth filter
    SOSFilter : Apply SOS coefficients (more stable)
    """
    
    def __init__(self, b: torch.Tensor, a: torch.Tensor, learnable: bool = False):
        super().__init__()
        
        if b.ndim != 1 or a.ndim != 1:
            raise ValueError("b and a must be 1D tensors")
        
        if learnable:
            self.b = nn.Parameter(b)
            self.a = nn.Parameter(a)
        else:
            self.register_buffer('b', b)
            self.register_buffer('a', a)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply IIR filter.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape (batch, channels, time) or (channels, time) or (time,).
            
        Returns
        -------
        torch.Tensor
            Filtered signal, same shape as input.
        """
        return apply_iir_pytorch(x, self.b, self.a)
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return f"n_b={len(self.b)}, n_a={len(self.a)}"

# ------------------------------------------------- Utilities ------------------------------------------------

def apply_sos_pytorch(x: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
    """
    Apply SOS filter using PyTorch native implementation.
    
    Applies Second-Order Sections filtering in PyTorch to maintain gradient flow.
    Each section is applied sequentially using Direct Form II structure.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape (..., time).
        Common shapes: (batch, channels, time), (channels, time), (time,).
    
    sos : torch.Tensor
        SOS coefficients, shape (n_sections, 6).
        Each row: [b0, b1, b2, a0, a1, a2]
    
    Returns
    -------
    torch.Tensor
        Filtered signal, same shape as input.
    
    Notes
    -----
    Pure PyTorch implementation maintains gradient flow for end-to-end training.
    Numerically equivalent to scipy.signal.sosfilt within machine precision (~1e-15).
    """
    original_shape = x.shape
    device = x.device
    dtype = x.dtype
    
    # Handle different input shapes
    if x.ndim == 1:
        # (time,) -> (1, 1, time)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        # (channels, time) -> (1, channels, time)
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError(f"Input must be 1D, 2D, or 3D, got shape {original_shape}")
    
    batch_size, num_channels, sig_len = x.shape
    n_sections = sos.shape[0]
    
    # Flatten batch and channels: (B, C, T) -> (B*C, T)
    x_flat = x.reshape(-1, sig_len)
    
    # Process each signal through all SOS sections
    y_flat = x_flat.clone()
    
    for section_idx in range(n_sections):
        # Extract coefficients for this section
        b0, b1, b2, a0, a1, a2 = sos[section_idx]
        
        # Normalize by a0
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        
        # Apply Direct Form II for all signals in parallel
        y_flat = _apply_sos_section_batch(y_flat, b0, b1, b2, a1, a2)
    
    # Reshape back
    y = y_flat.reshape(batch_size, num_channels, sig_len)
    
    # Restore original shape
    if len(original_shape) == 1:
        y = y.squeeze(0).squeeze(0)
    elif len(original_shape) == 2:
        y = y.squeeze(0)
    
    return y


def _apply_sos_section_batch(x: torch.Tensor, 
                             b0: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor,
                             a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """
    Apply single second-order section to batch of signals using Direct Form II.
    
    OPTIMIZED: Vectorized implementation that processes all signals in parallel
    instead of looping over each signal individually. Achieves ~6-7x speedup
    while maintaining identical output and gradient flow.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signals, shape (n_signals, time).
    
    b0, b1, b2 : torch.Tensor
        Numerator coefficients (scalars).
    
    a1, a2 : torch.Tensor
        Denominator coefficients (scalars).
    
    Returns
    -------
    torch.Tensor
        Filtered signals, shape (n_signals, time).
    """
    n_signals, T = x.shape
    device = x.device
    dtype = x.dtype
    
    # Pre-allocate output and state tensors
    y = torch.zeros_like(x)
    w1 = torch.zeros(n_signals, dtype=dtype, device=device)  # w[n-1]
    w2 = torch.zeros(n_signals, dtype=dtype, device=device)  # w[n-2]
    
    # Process all signals in parallel, one timestep at a time
    # This is still a loop over time, but processes all signals together
    for n in range(T):
        x_n = x[:, n]  # All signals at time n: (n_signals,)
        
        # Direct Form II state update (vectorized over signals)
        w0 = x_n - a1 * w1 - a2 * w2
        y[:, n] = b0 * w0 + b1 * w1 + b2 * w2
        
        # Update states
        w2 = w1.clone()
        w1 = w0.clone()
    
    return y


def apply_iir_pytorch(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Apply IIR filter using PyTorch native implementation.
    
    Applies standard IIR filtering with arbitrary order coefficients using
    Direct Form II Transposed structure to maintain gradient flow.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape (..., time).
    
    b : torch.Tensor
        Numerator coefficients.
    
    a : torch.Tensor
        Denominator coefficients.
    
    Returns
    -------
    torch.Tensor
        Filtered signal, same shape as input.
    
    Notes
    -----
    Pure PyTorch implementation maintains gradient flow for end-to-end training.
    Numerically equivalent to scipy.signal.lfilter within machine precision (~1e-15).
    
    For better numerical stability, use apply_sos_pytorch when possible.
    """
    original_shape = x.shape
    device = x.device
    dtype = x.dtype
    
    # Handle different input shapes
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError(f"Input must be 1D, 2D, or 3D, got shape {original_shape}")
    
    batch_size, num_channels, sig_len = x.shape
    
    # Flatten batch and channels
    x_flat = x.reshape(-1, sig_len)
    
    # Process each signal
    y_flat_list = []
    for i in range(x_flat.shape[0]):
        y_flat_list.append(_apply_iir_single(x_flat[i], b, a))
    
    y_flat = torch.stack(y_flat_list)
    
    # Reshape back
    y = y_flat.reshape(batch_size, num_channels, sig_len)
    
    # Restore original shape
    if len(original_shape) == 1:
        y = y.squeeze(0).squeeze(0)
    elif len(original_shape) == 2:
        y = y.squeeze(0)
    
    return y


def _apply_iir_single(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Apply IIR filter to single signal using Direct Form II Transposed.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape (T,).
    
    b : torch.Tensor
        Numerator coefficients, shape (n_b,).
    
    a : torch.Tensor
        Denominator coefficients, shape (n_a,).
    
    Returns
    -------
    torch.Tensor
        Filtered signal, shape (T,).
    """
    # Normalize by a[0]
    a0 = a[0]
    b = b / a0
    a = a / a0
    
    n_b = len(b)
    n_a = len(a)
    n_state = max(n_b, n_a) - 1
    
    if n_state == 0:
        # FIR filter (no feedback)
        return b[0] * x
    
    T = len(x)
    y_list = []
    state = torch.zeros(n_state, dtype=x.dtype, device=x.device)
    
    # Direct Form II Transposed
    for n in range(T):
        y_n = b[0] * x[n] + state[0] if n_state > 0 else b[0] * x[n]
        y_list.append(y_n)
        
        # Update state vector
        new_state = torch.zeros(n_state, dtype=x.dtype, device=x.device)
        for i in range(n_state - 1):
            b_i = b[i + 1] if i + 1 < n_b else torch.tensor(0.0, dtype=x.dtype, device=x.device)
            a_i = a[i + 1] if i + 1 < n_a else torch.tensor(0.0, dtype=x.dtype, device=x.device)
            new_state[i] = b_i * x[n] - a_i * y_n + state[i + 1]
        
        if n_state > 0:
            b_last = b[n_state] if n_state < n_b else torch.tensor(0.0, dtype=x.dtype, device=x.device)
            a_last = a[n_state] if n_state < n_a else torch.tensor(0.0, dtype=x.dtype, device=x.device)
            new_state[n_state - 1] = b_last * x[n] - a_last * y_n
        
        state = new_state
    
    return torch.stack(y_list)


def torch_firwin2(numtaps: int, freq: torch.Tensor, gain: torch.Tensor, fs: float = 2.0) -> torch.Tensor:
    """
    FIR filter design using frequency sampling method (PyTorch native).
    
    Equivalent to scipy.signal.firwin2 but uses PyTorch operations.
    
    Parameters
    ----------
    numtaps : int
        Number of filter coefficients (filter order + 1).
    
    freq : torch.Tensor
        Frequency points, shape (N,), in Hz (0 to fs/2).
    
    gain : torch.Tensor
        Desired gain at each frequency point, shape (N,).
    
    fs : float
        Sampling frequency in Hz. Default: 2.0 (normalized).
    
    Returns
    -------
    torch.Tensor
        FIR filter coefficients, shape (numtaps,).
    
    Notes
    -----
    Follows scipy.signal.firwin2 algorithm:
    1. Normalize frequencies to [0, π]
    2. Interpolate gain to uniformly spaced frequency grid
    3. Create symmetric spectrum for real-valued output
    4. IFFT and extract centered numtaps coefficients
    5. Apply Hamming window
    """
    device = freq.device
    dtype = freq.dtype
    
    # Normalize frequencies to [0, 1] (0 to Nyquist)
    freq_norm = freq / (fs / 2.0)
    
    # Determine FFT size (scipy uses next power of 2, minimum 512)
    nfft = max(512, int(2 ** torch.ceil(torch.log2(torch.tensor(float(numtaps), dtype=torch.float32))).item()))
    
    # Create uniform frequency grid from 0 to 1 (Nyquist)
    nfreqs = nfft // 2 + 1
    freq_grid = torch.linspace(0, 1, nfreqs, device=device, dtype=dtype)
    
    # Interpolate gains onto frequency grid (numpy interp equivalent)
    gains_grid = torch.zeros(nfreqs, device=device, dtype=dtype)
    
    for i in range(nfreqs):
        f = freq_grid[i]
        # Linear interpolation
        if f <= freq_norm[0]:
            gains_grid[i] = gain[0]
        elif f >= freq_norm[-1]:
            gains_grid[i] = gain[-1]
        else:
            # Find bracketing indices (keep as tensors for gradient!)
            idx = torch.searchsorted(freq_norm, f)
            f0 = freq_norm[idx - 1]
            f1 = freq_norm[idx]
            g0 = gain[idx - 1]
            g1 = gain[idx]
            # Interpolate (all tensor operations preserve gradient)
            gains_grid[i] = g0 + (g1 - g0) * (f - f0) / (f1 - f0 + 1e-10)
    
    # Build full symmetric spectrum for real-valued IFFT output
    # scipy does: fft_input = np.r_[gains_grid, gains_grid[-2:0:-1]]
    if nfft % 2 == 0:
        # Even length
        gains_full = torch.cat([gains_grid,  # [0, 1, ..., nfft/2]
                                torch.flip(gains_grid[1:-1], [0])  # [nfft/2-1, ..., 1]
                                ])
    else:
        # Odd length
        gains_full = torch.cat([gains_grid,  # [0, 1, ..., (nfft-1)/2]
                                torch.flip(gains_grid[1:], [0])  # [(nfft-1)/2, ..., 1]
                                ])
    
    # IFFT to get impulse response
    h_full = torch.fft.ifft(gains_full.to(torch.complex64)).real.to(dtype)
    
    # Extract centered numtaps coefficients
    h = torch.fft.fftshift(h_full)
    start = (nfft - numtaps) // 2
    h = h[start:start + numtaps]
    
    # Apply Hamming window
    window = torch.hamming_window(numtaps, periodic=False, device=device, dtype=dtype)
    h = h * window
    
    return h


def torch_minimum_phase(h: torch.Tensor) -> torch.Tensor:
    """
    Convert FIR filter to minimum phase (PyTorch native).
    
    Parameters
    ----------
    h : torch.Tensor
        Input impulse response, shape (N,).
    
    Returns
    -------
    torch.Tensor
        Minimum phase impulse response, shape (N,).
    
    Notes
    -----
    Algorithm:
    1. FFT of impulse response
    2. Log magnitude spectrum
    3. Hilbert transform to get minimum phase
    4. Reconstruct spectrum with minimum phase
    5. IFFT to get minimum phase impulse response
    """
    # FFT
    H = torch.fft.fft(h)
    
    # Log magnitude
    log_mag = torch.log(torch.abs(H) + 1e-10)
    
    # Hilbert transform to get minimum phase
    log_mag_analytic = torch_hilbert(log_mag)
    min_phase = -torch.imag(log_mag_analytic)
    
    # Reconstruct spectrum
    H_min = torch.abs(H) * torch.exp(1j * min_phase)
    
    # IFFT
    h_min = torch.real(torch.fft.ifft(H_min))
    
    return h_min


def torch_filtfilt(b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Zero-phase filtering using forward-backward filtering (PyTorch native).
    
    Equivalent to scipy.signal.filtfilt but uses PyTorch operations.
    
    Parameters
    ----------
    b : torch.Tensor
        FIR filter coefficients, shape (M,).
    
    x : torch.Tensor
        Input signal, shape (B, C, T).
    
    Returns
    -------
    torch.Tensor
        Filtered signal, shape (B, C, T).
    
    Notes
    -----
    Algorithm:
    1. Pad signal at boundaries (reflect mode)
    2. Forward filtering with FIR filter
    3. Reverse signal
    4. Forward filtering again
    5. Reverse back
    6. Remove padding
    
    This achieves zero-phase response by canceling the phase delay
    in the forward and backward passes.
    """
    B, C, T = x.shape
    device = x.device
    dtype = x.dtype
    
    # Pad length (3 * filter length is typical for filtfilt)
    pad_len = 3 * len(b)
    
    # Reflect padding at boundaries
    x_padded = F.pad(x, (pad_len, pad_len), mode='reflect')
    
    # Reshape for conv1d: (B*C, 1, T_padded)
    x_reshaped = x_padded.reshape(B * C, 1, -1)
    
    # Prepare filter: (1, 1, M)
    # Flip coefficients for true convolution (conv1d does cross-correlation)
    b_reshaped = b.flip(0).reshape(1, 1, -1).to(dtype=dtype, device=device)
    
    # Forward filtering
    y_forward = F.conv1d(x_reshaped, b_reshaped, padding=0)
    
    # Reverse signal
    y_reversed = torch.flip(y_forward, dims=[-1])
    
    # Backward filtering
    y_backward = F.conv1d(y_reversed, b_reshaped, padding=0)
    
    # Reverse back
    y_final = torch.flip(y_backward, dims=[-1])
    
    # Remove padding and reshape
    # Account for filter length reduction in both passes
    filter_delay = len(b) - 1
    total_delay = 2 * filter_delay
    
    # Extract valid region
    start_idx = pad_len - filter_delay
    end_idx = start_idx + T
    y = y_final[:, :, start_idx:end_idx]
    
    # Reshape back to (B, C, T)
    y = y.reshape(B, C, T)
    
    return y
