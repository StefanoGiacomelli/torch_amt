"""
Outer & Middle Ear Filters
===========================

Author: 
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements frequency-dependent filtering for the outer and middle ear 
transmission path in auditory models. Three main filter types are provided:

1. **HeadphoneFilter**: Combined headphone and outer ear response (Pralong & Carlile 1996)
2. **OuterMiddleEarFilter**: Outer and middle ear transfer functions (ANSI S3.4-2007)
3. **MiddleEarFilter**: Middle ear transmission characteristics (Lopez-Poveda 2001, Jepsen 2008)

These filters model the frequency-dependent gain of the auditory periphery from 
sound presentation (headphones or free/diffuse field) through the outer ear 
(pinna, ear canal) and middle ear (ossicles) to the cochlea.

The implementations use FIR filter design with optional minimum phase transformation 
and are compatible with the Auditory Modeling Toolbox (AMT) for MATLAB/Octave.

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

from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal

from .filters import torch_firwin2, torch_minimum_phase

# -------------------------------------------------- Data ----------------------------------------------------

# Data from Pralong & Carlile (1996), Figure 1(e)
# Gain of Sennheiser HD 250 Linear circumaural headphones
# Format: [frequency (Hz), gain (linear)]
PRALONG1996_DATA = torch.tensor([
    [125.0,         1.0],
    [250.0,         1.0],
    [500.0,         1.0],
    [1000.0,        0.994850557],
    [1237.384651,   0.994850557],
    [1531.120775,   0.994850557],
    [1894.585346,   1.114513162],
    [2002.467159,   1.235743262],
    [2344.330828,   1.867671314],
    [2721.273584,   2.822751493],
    [3001.403462,   2.180544843],
    [3589.453635,   1.442755787],
    [4001.342781,   1.173563859],
    [4441.534834,   1.37016005],
    [5004.212211,   1.599690164],
    [5495.887031,   1.37016005],
    [5997.423738,   1.114513162],
    [6800.526258,   0.648125625],
    [6946.931144,   0.631609176],
    [7995.508928,   0.276505667],
    [8414.866811,   0.084335217],
    [9008.422743,   0.084335217],
], dtype=torch.float32)

# Data from Lopez-Poveda & Meddis (2001), Figure 2b
# Based on Goode et al. (1994) stapes peak velocity at 0dB SPL
# Format: [frequency (Hz), stapes velocity (m/s)]
LOPEZPOVEDA2001_DATA = torch.tensor([
    [100.0,     1.181E-09],
    [200.0,     2.363E-09],
    [400.0,     4.728E-09],
    [600.0,     7.577E-09],
    [800.0,     1.000E-08],
    [1000.0,    8.235E-09],
    [1200.0,    6.240E-09],
    [1400.0,    5.585E-09],
    [1600.0,    5.000E-09],
    [1800.0,    4.232E-09],
    [2000.0,    3.787E-09],
    [2200.0,    3.000E-09],
    [2400.0,    2.715E-09],
    [2600.0,    2.498E-09],
    [2800.0,    2.174E-09],
    [3000.0,    1.893E-09],
    [3500.0,    1.742E-09],
    [4000.0,    1.516E-09],
    [4500.0,    1.117E-09],
    [5000.0,    1.320E-09],
    [5500.0,    1.214E-09],
    [6000.0,    9.726E-10],
    [6500.0,    9.460E-10],
    [7000.0,    8.705E-10],
    [7500.0,    8.000E-10],
    [8000.0,    7.577E-10],
    [8500.0,    7.168E-10],
    [9000.0,    6.781E-10],
    [9500.0,    6.240E-10],
    [10000.0,   6.000E-10],
], dtype=torch.float32)

# Data from Jepsen et al. (2008)
# Stapes impedance data (inverted to get velocity)
# Format: [frequency (Hz), impedance (inverse units)]
JEPSEN2008_DATA = torch.tensor([
    [50.0,      48046.39731],
    [100.0,     24023.19865],
    [200.0,     12011.59933],
    [400.0,     6005.799663],
    [600.0,     3720.406871],
    [800.0,     2866.404385],
    [1000.0,    3363.247811],
    [1200.0,    4379.228921],
    [1400.0,    4804.639731],
    [1600.0,    5732.808769],
    [1800.0,    6228.236688],
    [2000.0,    7206.959596],
    [2200.0,    9172.494031],
    [2400.0,    9554.681282],
    [2600.0,    10779.64042],
    [2800.0,    12011.59933],
    [3000.0,    14013.53255],
    [3500.0,    16015.46577],
    [4000.0,    18017.39899],
    [4500.0,    23852.82136],
    [5000.0,    21020.29882],
    [5500.0,    22931.23508],
    [6000.0,    28027.06509],
    [6500.0,    28745.70779],
    [7000.0,    32098.9],
    [7500.0,    34504.4],
    [8000.0,    36909.9],
    [8500.0,    39315.4],
    [9000.0,    41720.9],
    [9500.0,    44126.4],
    [10000.0,   46531.9],
], dtype=torch.float32)

FilterType = Literal['lopezpoveda2001', 'jepsen2008']

# ------------------------------------------ Headphone & Outer Ear -------------------------------------------

class HeadphoneFilter(nn.Module):
    r"""
    Combined headphone and outer ear filter.
    
    Implements a FIR filter approximating the combined frequency response of 
    headphones and the outer ear based on measurements from Pralong & Carlile (1996). 
    The filter emphasizes the 2-3 kHz region characteristic of outer ear resonance 
    and headphone coloration.
    
    This filter is commonly used in auditory models when sounds are presented via 
    headphones, compensating for the frequency-dependent transmission from the 
    headphone driver to the eardrum. The measurements are based on Sennheiser HD 250 
    Linear circumaural headphones.
    
    Algorithm Overview
    ------------------
    The filter design follows a frequency sampling approach:
    
    1. **Frequency response data**: Load empirical frequency-amplitude pairs from 
       Pralong & Carlile (1996) Figure 1(e)
    
    2. **Nyquist clipping** (if ``fs ≤ 20 kHz``): Remove data points above ``fs/2``
    
    3. **FIR design**: Use frequency sampling (``firwin2`` equivalent):
       
       .. math::
           H(\omega) = \text{fir2}(\text{order}, f_{\text{norm}}, A)
       
       where :math:`f_{\text{norm}} = f / (f_s/2)` and :math:`A` are amplitudes
    
    4. **Phase transformation** (optional):
       
       - **Zero-phase**: :math:`A_{\text{used}} = \sqrt{A}` (compensates for filtfilt squaring)
       - **Minimum-phase**: Apply Hilbert transform to log-magnitude spectrum
    
    5. **Filtering**:
       
       - **Zero-phase**: Forward-backward filtering (non-causal, no delay)
       - **Minimum-phase**: Convolution (causal, introduces delay)
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    
    order : int, optional
        FIR filter order (number of taps - 1). Higher orders provide better 
        frequency resolution but increase computational cost. Default: 512.
    
    phase_type : {'minimum', 'zero'}, optional
        Phase characteristic of the filter:
        
        * ``'minimum'``: Minimum phase filter (causal, frequency-dependent group delay)
        * ``'zero'``: Zero phase filter (non-causal, linear phase via filtfilt)
        
        Default: ``'minimum'``.
    
    learnable : bool, optional
        If True, the frequency response gains become trainable ``nn.Parameter`` objects. 
        The frequency grid remains fixed, but amplitudes can be adjusted during training. 
        Default: ``False``.
    
    compensate_delay : bool, optional
        If True, compensates for group delay by removing ``order//2`` samples. 
        Only applies to zero-phase filters. Default: ``True``.
    
    dtype : torch.dtype, optional
        Data type for filter coefficients and computations. Default: ``torch.float32``.
    
    Attributes
    ----------
    fir_coeffs : torch.Tensor
        FIR filter coefficients of shape ``[order+1]``.
    
    frequency_data : torch.Tensor or nn.Parameter
        Frequency response data of shape ``[N, 2]`` where column 0 contains 
        frequencies (Hz) and column 1 contains linear gains.
    
    group_delay : int
        Group delay in samples (``order // 2``).
    
    Shape
    -----
    - Input: :math:`(B, T)` or :math:`(B, F, T)` where
        * :math:`B` = batch size
        * :math:`F` = frequency channels (optional)
        * :math:`T` = time samples
    - Output: Same shape as input
    
    Notes
    -----
    **Filter Design:**
    
    - The frequency sampling method (``fir2``) creates a FIR filter matching 
      arbitrary frequency response specifications
    - For zero-phase filtering, ``filtfilt`` is used (forward-backward pass)
    - For minimum-phase, Hilbert transform converts linear-phase to minimum-phase
    
    **Delay Compensation:**
    
    - Zero-phase: No inherent delay (filtfilt is symmetric)
    - Minimum-phase: Causal with frequency-dependent group delay minimized
    - Compensation removes ``order//2`` samples to align signals
    
    **Learnable Parameters:**
    
    When ``learnable=True``, the filter is recomputed each forward pass to 
    reflect updated frequency response gains. This enables adaptive equalization 
    or personalized HRTF modeling.
    
    **Device Handling:**
    
    This module is device-agnostic and works on CPU, CUDA, and MPS.
    
    See Also
    --------
    MiddleEarFilter : Middle ear transmission filter
    OuterMiddleEarFilter : Combined outer and middle ear (ANSI S3.4-2007)
    
    References
    ----------
    .. [1] Pralong, D., & Carlile, S. (1996). The role of individualized 
           headphone calibration for the generation of high fidelity virtual 
           auditory space. *The Journal of the Acoustical Society of America*, 
           100(6), 3785-3793.
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> import torch
    >>> from torch_amt.common.ears import HeadphoneFilter
    >>> 
    >>> # Create filter for 16 kHz audio
    >>> hpf = HeadphoneFilter(fs=16000, order=512)
    >>> print(hpf)
    HeadphoneFilter(fs=16000, order=512, phase_type=minimum, learnable=False)
    >>> 
    >>> # Filter a batch of signals
    >>> signal = torch.randn(2, 16000)  # 2 channels, 1 second
    >>> filtered = hpf(signal)
    >>> print(f"Input: {signal.shape} -> Output: {filtered.shape}")
    Input: torch.Size([2, 16000]) -> Output: torch.Size([2, 16000])
    
    Multi-channel audio (e.g., from gammatone filterbank):
    
    >>> # Shape: (batch=4, channels=31, time=1600)
    >>> multichannel = torch.randn(4, 31, 1600)
    >>> filtered_multi = hpf(multichannel)
    >>> print(filtered_multi.shape)
    torch.Size([4, 31, 1600])
    
    Zero-phase filtering for offline processing:
    
    >>> hpf_zero = HeadphoneFilter(fs=16000, phase_type='zero')
    >>> # Non-causal, better frequency response but cannot be used in real-time
    >>> offline_filtered = hpf_zero(signal)
    
    Learnable filter for adaptive processing:
    
    >>> hpf_learn = HeadphoneFilter(fs=16000, learnable=True)
    >>> print(f"Learnable parameters: {sum(p.numel() for p in hpf_learn.parameters())}")
    Learnable parameters: 44
    >>> 
    >>> # Use in training loop
    >>> optimizer = torch.optim.Adam(hpf_learn.parameters(), lr=1e-3)
    >>> # Filter adapts during backpropagation
    
    Get frequency response:
    
    >>> freqs, H = hpf.get_frequency_response(nfft=8192)
    >>> magnitude_db = 20 * torch.log10(torch.abs(H) + 1e-10)
    >>> print(f"Freq range: [{freqs[0]:.1f}, {freqs[-1]:.1f}] Hz")
    Freq range: [0.0, 8000.0] Hz
    >>> print(f"Peak gain: {magnitude_db.max():.2f} dB at {freqs[magnitude_db.argmax()]:.1f} Hz")
    Peak gain: 9.02 dB at 2721.3 Hz
    """
    
    def __init__(self,
                 fs: float,
                 order: int = 512,
                 phase_type: str = 'minimum',
                 learnable: bool = False,
                 compensate_delay: bool = True,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        if phase_type not in ['minimum', 'zero']:
            raise ValueError(f"phase_type must be 'minimum' or 'zero', got {phase_type}")
        
        self.fs = float(fs)
        self.order = order
        self.phase_type = phase_type
        self.learnable = learnable
        self.compensate_delay = compensate_delay
        self.dtype = dtype
        self.group_delay = order // 2
        
        # Load and prepare frequency response data
        freq_data = PRALONG1996_DATA.clone().to(dtype)
        
        # Clip to Nyquist if needed
        if self.fs <= 20000:
            nyquist = self.fs / 2
            valid_indices = freq_data[:, 0] < nyquist
            freq_data = freq_data[valid_indices]
        
        # Store as parameter if learnable, otherwise as buffer
        if learnable:
            self.frequency_data = nn.Parameter(freq_data)
        else:
            self.register_buffer('frequency_data', freq_data)
        
        # Design FIR filter
        self._design_filter()
    
    def _design_filter(self, save_as_buffer=True):
        """
        Design the FIR filter from frequency response data.
        
        Parameters
        ----------
        save_as_buffer : bool
            If True, saves result as buffer (breaks gradient).
            If False, returns coefficients (preserves gradient).
        
        Returns
        -------
        torch.Tensor or None
            FIR filter coefficients if ``save_as_buffer=False``, else ``None``.
        """
        # Use torch operations for differentiability (instead of scipy+numpy)
        freq_data = self.frequency_data  # Keep as tensor
        
        # Prepare frequency and amplitude vectors for fir2
        # Format: [0, f1, f2, ..., fN, fs/2]
        device = freq_data.device
        dtype = freq_data.dtype
        
        frequencies = torch.cat([torch.zeros(1, device=device, dtype=dtype),
                                 freq_data[:, 0],
                                 torch.tensor([self.fs / 2], device=device, dtype=dtype)])
        
        # If zero-phase (filtfilt), use sqrt of amplitudes to compensate for squaring
        if self.phase_type == 'zero':
            amplitudes = torch.cat([torch.zeros(1, device=device, dtype=dtype),
                                    torch.sqrt(freq_data[:, 1]),
                                    torch.zeros(1, device=device, dtype=dtype)])
        else:
            amplitudes = torch.cat([torch.zeros(1, device=device, dtype=dtype),
                                    freq_data[:, 1],
                                    torch.zeros(1, device=device, dtype=dtype)])
        
        # Design FIR filter using torch_firwin2 (differentiable!)
        fir_coeffs = torch_firwin2(self.order + 1, frequencies, amplitudes, fs=self.fs)
        
        # Apply minimum phase transformation if requested
        if self.phase_type == 'minimum':
            # Use PyTorch native minimum phase transform (differentiable!)
            fir_coeffs = torch_minimum_phase(fir_coeffs)
        
        if save_as_buffer:
            # Store as buffer (breaks gradient, used for non-learnable mode)
            self.register_buffer('fir_coeffs', fir_coeffs)
            return None
        else:
            # Return coefficients (preserves gradient, used for learnable mode)
            return fir_coeffs
    
    def _minimum_phase(self, h: np.ndarray) -> np.ndarray:
        r"""
        Convert linear-phase FIR filter to minimum-phase using Hilbert transform.
        
        Minimum-phase filters have the shortest possible group delay for a given
        magnitude response, making them suitable for real-time causal processing.
        
        Parameters
        ----------
        h : np.ndarray
            Linear-phase FIR coefficients of shape ``[order+1]``.
        
        Returns
        -------
        np.ndarray
            Minimum-phase FIR coefficients of shape ``[order+1]``.
            
        Notes
        -----
        Algorithm:
        1. Compute FFT of impulse response: :math:`H(\omega) = \text{FFT}(h)`
        2. Extract log-magnitude: :math:`L(\omega) = \log|H(\omega)|`
        3. Apply Hilbert transform to get minimum phase: :math:`\phi_{\min}(\omega) = -\mathcal{H}\{L(\omega)\}`
        4. Reconstruct: :math:`H_{\min}(\omega) = |H(\omega)| e^{j\phi_{\min}(\omega)}`
        5. Return IFFT: :math:`h_{\min} = \text{IFFT}(H_{\min})`
        
        Regularization (``+1e-10``) prevents log(0) for near-zero magnitude components.
        """
        # Compute FFT of impulse response
        H = np.fft.fft(h)
        
        # Compute minimum phase via Hilbert transform of log magnitude
        log_mag = np.log(np.abs(H) + 1e-10)
        min_phase = -np.imag(scipy_signal.hilbert(log_mag))
        
        # Reconstruct minimum phase spectrum
        H_min = np.abs(H) * np.exp(1j * min_phase)
        
        # Convert back to time domain
        h_min = np.real(np.fft.ifft(H_min))
        
        return h_min
    
    def _torch_filtfilt(self, x: torch.Tensor, fir_coeffs: torch.Tensor) -> torch.Tensor:
        r"""
        Apply zero-phase filtering using forward-backward filtering (filtfilt).
        
        Implements the filtfilt algorithm in PyTorch for GPU acceleration.
        Zero-phase filtering eliminates phase distortion by processing the signal
        in both forward and reverse directions, effectively squaring the magnitude
        response while maintaining zero phase shift.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(B, C, T)`` where:
            
            - B = batch size
            - C = number of channels
            - T = time samples
        fir_coeffs : torch.Tensor
            FIR filter coefficients, shape ``(order+1,)``.
            
        Returns
        -------
        torch.Tensor
            Zero-phase filtered signal of shape ``(B, C, T)``.
            
        Notes
        -----
        Algorithm:
        1. Reflect-pad signal boundaries to minimize edge artifacts
        2. **Forward pass**: :math:`y_1(t) = h \ast x(t)`
        3. **Reverse**: :math:`y_2(t) = y_1(-t)`
        4. **Backward pass**: :math:`y_3(t) = h \ast y_2(t)`
        5. **Reverse back**: :math:`y(t) = y_3(-t)`
        
        The effective frequency response is :math:`|H(\omega)|^2` with zero phase.
        Reflection padding uses ``mode='reflect'`` with ``pad_len = order`` samples.
        """
        # Prepare kernel for convolution
        device = x.device
        dtype = x.dtype
        kernel = fir_coeffs.to(device=device, dtype=dtype).flip(0).view(1, 1, -1)
        
        B, C, T = x.shape
        x_flat = x.reshape(B * C, 1, T)
        
        # Reflect padding to handle boundaries
        pad_len = len(fir_coeffs) - 1
        x_padded = F.pad(x_flat, (pad_len, pad_len), mode='reflect')
        
        # Forward pass
        y_fwd = F.conv1d(x_padded, kernel, padding=0)
        
        # Reverse and filter again (backward pass)
        y_rev = torch.flip(y_fwd, dims=[2])
        y_rev_padded = F.pad(y_rev, (pad_len, pad_len), mode='reflect')
        y_bwd = F.conv1d(y_rev_padded, kernel, padding=0)
        
        # Reverse back
        y = torch.flip(y_bwd, dims=[2])
        
        # Reshape to original batch/channel structure
        y = y.reshape(B, C, -1)
        
        # Trim to original length (remove any extra samples from convolution)
        if y.shape[2] > T:
            start = (y.shape[2] - T) // 2
            y = y[:, :, start:start+T]
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply headphone filter to input signal.
        
        Filters the input through the combined headphone and outer ear frequency
        response. Automatically handles 2D (batch, time) and 3D (batch, channels, time)
        inputs. Refreshes filter coefficients if learnable parameters have changed.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape:
            
            - ``(B, T)``: Batch of time-domain signals
            - ``(B, F, T)``: Batch of multi-channel signals (e.g., from filterbank)
            
            where B = batch size, F = frequency channels, T = time samples.
        
        Returns
        -------
        torch.Tensor
            Filtered signal with same shape as input.
            
        Notes
        -----
        **Learnable filter update:**
        If ``learnable=True`` and ``frequency_data`` has changed since last call,
        the filter is redesigned automatically. This enables gradient-based adaptation.
        
        **Phase type behavior:**
        - **Minimum-phase**: Causal filtering via convolution, introduces group delay
        - **Zero-phase**: Non-causal via filtfilt, no phase distortion
        
        **Device handling:**
        Filter coefficients are automatically moved to match input device (CPU/CUDA/MPS).
        """
        # Compute filter coefficients (preserving gradient if learnable)
        if isinstance(self.frequency_data, nn.Parameter) and self.training:
            # Learnable mode: recompute filter each forward to preserve gradient
            fir_coeffs = self._design_filter(save_as_buffer=False)
        else:
            # Non-learnable mode: use cached buffer
            # Refresh filter if gains have changed
            if isinstance(self.frequency_data, nn.Parameter):
                if not hasattr(self, '_last_freq_data'):
                    self._design_filter(save_as_buffer=True)
                    self._last_freq_data = self.frequency_data.detach().clone()
                elif not torch.equal(self._last_freq_data, self.frequency_data):
                    self._design_filter(save_as_buffer=True)
                    self._last_freq_data = self.frequency_data.detach().clone()
            fir_coeffs = self.fir_coeffs
        
        original_shape = x.shape
        
        # Handle 2D input (B, T)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        
        # Use filtfilt for zero-phase or regular convolution for minimum-phase
        if self.phase_type == 'zero':
            # Zero-phase filtering (non-causal, no delay compensation needed)
            y = self._torch_filtfilt(x, fir_coeffs)
        else:
            # Minimum-phase filtering (causal)
            # Apply FIR filtering via convolution
            # fir_coeffs: [order+1] → [1, 1, order+1]
            # x: [B, num_channels, T]
            device = x.device
            dtype = x.dtype
            kernel = fir_coeffs.to(device=device, dtype=dtype).flip(0).view(1, 1, -1)
            
            B, num_channels, T = x.shape
            x_flat = x.reshape(B * num_channels, 1, T)
            y_flat = F.conv1d(x_flat, kernel, padding=self.order // 2)
            y = y_flat.reshape(B, num_channels, -1)
        
        # Restore original shape
        if len(original_shape) == 2:
            y = y.squeeze(1)
        
        return y
    
    def get_frequency_response(self, nfft: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frequency response of the filter.
        
        Parameters
        ----------
        nfft : int, optional
            Number of FFT points for frequency resolution. Default: 8192.
        
        Returns
        -------
        freqs : torch.Tensor
            Frequency vector in Hz of shape ``[nfft//2 + 1]``.
        
        response : torch.Tensor
            Complex frequency response of shape ``[nfft//2 + 1]``.
        """
        # Compute FFT of filter coefficients
        h_padded = F.pad(self.fir_coeffs, (0, nfft - len(self.fir_coeffs)))
        H = torch.fft.rfft(h_padded, n=nfft)
        freqs = torch.linspace(0, self.fs / 2, nfft // 2 + 1, device=H.device)
        
        return freqs, H
    
    def get_parameters(self) -> dict:
        """
        Get current filter parameters.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - ``'fs'``: Sampling rate in Hz
            - ``'order'``: FIR filter order
            - ``'phase_type'``: Phase characteristic ('minimum' or 'zero')
            - ``'learnable'``: Whether gains are trainable
            - ``'num_frequency_points'``: Number of frequency response points
            - ``'frequency_range'``: [min_freq, max_freq] in Hz
            - ``'compensate_delay'``: Whether delay compensation is applied
        """
        freq_data = self.frequency_data if not isinstance(self.frequency_data, nn.Parameter) \
                    else self.frequency_data.detach()
        
        return {'fs': self.fs,
                'order': self.order,
                'phase_type': self.phase_type,
                'learnable': self.learnable,
                'num_frequency_points': freq_data.shape[0],
                'frequency_range': [freq_data[0, 0].item(), freq_data[-1, 0].item()],
                'compensate_delay': self.compensate_delay}
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return (f'fs={self.fs}, order={self.order}, phase_type={self.phase_type}, '
                f'learnable={self.learnable}')

# -------------------------------------------- Outer & Middle Ear --------------------------------------------

# Following ANSI S3.4-2007
class OuterMiddleEarFilter(nn.Module):
    r"""
    Outer and middle ear transfer function filter.
    
    Implements FIR filtering based on the combined frequency-dependent transmission 
    characteristics of the outer ear (pinna, ear canal) and middle ear (tympanic 
    membrane, ossicular chain) transfer functions from ANSI S3.4-2007 and Moore et al. 
    (1997). The filter compensates for the acoustic gain from free-field or diffuse-field 
    sound pressure to the oval window.
    
    This filter is essential in auditory models to account for the pre-cochlear filtering 
    that shapes the acoustic input before neural transduction. The outer ear provides 
    resonance around 2-4 kHz, while the middle ear acts as an impedance matching network 
    between air and cochlear fluid.
    
    Algorithm Overview
    ------------------
    The filter design combines two transfer functions:
    
    1. **Outer ear transfer function** :math:`H_{\text{outer}}(f)`:
       
       - **Free field**: Direct sound from frontal incidence
       - **Diffuse field**: Sound from all directions (reverberant)
       - Provides gain boost at 2-4 kHz due to ear canal resonance
    
    2. **Middle ear transfer function** :math:`H_{\text{middle}}(f)`:
       
       - **tfOuterMiddle1997**: Moore et al. (1997) / Glasberg & Moore (2002)
       - **tfOuterMiddle2007**: ANSI S3.4-2007 standard
       - Attenuation at low frequencies (< 500 Hz) and high frequencies (> 8 kHz)
    
    3. **Combined transfer function** (dB):
       
       .. math::
           H_{\text{combined}}(f) = H_{\text{outer}}(f) + H_{\text{middle}}(f)
    
    4. **Interpolation**: PCHIP (Piecewise Cubic Hermite) from tabulated values 
       to dense frequency grid :math:`[f_{\text{low}}, f_{\text{high}}]`
    
    5. **FIR filter design**: Frequency sampling method (``firwin2``):
       
       .. math::
           h[n] = \text{IFFT}\{10^{H_{\text{combined}}(f)/10}\}
    
    6. **Zero-phase filtering**: Forward-backward pass (``filtfilt``):
       
       .. math::
           y(t) = \text{filtfilt}(h, x(t))
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    
    compensation_type : {'tfOuterMiddle1997', 'tfOuterMiddle2007'}, optional
        Transfer function version:
        
        * ``'tfOuterMiddle1997'``: Original from Moore et al. (1997), revised 2006. 
          Used in Glasberg & Moore (2002) loudness model. Based on Goode et al. (1994) 
          middle ear measurements.
        * ``'tfOuterMiddle2007'``: Updated version following ANSI S3.4-2007 standard. 
          Slight differences at mid frequencies (1-2 kHz).
        
        Default: ``'tfOuterMiddle1997'``.
    
    field_type : {'free', 'diffuse'}, optional
        Sound field type:
        
        * ``'free'``: Free-field presentation (anechoic, frontal incidence). 
          Typical for experiments with loudspeakers.
        * ``'diffuse'``: Diffuse-field presentation (reverberant, omnidirectional). 
          Typical for headphone equalization.
        
        Default: ``'free'``.
    
    flow : float, optional
        Lowest frequency for transfer function evaluation in Hz. Default: 20 Hz.
    
    fhigh : float, optional
        Highest frequency for transfer function evaluation in Hz. Default: 16000 Hz.
    
    order : int, optional
        FIR filter order (number of taps - 1). Higher orders provide better 
        frequency resolution but increase computational cost and latency. 
        Default: 4096.
    
    learnable : bool, optional
        If True, the transfer function gains become trainable ``nn.Parameter`` objects. 
        The frequency grid remains fixed, but the dB gains can be adjusted during 
        training for adaptive equalization. Default: ``False``.
    
    dtype : torch.dtype, optional
        Data type for filter coefficients and computations. Default: ``torch.float32``.
        
    Attributes
    ----------
    tf_gains : torch.Tensor or nn.Parameter
        Transfer function gains in dB of shape ``[num_freqs]``, where 
        ``num_freqs = fhigh - flow + 1`` (1 Hz spacing). 
        If ``learnable=True``, this is a trainable parameter.
    
    fvec : torch.Tensor
        Frequency vector in Hz of shape ``[num_freqs]`` from ``flow`` to ``fhigh``.
    
    fir_coeffs : torch.Tensor
        FIR filter coefficients of shape ``[order+1]``.
    
    Shape
    -----
    - Input: :math:`(B, T)` where
        * :math:`B` = batch size
        * :math:`T` = time samples
    - Output: :math:`(B, T)` - Filtered signal with same shape as input
    
    Notes
    -----
    **Transfer Function Differences:**
    
    - **tfOuterMiddle1997**: Revised data from 2006 based on Goode et al. (1994)
    - **tfOuterMiddle2007**: ANSI S3.4-2007 with updated middle ear values at 1.25 kHz
    - Main difference: Middle ear gain at 1250 Hz (3.2 dB vs 4.5 dB)
    
    **Field Type Impact:**
    
    - **Free field**: Higher gain at 2-4 kHz (pinna resonance peak ~12 dB at 2 kHz)
    - **Diffuse field**: More uniform gain distribution (peak ~10 dB at 2 kHz)
    
    **Filter Characteristics:**
    
    - Zero-phase filtering via ``filtfilt`` (non-causal, no phase distortion)
    - Dense frequency sampling (1 Hz resolution) for accurate transfer function
    - PCHIP interpolation preserves monotonicity and smoothness
    
    **Learnable Parameters:**
    
    When ``learnable=True``, the filter can adapt to:
    - Individual anatomical differences (personalized HRTFs)
    - Specific headphone frequency responses
    - Equalization for particular acoustic environments
    
    **Device Handling:**
    
    This module is device-agnostic and works on CPU, CUDA, and MPS.
    
    See Also
    --------
    HeadphoneFilter : Headphone and outer ear response (Pralong & Carlile 1996)
    MiddleEarFilter : Middle ear only (Lopez-Poveda & Meddis 2001)
    
    References
    ----------
    .. [1] ANSI S3.4-2007, "Procedure for the Computation of Loudness of Steady Sounds," 
           American National Standards Institute, 2007.

    .. [2] Moore, B. C. J., Glasberg, B. R., & Baer, T. (1997). A model for the 
           prediction of thresholds, loudness, and partial loudness. 
           *Journal of the Audio Engineering Society*, 45(4), 224-240.

    .. [3] Glasberg, B. R., & Moore, B. C. J. (2002). A model of loudness applicable 
           to time-varying sounds. *Journal of the Audio Engineering Society*, 
           50(5), 331-342.

    .. [4] Goode, R. L., Killion, M., Nakamura, K., & Nishihara, S. (1994). 
           New knowledge about the function of the human middle ear: Development of 
           an improved analog model. *American Journal of Otology*, 15(2), 145-154.
    
    Examples
    --------
    Basic usage with default parameters (1997 version, free field):
    
    >>> import torch
    >>> from torch_amt.common.ears import OuterMiddleEarFilter
    >>> 
    >>> # Create filter for 32 kHz audio
    >>> omef = OuterMiddleEarFilter(fs=32000, compensation_type='tfOuterMiddle1997', 
    ...                              field_type='free')
    >>> print(omef)
    OuterMiddleEarFilter(fs=32000, type='tfOuterMiddle1997', field='free')
    >>> 
    >>> # Filter a signal (1 second stereo)
    >>> signal = torch.randn(2, 32000)
    >>> filtered = omef(signal)
    >>> print(f"Input: {signal.shape} -> Output: {filtered.shape}")
    Input: torch.Size([2, 32000]) -> Output: torch.Size([2, 32000])
    
    Compare 1997 vs 2007 versions:
    
    >>> omef_1997 = OuterMiddleEarFilter(fs=32000, compensation_type='tfOuterMiddle1997')
    >>> omef_2007 = OuterMiddleEarFilter(fs=32000, compensation_type='tfOuterMiddle2007')
    >>> 
    >>> # Get transfer functions
    >>> freqs_1997, tf_1997 = omef_1997.get_transfer_function()
    >>> freqs_2007, tf_2007 = omef_2007.get_transfer_function()
    >>> 
    >>> # Main difference at 1.25 kHz
    >>> idx = (freqs_1997 - 1250).abs().argmin()
    >>> print(f"1997 at 1250 Hz: {tf_1997[idx]:.2f} dB")
    1997 at 1250 Hz: 0.60 dB
    >>> print(f"2007 at 1250 Hz: {tf_2007[idx]:.2f} dB")
    2007 at 1250 Hz: 1.90 dB
    
    Free field vs diffuse field:
    
    >>> omef_free = OuterMiddleEarFilter(fs=32000, field_type='free')
    >>> omef_diff = OuterMiddleEarFilter(fs=32000, field_type='diffuse')
    >>> 
    >>> # Diffuse field has less pronounced resonance peak
    >>> x = torch.randn(1, 32000)
    >>> y_free = omef_free(x)
    >>> y_diff = omef_diff(x)
    
    Learnable filter for adaptive processing:
    
    >>> omef_learn = OuterMiddleEarFilter(fs=32000, learnable=True)
    >>> print(f"Learnable parameters: {sum(p.numel() for p in omef_learn.parameters())}")
    Learnable parameters: 15981
    >>> 
    >>> # Use in training loop with optimizer
    >>> optimizer = torch.optim.Adam(omef_learn.parameters(), lr=1e-4)
    >>> # Transfer function adapts during backpropagation
    
    Get frequency response:
    
    >>> freqs, H_db = omef.get_frequency_response(nfft=8192)
    >>> print(f"Freq range: [{freqs[0]:.1f}, {freqs[-1]:.1f}] Hz")
    Freq range: [0.0, 16000.0] Hz
    >>> print(f"Peak gain: {H_db.max():.2f} dB at {freqs[H_db.argmax()]:.1f} Hz")
    Peak gain: 16.41 dB at 2978.5 Hz
    """
    
    def __init__(self,
                 fs: float,
                 compensation_type: Literal['tfOuterMiddle1997', 'tfOuterMiddle2007'] = 'tfOuterMiddle1997',
                 field_type: Literal['free', 'diffuse'] = 'free',
                 flow: float = 20.0,
                 fhigh: float = 16000.0,
                 order: int = 4096,
                 learnable: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        self.fs = fs
        self.compensation_type = compensation_type
        self.field_type = field_type
        self.flow = flow
        self.fhigh = fhigh
        self.order = order
        self.learnable = learnable
        self.dtype = dtype
        
        # Generate frequency vector
        self.fvec = torch.arange(flow, fhigh + 1, dtype=dtype)
        
        # Get transfer function data
        tf_outer, tf_middle = self._get_transfer_functions()
        
        # Combined transfer function (outer + middle ear)
        tf_combined = tf_outer + tf_middle
        
        # Store as parameter or buffer
        if learnable:
            self.tf_gains = nn.Parameter(tf_combined)
        else:
            self.register_buffer('tf_gains', tf_combined)
        
        # Design FIR filter
        self._design_filter()
        
    def _get_transfer_functions(self):
        """
        Get outer and middle ear transfer functions based on settings.
        
        Loads empirical transfer function data from ANSI S3.4-2007 tables and
        interpolates to the dense frequency grid using PCHIP interpolation.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - **tf_outer_interp**: Outer ear transfer function in dB, shape ``[num_freqs]``
            - **tf_middle_interp**: Middle ear transfer function in dB, shape ``[num_freqs]``
            
        Notes
        -----
        **Data sources:**
        
        - **Outer ear**: ANSI S3.4 Tables B.1 (free field) and B.2 (diffuse field)
        - **Middle ear**: tfOuterMiddle1997 (Moore 1997, revised 2006) or tfOuterMiddle2007 (ANSI S3.4-2007)
        
        **Key differences:**
        
        - At 1250 Hz: 1997 = -2.6 dB, 2007 = -4.5 dB (middle ear)
        - Free field has stronger pinna resonance (~12 dB at 2 kHz)
        - Diffuse field more uniform (~10 dB at 2 kHz)
        
        PCHIP interpolation ensures smooth, monotonic transitions between tabulated points.
        """
        # Outer ear transfer function frequencies (Hz)
        f_outer = torch.tensor([
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
            630, 750, 800, 1000, 1250, 1500, 1600, 2000, 2500, 3000, 3150, 4000, 
            5000, 6000, 6300, 8000, 9000, 10000, 11200, 12500, 14000, 15000, 16000, 20000
        ], dtype=self.dtype)
        
        if self.field_type == 'free':
            # Free field transfer function (dB)
            tf_outer_vals = torch.tensor([
                0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.9, 1.4, 1.6, 1.7, 2.5, 2.7, 
                2.6, 2.6, 3.2, 5.2, 6.6, 12, 16.8, 15.3, 15.2, 14.2, 10.7, 7.1, 6.4, 
                1.8, -0.9, -1.6, 1.9, 4.9, 2, -2, 2.5, 2.5
            ], dtype=self.dtype)
        elif self.field_type == 'diffuse':
            # Diffuse field transfer function (dB)
            tf_outer_vals = torch.tensor([
                0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.4, 0.5, 1, 1.6, 1.7, 2.2, 2.7, 
                2.9, 3.8, 5.3, 6.8, 7.2, 10.2, 14.9, 14.5, 14.4, 12.7, 10.8, 8.9, 
                8.7, 8.5, 6.2, 5, 4.5, 4, 3.3, 2.6, 2, 2
            ], dtype=self.dtype)
        else:
            raise ValueError(f"Invalid field_type: {self.field_type}. Use 'free' or 'diffuse'.")
        
        # Middle ear transfer function frequencies (Hz)
        f_middle = torch.tensor([
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
            630, 750, 800, 1000, 1250, 1500, 1600, 2000, 2500, 3000, 3150, 4000, 
            5000, 6000, 6300, 8000, 9000, 10000, 11200, 12500, 14000, 15000, 16000, 
            18000, 20000
        ], dtype=self.dtype)
        
        if self.compensation_type == 'tfOuterMiddle1997':
            # Revised data 2006 (used in Glasberg 2002)
            tf_middle_vals = -torch.tensor([
                39.6, 32, 25.85, 21.4, 18.5, 15.9, 14.1, 12.4, 11, 9.6, 8.3, 7.4, 6.2, 
                4.8, 3.8, 3.3, 2.9, 2.6, 2.6, 3.2, 4.5, 5.5, 8.5, 10.4, 7.3, 7, 6.6, 7, 
                9.2, 10.2, 12.2, 10.8, 10.1, 12.7, 15, 18.2, 23.8, 32.3, 45.5, 50
            ], dtype=self.dtype)
        elif self.compensation_type == 'tfOuterMiddle2007':
            # ANSI S3.4-2007 values
            tf_middle_vals = -torch.tensor([
                39.6, 32, 25.85, 21.4, 18.5, 15.9, 14.1, 12.4, 11, 9.6, 8.3, 7.4, 6.2, 
                4.8, 3.8, 3.3, 2.9, 2.6, 2.6, 4.5, 5.4, 6.1, 8.5, 10.4, 7.3, 7, 6.6, 7, 
                9.2, 10.2, 12.2, 10.8, 10.1, 12.7, 15, 18.2, 23.8, 32.3, 45.5, 50
            ], dtype=self.dtype)
        else:
            raise ValueError(f"Invalid compensation_type: {self.compensation_type}.")
        
        # Interpolate to fvec using cubic interpolation (PyTorch native)
        from .filters import torch_pchip_interp
        tf_outer_interp = torch_pchip_interp(f_outer, tf_outer_vals, self.fvec)
        tf_middle_interp = torch_pchip_interp(f_middle, tf_middle_vals, self.fvec)
        
        return tf_outer_interp, tf_middle_interp
    
    def _design_filter(self):
        r"""
        Design FIR filter from transfer function using frequency sampling.
        
        Converts the dB transfer function to linear amplitude, normalizes frequencies
        to Nyquist, and uses torch_firwin2 (PyTorch implementation) to create the FIR 
        filter in a differentiable way when learnable=True.
        
        Notes
        -----
        **Process:**
        1. Convert dB to linear: :math:`A(f) = 10^{H(f)/10}`
        2. Normalize frequencies to [0, 1]: :math:`f_{\text{norm}} = f / (f_s/2)`
        3. Handle Nyquist clipping (remove duplicates at 1.0)
        4. Add DC (0 Hz) and Nyquist (fs/2) boundary points
        5. Design FIR: ``torch_firwin2(order+1, freq_norm, gains)`` (differentiable)
        
        **Nyquist handling:**
        If ``fhigh >= fs/2``, multiple frequencies map to 1.0 and are deduplicated.
        Only the first occurrence is kept to avoid errors.
        """       
        # Convert dB to linear amplitude
        tf_linear = 10.0 ** (self.tf_gains / 10.0)
        
        # Normalize frequency vector to [0, 1] for firwin2
        nyquist = self.fs / 2.0
        
        # Normalize fvec and clip to [0, 1] (some freqs may exceed Nyquist for low fs)
        fvec_normalized = (self.fvec / nyquist).clamp(0.0, 1.0)
        
        # Remove duplicate values at 1.0 (if fhigh >= Nyquist, multiple freqs map to 1.0)
        # Keep only up to first occurrence of 1.0
        mask = fvec_normalized < 1.0
        if mask.any():
            fvec_normalized = fvec_normalized[mask]
            tf_linear_used = tf_linear[mask]
        else:
            # All frequencies exceed Nyquist, use only first one
            fvec_normalized = fvec_normalized[:1]
            tf_linear_used = tf_linear[:1]
        
        # Build frequency and gain arrays: DC, fvec (clipped), Nyquist
        freq_normalized = torch.cat([
            torch.tensor([0.0], dtype=self.dtype, device=self.fvec.device),  # DC
            fvec_normalized,                         # Clipped and deduplicated freqs
            torch.tensor([1.0], dtype=self.dtype, device=self.fvec.device)   # Nyquist
        ])
        
        gains_extended = torch.cat([
            tf_linear[:1],       # Use first gain for DC
            tf_linear_used,      # Gains for used frequencies
            tf_linear[-1:]       # Use last gain for Nyquist
        ])
        
        # Design FIR filter using PyTorch (differentiable!)
        fir_coeffs = torch_firwin2(self.order + 1, freq_normalized, gains_extended)
        
        # Convert to correct dtype
        fir_coeffs = fir_coeffs.to(dtype=self.dtype)
        
        # Store as buffer
        self.register_buffer('fir_coeffs', fir_coeffs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply outer and middle ear filtering.
        
        Uses zero-phase filtering (equivalent to MATLAB's ``filtfilt``) by applying
        the FIR filter in forward and backward passes. Handles 1D and 2D inputs
        automatically. Refreshes filter if learnable parameters have changed.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape:
            
            - ``(T,)``: Single-channel time-domain signal
            - ``(B, T)``: Batch of time-domain signals
            
            where B = batch size, T = time samples.
            
        Returns
        -------
        torch.Tensor
            Zero-phase filtered signal with same shape as input.
            
        Notes
        -----
        **Zero-phase filtering:**\n        Applies filtfilt algorithm (forward + reverse + filter + reverse) to achieve\n        zero phase distortion. The effective frequency response is :math:`|H(\\omega)|^2`.\n        \n        **Learnable filter update:**\n        If ``learnable=True`` and training mode, the filter is redesigned each forward\n        pass to reflect updated transfer function gains.\n        \n        **Device handling:**\n        Filter coefficients are automatically moved to match input device.\n        """
        # Handle 1D input
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # If learnable, recompute filter
        if self.learnable and self.training:
            self._design_filter()
        
        # Apply zero-phase filtering (filtfilt equivalent)
        # Forward pass
        x_fwd = self._fir_filter(x, self.fir_coeffs)
        
        # Backward pass (flip, filter, flip back)
        x_rev = torch.flip(x_fwd, dims=[-1])
        x_bwd = self._fir_filter(x_rev, self.fir_coeffs)
        y = torch.flip(x_bwd, dims=[-1])
        
        if squeeze_output:
            y = y.squeeze(0)
        
        return y
    
    def _fir_filter(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply FIR filter using convolution.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape: [batch, samples].
        coeffs : torch.Tensor
            Filter coefficients. Shape: [num_taps].
            
        Returns
        -------
        torch.Tensor
            Filtered signal. Shape: [batch, samples].
        """
        # Reshape for conv1d: [batch, 1, samples]
        x_3d = x.unsqueeze(1)
        
        # Reshape coefficients: [1, 1, num_taps]
        device = x.device
        dtype = x.dtype
        coeffs_3d = coeffs.to(device=device, dtype=dtype).flip(0).unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding to maintain length
        padding = len(coeffs) - 1
        y_3d = F.conv1d(x_3d, coeffs_3d, padding=padding)
        
        # Remove padding to match input length
        # conv1d with padding=len(coeffs)-1 adds padding//2 on each side
        # We need to trim to original length
        y = y_3d.squeeze(1)
        
        # Center the output (remove initial and final transients)
        trim = padding // 2
        y = y[..., trim:trim + x.shape[-1]]
        
        return y
    
    def get_frequency_response(self, nfft: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frequency response of the filter.
        
        Parameters
        ----------
        nfft : int, optional
            FFT length for frequency response. Default: 8192.
            
        Returns
        -------
        freqs : torch.Tensor
            Frequency vector in Hz. Shape: [nfft//2 + 1].
        response : torch.Tensor
            Frequency response in dB. Shape: [nfft//2 + 1].
        """
        # FFT of filter coefficients
        fft_result = torch.fft.rfft(self.fir_coeffs, n=nfft)
        
        # Magnitude in dB
        magnitude = torch.abs(fft_result)
        magnitude_db = 20 * torch.log10(magnitude + 1e-12)
        
        # Frequency vector
        freqs = torch.linspace(0, self.fs / 2, nfft // 2 + 1, dtype=self.dtype)
        
        return freqs, magnitude_db
    
    def get_transfer_function(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the original transfer function (before FIR design).
        
        Returns
        -------
        freqs : torch.Tensor
            Frequency vector in Hz. Shape: [num_freqs].
        gains : torch.Tensor
            Transfer function gains in dB. Shape: [num_freqs].
        """
        return self.fvec, self.tf_gains
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return (f"fs={self.fs}, type='{self.compensation_type}', field='{self.field_type}', "
                f"order={self.order}, learnable={self.learnable}")

# ----------------------------------------------- Middle Ear -------------------------------------------------

class MiddleEarFilter(nn.Module):
    r"""
    Middle ear filter for auditory models.
    
    Implements FIR filters approximating the frequency-dependent transmission 
    characteristics of the human middle ear, modeling the mechanical transfer 
    from the tympanic membrane through the ossicular chain to the cochlear oval 
    window. The filter captures the impedance matching function that efficiently 
    transmits sound energy from air to the fluid-filled cochlea.
    
    Two empirically-derived filter variants are available based on stapes velocity 
    measurements, representing the dominant approach in computational auditory models.
    
    Algorithm Overview
    ------------------
    The filter design process:
    
    1. **Load frequency response data**:
       
       - **lopezpoveda2001**: Stapes velocity from Goode et al. (1994) measurements
       - **jepsen2008**: Stapes impedance (inverted to velocity)
    
    2. **Nyquist handling**: Clip data above ``fs/2`` or extrapolate with decay
    
    3. **FIR filter design** (frequency sampling):
       
       .. math::
           h[n] = \text{fir2}(\text{order}, f_{\text{norm}}, A)
       
       where :math:`f_{\text{norm}} = f / (f_s/2)` and :math:`A` are amplitudes
    
    4. **Phase transformation** (optional):
       
       - **Zero-phase**: :math:`A_{\text{used}} = \sqrt{A}` (for filtfilt)
       - **Minimum-phase**: Hilbert transform of log-magnitude
    
    5. **Scaling**:
       
       - **lopezpoveda2001**: :math:`h[n] = h[n] / (20 \times 10^{-6})` (SPL reference)
       - **jepsen2008**: :math:`h[n] = h[n] / \max(|H(\omega)|) \times 10^{-8} \times 10^{104/20}`
    
    6. **Gain normalization** (if enabled):
       
       .. math::
           G_{\text{norm}} = -20 \log_{10}(\max(|H(\omega)|))
       
       Applied to ensure 0 dB passband gain
    
    7. **Filtering**:
       
       - **Zero-phase**: Forward-backward pass (``filtfilt``)
       - **Minimum-phase**: Convolution with delay compensation
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    
    filter_type : {'lopezpoveda2001', 'jepsen2008'}, optional
        Middle ear filter variant:
        
        * ``'lopezpoveda2001'``: Based on Goode et al. (1994) stapes velocity 
          measurements at 0 dB SPL. Default for Osses et al. (2021) model. 
          Provides bandpass characteristic with peak around 800 Hz - 1 kHz.
        * ``'jepsen2008'``: Based on stapes impedance data (inverted to velocity). 
          Used in Jepsen et al. (2008) model. Similar frequency response with 
          slightly different scaling.
        
        Default: ``'lopezpoveda2001'``.
    
    order : int, optional
        FIR filter order (number of taps - 1). Higher orders provide better 
        frequency resolution but increase computational cost and latency. 
        Default: 512.
    
    phase_type : {'minimum', 'zero'}, optional
        Phase characteristic of the filter:
        
        * ``'minimum'``: Minimum phase filter (causal, frequency-dependent group delay)
        * ``'zero'``: Zero phase filter (non-causal, linear phase via filtfilt)
        
        Default: ``'minimum'``.
    
    normalize_gain : bool, optional
        If True, normalizes the filter to have 0 dB gain at the passband peak. 
        This ensures consistent signal levels across different filter types and 
        sampling rates. Default: ``True``.
    
    learnable : bool, optional
        If True, both the frequency response gains and the gain normalization 
        factor become trainable ``nn.Parameter`` objects. The frequency grid 
        remains fixed, but amplitudes can adapt during training. Default: ``False``.
    
    compensate_delay : bool, optional
        If True, compensates for group delay by removing ``order//2`` samples. 
        Only effective for minimum-phase filters (zero-phase has no delay from 
        filtfilt). Default: ``True``.
    
    dtype : torch.dtype, optional
        Data type for filter coefficients and computations. Default: ``torch.float32``.
    
    Attributes
    ----------
    fir_coeffs : torch.Tensor
        FIR filter coefficients of shape ``[order+1]``.
    
    frequency_data : torch.Tensor or nn.Parameter
        Frequency response data of shape ``[N, 2]`` where column 0 contains 
        frequencies (Hz) and column 1 contains linear gains (velocity or 
        inverted impedance).
    
    gain_normalization : torch.Tensor or nn.Parameter
        Gain normalization factor in dB. Scalar value ensuring 0 dB passband gain.
    
    group_delay : int
        Group delay in samples (``order // 2``).
    
    Shape
    -----
    - Input: :math:`(B, T)` or :math:`(B, F, T)` where
        * :math:`B` = batch size
        * :math:`F` = frequency channels (optional)
        * :math:`T` = time samples
    - Output: Same shape as input (note: length reduced by ``group_delay`` if 
      ``compensate_delay=True`` and ``phase_type='minimum'``)
    
    Notes
    -----
    **Filter Characteristics:**
    
    - **lopezpoveda2001**: Peak gain around 800 Hz, -40 dB at 100 Hz, -20 dB at 10 kHz
    - **jepsen2008**: Similar bandpass shape with slightly different scaling
    - Both models capture the resonant behavior of the ossicular chain
    
    **Scaling Differences:**
    
    - **lopezpoveda2001**: Scaled to 0 dB SPL reference (20 µPa)
    - **jepsen2008**: Normalized by max FFT magnitude, then scaled by 1e-8 × 10^(104/20)
    
    **Gain Normalization:**
    
    When ``normalize_gain=True``, the filter is adjusted so the maximum gain 
    in the passband is exactly 0 dB. This:
    - Ensures consistent output levels across sampling rates
    - Facilitates comparison between filter variants
    - Prevents unexpected level changes in auditory model pipelines
    
    **Phase Options:**
    
    - **Minimum-phase**: Suitable for real-time processing, causal
    - **Zero-phase**: Suitable for offline analysis, preserves temporal symmetry
    
    **Learnable Parameters:**
    
    When ``learnable=True``, the filter can adapt to:
    - Individual middle ear transfer functions (pathologies, age-related changes)
    - Calibration for specific experimental setups
    - End-to-end optimization in neural auditory models
    
    **Device Handling:**
    
    This module is device-agnostic and works on CPU, CUDA, and MPS.
    
    See Also
    --------
    HeadphoneFilter : Headphone and outer ear response
    OuterMiddleEarFilter : Combined outer and middle ear (ANSI S3.4-2007)
    
    References
    ----------
    .. [1] Lopez-Poveda, E. A., & Meddis, R. (2001). A human nonlinear cochlear 
           filterbank. *The Journal of the Acoustical Society of America*, 
           110(6), 3107-3118.

    .. [2] Goode, R. L., Killion, M., Nakamura, K., & Nishihara, S. (1994). 
           New knowledge about the function of the human middle ear: Development 
           of an improved analog model. *American Journal of Otology*, 15(2), 145-154.

    .. [3] Jepsen, M. L., Ewert, S. D., & Dau, T. (2008). A computational model 
           of human auditory signal processing and perception. 
           *The Journal of the Acoustical Society of America*, 124(1), 422-438.

    .. [4] Osses, A., Decorsière, R., & Dau, T. (2021). A computational model for 
           the peripheral auditory system. *Acta Acustica*, 5, 56.
    
    Examples
    --------
    Basic usage with lopezpoveda2001 filter:
    
    >>> import torch
    >>> from torch_amt.common.ears import MiddleEarFilter
    >>> 
    >>> # Create filter for 16 kHz audio
    >>> mef = MiddleEarFilter(fs=16000, filter_type='lopezpoveda2001')
    >>> print(mef)
    MiddleEarFilter(fs=16000, filter_type=lopezpoveda2001, order=512, ...)
    >>> 
    >>> # Filter a stereo signal
    >>> signal = torch.randn(2, 16000)
    >>> filtered = mef(signal)
    >>> print(f"Input: {signal.shape} -> Output: {filtered.shape}")
    Input: torch.Size([2, 16000]) -> Output: torch.Size([2, 15744])
    >>> # Note: Output shorter by group_delay (512//2 = 256 samples)
    
    Multi-channel audio (e.g., from gammatone filterbank):
    
    >>> # Shape: (batch=4, channels=31, time=1600)
    >>> multichannel = torch.randn(4, 31, 1600)
    >>> filtered_multi = mef(multichannel)
    >>> print(filtered_multi.shape)
    torch.Size([4, 31, 1344])
    
    Compare lopezpoveda2001 vs jepsen2008:
    
    >>> mef_lp = MiddleEarFilter(fs=16000, filter_type='lopezpoveda2001')
    >>> mef_jp = MiddleEarFilter(fs=16000, filter_type='jepsen2008')
    >>> 
    >>> # Get frequency responses
    >>> freqs_lp, H_lp = mef_lp.get_frequency_response(nfft=8192)
    >>> freqs_jp, H_jp = mef_jp.get_frequency_response(nfft=8192)
    >>> 
    >>> # Both have peak around 800 Hz
    >>> mag_lp = 20 * torch.log10(torch.abs(H_lp) + 1e-10)
    >>> mag_jp = 20 * torch.log10(torch.abs(H_jp) + 1e-10)
    >>> print(f"LP peak: {mag_lp.max():.2f} dB at {freqs_lp[mag_lp.argmax()]:.1f} Hz")
    LP peak: 0.00 dB at 800.8 Hz
    >>> print(f"JP peak: {mag_jp.max():.2f} dB at {freqs_jp[mag_jp.argmax()]:.1f} Hz")
    JP peak: 0.00 dB at 800.8 Hz
    
    Zero-phase filtering for offline analysis:
    
    >>> mef_zero = MiddleEarFilter(fs=16000, phase_type='zero')
    >>> # Non-causal, no delay compensation needed
    >>> filtered_zero = mef_zero(signal)
    >>> print(filtered_zero.shape)
    torch.Size([2, 16000])
    
    Learnable filter for adaptive processing:
    
    >>> mef_learn = MiddleEarFilter(fs=16000, learnable=True, normalize_gain=True)
    >>> print(f"Learnable parameters: {sum(p.numel() for p in mef_learn.parameters())}")
    Learnable parameters: 53
    >>> # frequency_data: 26x2=52 params, gain_normalization: 1 param
    >>> 
    >>> # Use in training loop
    >>> optimizer = torch.optim.Adam(mef_learn.parameters(), lr=1e-4)
    >>> # Filter adapts during backpropagation
    
    Disable gain normalization:
    
    >>> mef_raw = MiddleEarFilter(fs=16000, normalize_gain=False)
    >>> # Filter preserves original scaling from data
    >>> freqs, H = mef_raw.get_frequency_response()
    >>> mag_db = 20 * torch.log10(torch.abs(H) + 1e-10)
    >>> print(f"Peak gain: {mag_db.max():.2f} dB")
    Peak gain: 65.16 dB
    """
    
    def __init__(self,
                 fs: float,
                 filter_type: FilterType = 'lopezpoveda2001',
                 order: int = 512,
                 phase_type: str = 'minimum',
                 normalize_gain: bool = True,
                 learnable: bool = False,
                 compensate_delay: bool = True,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        if filter_type not in ['lopezpoveda2001', 'jepsen2008']:
            raise ValueError(f"filter_type must be 'lopezpoveda2001' or 'jepsen2008', got {filter_type}")
        
        if phase_type not in ['minimum', 'zero']:
            raise ValueError(f"phase_type must be 'minimum' or 'zero', got {phase_type}")
        
        self.fs = float(fs)
        self.filter_type = filter_type
        self.order = order
        self.phase_type = phase_type
        self.normalize_gain = normalize_gain
        self.learnable = learnable
        self.compensate_delay = compensate_delay
        self.dtype = dtype
        self.group_delay = order // 2
        
        # Load appropriate frequency response data
        if filter_type == 'lopezpoveda2001':
            freq_data = LOPEZPOVEDA2001_DATA.clone().to(dtype)
        else:  # jepsen2008
            freq_data = JEPSEN2008_DATA.clone().to(dtype)
            # Invert impedance to get velocity
            freq_data[:, 1] = 1.0 / freq_data[:, 1]
        
        # Clip to Nyquist and extrapolate if needed
        freq_data = self._prepare_frequency_data(freq_data)
        
        # Store as parameter if learnable, otherwise as buffer
        if learnable:
            self.frequency_data = nn.Parameter(freq_data)
        else:
            self.register_buffer('frequency_data', freq_data)
        
        # Initialize gain normalization factor
        if learnable and normalize_gain:
            self.gain_normalization = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        else:
            self.register_buffer('gain_normalization', torch.tensor(1.0, dtype=dtype))
        
        # Design FIR filter
        self._design_filter()
    
    def _prepare_frequency_data(self, freq_data: torch.Tensor) -> torch.Tensor:
        """
        Prepare frequency data by clipping to Nyquist and extrapolating if needed.
        
        Handles frequency response data that may extend beyond the Nyquist frequency
        or need extrapolation to reach it. Ensures the last data point is exactly
        at ``fs/2`` for proper FIR filter design.
        
        Parameters
        ----------
        freq_data : torch.Tensor
            Raw frequency response data of shape ``[N, 2]`` where:
            
            - Column 0: Frequencies in Hz
            - Column 1: Linear amplitude gains
            
        Returns
        -------
        torch.Tensor
            Prepared frequency data of shape ``[M, 2]`` where:
            
            - All frequencies are ≤ ``fs/2``
            - Last frequency is exactly ``fs/2``
            - Extrapolated points added if needed
            
        Notes
        -----
        **For low sampling rates** (``fs ≤ 20 kHz``):
        Simply clips frequencies above Nyquist.
        
        **For high sampling rates** (``fs > 20 kHz``):
        Extrapolates from last data point to Nyquist using:
        
        - Frequency step: 1000 Hz
        - Amplitude decay: 1.1× per step
        - Final decay adjustment to reach exactly Nyquist
        
        This prevents abrupt transitions in the frequency response at high frequencies.
        """
        nyquist = self.fs / 2
        
        if self.fs <= 20000:
            # Clip data above Nyquist
            valid_indices = freq_data[:, 0] < nyquist
            freq_data = freq_data[valid_indices]
        else:
            # Extrapolate towards fs/2 with decay
            last_freq = freq_data[-1, 0].item()
            if last_freq < nyquist:
                # Add points every 1000 Hz with 1.1x decay
                extra_freqs = []
                extra_amps = []
                current_freq = last_freq
                current_amp = freq_data[-1, 1].item()
                
                while current_freq < nyquist:
                    current_freq += 1000
                    current_amp /= 1.1
                    if current_freq <= nyquist:
                        extra_freqs.append(current_freq)
                        extra_amps.append(current_amp)
                
                if extra_freqs:
                    extra_data = torch.tensor(
                        [[f, a] for f, a in zip(extra_freqs, extra_amps)],
                        dtype=freq_data.dtype
                    )
                    freq_data = torch.cat([freq_data, extra_data], dim=0)
        
        # Ensure last point is exactly at Nyquist
        if freq_data[-1, 0] != nyquist:
            last_amp = freq_data[-1, 1]
            freq_diff = nyquist - freq_data[-1, 0].item()
            decay = 1 + freq_diff * 0.1 / 1000
            nyquist_point = torch.tensor(
                [[nyquist, last_amp / decay]],
                dtype=freq_data.dtype
            )
            freq_data = torch.cat([freq_data, nyquist_point], dim=0)
        
        return freq_data
    
    def _design_filter(self, save_as_buffer=True):
        """
        Design the FIR filter from frequency response data.
        
        Parameters
        ----------
        save_as_buffer : bool
            If True, saves result as buffer (breaks gradient).
            If False, returns coefficients (preserves gradient).
        """
        # Use torch operations for differentiability (instead of scipy+numpy)
        freq_data = self.frequency_data  # Keep as tensor
        device = freq_data.device
        dtype = freq_data.dtype
        
        # Prepare frequency and amplitude vectors for fir2
        # Format: [0, f1, f2, ..., fN] where fN is already at fs/2 from _prepare_frequency_data
        # Check if last frequency is already at Nyquist
        if freq_data[-1, 0].item() == self.fs / 2:
            frequencies = torch.cat([
                torch.zeros(1, device=device, dtype=dtype),
                freq_data[:, 0]
            ])
            amplitudes = torch.cat([
                torch.zeros(1, device=device, dtype=dtype),
                freq_data[:, 1]
            ])
        else:
            frequencies = torch.cat([
                torch.zeros(1, device=device, dtype=dtype),
                freq_data[:, 0],
                torch.tensor([self.fs / 2], device=device, dtype=dtype)
            ])
            amplitudes = torch.cat([
                torch.zeros(1, device=device, dtype=dtype),
                freq_data[:, 1],
                torch.zeros(1, device=device, dtype=dtype)
            ])
        
        # If zero-phase (filtfilt), use sqrt of amplitudes to compensate for squaring
        if self.phase_type == 'zero':
            amplitudes = torch.sqrt(amplitudes)
        
        # Design FIR filter using torch_firwin2 (differentiable!)
        fir_coeffs = torch_firwin2(self.order + 1, frequencies, amplitudes, fs=self.fs)
        
        # Apply filter-specific scaling
        if self.filter_type == 'lopezpoveda2001':
            # Scale for SPL in dB re 20 µPa
            fir_coeffs = fir_coeffs / 20e-6
        elif self.filter_type == 'jepsen2008':
            # Normalize by max FFT magnitude and apply scaling
            # See Lopez (2001) figure text for figure 1
            H = torch.fft.fft(fir_coeffs, n=8192)
            max_mag = torch.abs(H).max()
            fir_coeffs = fir_coeffs / max_mag * 1e-8 * 10**(104/20)
        
        # Apply minimum phase transformation if requested
        if self.phase_type == 'minimum':
            # Use PyTorch native minimum phase transform (differentiable!)
            fir_coeffs = torch_minimum_phase(fir_coeffs)
        
        # Compute gain normalization if enabled
        if self.normalize_gain:
            if self.phase_type == 'zero':
                # For zero-phase, compute normalization based on actual filtfilt output
                # Create impulse and pass through filtfilt simulation
                # Use torch operations where possible
                impulse_test = torch.zeros(8192, device=device, dtype=dtype)
                impulse_test[0] = 1.0
                # For gain calculation, use scipy filtfilt on cpu (non-differentiable part)
                from scipy.signal import filtfilt as scipy_filtfilt
                impulse_filtered = scipy_filtfilt(fir_coeffs.detach().cpu().numpy(), [1.0], impulse_test.cpu().numpy())
                max_gain_linear = np.max(np.abs(impulse_filtered))
                max_gain_db = 20 * np.log10(max_gain_linear + 1e-10)
            else:
                # For minimum-phase, use torch FFT
                H = torch.fft.fft(fir_coeffs, n=8192)
                max_gain_db = 20 * torch.log10(torch.abs(H).max() + 1e-10).item()
            
            if not isinstance(self.gain_normalization, nn.Parameter):
                self.gain_normalization.data.fill_(-max_gain_db)
        
        if save_as_buffer:
            # Store as buffer (breaks gradient, used for non-learnable mode)
            self.register_buffer('fir_coeffs', fir_coeffs)
            return None
        else:
            # Return coefficients (preserves gradient, used for learnable mode)
            return fir_coeffs
    
    def _minimum_phase(self, h: np.ndarray) -> np.ndarray:
        r"""
        Convert linear-phase FIR filter to minimum-phase using Hilbert transform.
        
        Minimum-phase filters minimize group delay for a given magnitude response,
        making them optimal for causal, real-time applications.
        
        Parameters
        ----------
        h : np.ndarray
            Linear-phase FIR coefficients of shape ``[order+1]``.
        
        Returns
        -------
        np.ndarray
            Minimum-phase FIR coefficients of shape ``[order+1]``.
            
        Notes
        -----
        **Algorithm:**
        
        1. Compute FFT: :math:`H(\omega) = \text{FFT}(h)`
        2. Log-magnitude: :math:`L(\omega) = \log|H(\omega)|` (with 1e-10 regularization)
        3. Hilbert transform for minimum phase: :math:`\phi_{\min}(\omega) = -\mathcal{H}\{L(\omega)\}`
        4. Reconstruct: :math:`H_{\min}(\omega) = |H(\omega)| e^{j\phi_{\min}(\omega)}`
        5. IFFT: :math:`h_{\min} = \text{IFFT}(H_{\min})`
        
        Regularization prevents ``log(0)`` for near-zero frequency components.
        """
        # Compute FFT of impulse response
        H = np.fft.fft(h)
        
        # Compute minimum phase via Hilbert transform of log magnitude
        log_mag = np.log(np.abs(H) + 1e-10)
        min_phase = -np.imag(scipy_signal.hilbert(log_mag))
        
        # Reconstruct minimum phase spectrum
        H_min = np.abs(H) * np.exp(1j * min_phase)
        
        # Convert back to time domain
        h_min = np.real(np.fft.ifft(H_min))
        
        return h_min
    
    def _torch_filtfilt(self, x: torch.Tensor, fir_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply zero-phase filtering using PyTorch native implementation.
        
        Uses forward-backward filtering for zero-phase response while
        maintaining gradient flow for end-to-end training.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(B, C, T)`` where:
            
            - B = batch size
            - C = number of channels
            - T = time samples
        fir_coeffs : torch.Tensor
            FIR filter coefficients, shape ``(order+1,)``.
            
        Returns
        -------
        torch.Tensor
            Zero-phase filtered signal of shape ``(B, C, T)``.
            
        Notes
        -----
        **Algorithm:**
        
        1. Pad signal with reflection at boundaries
        2. Forward filtering using convolution
        3. Reverse signal
        4. Forward filtering again
        5. Reverse back and remove padding
        
        This achieves zero-phase response by canceling the phase delay
        in forward and backward passes. Fully differentiable.
        """
        from .filters import torch_filtfilt
        
        return torch_filtfilt(fir_coeffs, x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply middle ear filter to input signal.
        
        Filters the input through the middle ear frequency response. Automatically
        handles 2D (batch, time) and 3D (batch, channels, time) inputs. Applies
        gain normalization and delay compensation if configured. Refreshes filter
        coefficients if learnable parameters have changed.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape:
            
            - ``(B, T)``: Batch of time-domain signals
            - ``(B, F, T)``: Batch of multi-channel signals (e.g., from filterbank)
            
            where B = batch size, F = frequency channels, T = time samples.
        
        Returns
        -------
        torch.Tensor
            Filtered signal. Shape depends on phase type and delay compensation:
            
            - **Zero-phase**: Same shape as input (no delay)
            - **Minimum-phase with compensation**: Shorter by ``group_delay`` samples
            - **Minimum-phase without compensation**: Same shape as input
            
        Notes
        -----
        **Processing order:**
        
        1. Refresh filter if learnable parameters changed
        2. Reshape input for processing
        3. Apply filtering (filtfilt or convolution)
        4. Apply gain normalization (if enabled)
        5. Compensate delay (if minimum-phase and enabled)
        6. Restore original shape
        
        **Gain normalization:**
        
        If ``normalize_gain=True``, applies linear gain to achieve 0 dB passband peak:
        
        .. math::
            y(t) = y(t) \times 10^{G_{\text{norm}}/20}
        
        **Device handling:**
        
        Filter coefficients are automatically moved to match input device (CPU/CUDA/MPS).
        """
        # Compute filter coefficients (preserving gradient if learnable)
        if isinstance(self.frequency_data, nn.Parameter) and self.training:
            # Learnable mode: recompute filter each forward to preserve gradient
            fir_coeffs = self._design_filter(save_as_buffer=False)
        else:
            # Non-learnable mode: use cached buffer
            # Refresh filter if gains have changed
            if isinstance(self.frequency_data, nn.Parameter):
                if not hasattr(self, '_last_freq_data'):
                    self._design_filter(save_as_buffer=True)
                    self._last_freq_data = self.frequency_data.detach().clone()
                elif not torch.equal(self._last_freq_data, self.frequency_data):
                    self._design_filter(save_as_buffer=True)
                    self._last_freq_data = self.frequency_data.detach().clone()
            fir_coeffs = self.fir_coeffs
        
        original_shape = x.shape
        
        # Handle different input shapes
        if x.ndim == 1:
            # (T,) -> (1, 1, T)
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif x.ndim == 2:
            # (B, T) -> (B, 1, T)
            x = x.unsqueeze(1)
            squeeze_output = False
        elif x.ndim == 3:
            # (B, C, T) - already correct
            squeeze_output = False
        else:
            raise ValueError(f"Input must be 1D, 2D, or 3D, got shape {original_shape}")
        
        # Use filtfilt for zero-phase or regular convolution for minimum-phase
        if self.phase_type == 'zero':
            # Zero-phase filtering (non-causal)
            y = self._torch_filtfilt(x, fir_coeffs)
        else:
            # Minimum-phase filtering (causal)
            # Apply FIR filtering via convolution
            device = x.device
            dtype = x.dtype
            kernel = fir_coeffs.to(device=device, dtype=dtype).flip(0).view(1, 1, -1)
            
            B, num_channels, T = x.shape
            x_flat = x.reshape(B * num_channels, 1, T)
            y_flat = F.conv1d(x_flat, kernel, padding=self.order // 2)
            y = y_flat.reshape(B, num_channels, -1)
        
        # Apply gain normalization if enabled
        if self.normalize_gain:
            gain_linear = 10 ** (self.gain_normalization / 20)
            y = y * gain_linear
        
        # Remove delay compensation ONLY for minimum-phase (zero-phase has no delay from filtfilt)
        if self.compensate_delay and self.phase_type == 'minimum' and y.shape[-1] > self.group_delay:
            y = y[:, :, self.group_delay:]
        
        # Restore original shape
        if squeeze_output:
            y = y.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            y = y.squeeze(1)
        
        return y
    
    def get_frequency_response(self, nfft: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frequency response of the filter.
        
        Returns the complex frequency response after applying gain normalization
        (if enabled). Useful for analyzing filter characteristics and debugging.
        
        Parameters
        ----------
        nfft : int, optional
            Number of FFT points for frequency resolution. Higher values provide
            finer frequency resolution. Default: 8192.
        
        Returns
        -------
        freqs : torch.Tensor
            Frequency vector in Hz of shape ``[nfft//2 + 1]``.
            
        response : torch.Tensor
            Complex frequency response of shape ``[nfft//2 + 1]``.
            Includes gain normalization if ``normalize_gain=True``.
            
        Notes
        -----
        To get magnitude in dB:
        
        .. code-block:: python
        
            freqs, H = mef.get_frequency_response()
            magnitude_db = 20 * torch.log10(torch.abs(H) + 1e-10)
        
        With ``normalize_gain=True``, the peak magnitude should be exactly 0 dB.
        """
        # Compute FFT of filter coefficients
        h_padded = F.pad(self.fir_coeffs, (0, nfft - len(self.fir_coeffs)))
        H = torch.fft.rfft(h_padded, n=nfft)
        
        # Apply gain normalization if enabled
        if self.normalize_gain:
            gain_linear = 10 ** (self.gain_normalization.to(device=H.device) / 20)
            H = H * gain_linear
        
        freqs = torch.linspace(0, self.fs / 2, nfft // 2 + 1, device=H.device)
        
        return freqs, H
    
    def extra_repr(self) -> str:
        """
        Extra representation string for module printing.
        
        Returns
        -------
        str
            String containing key module parameters.
        """
        return (f'fs={self.fs}, filter_type={self.filter_type}, order={self.order}, '
                f'phase_type={self.phase_type}, normalize_gain={self.normalize_gain}, '
                f'learnable={self.learnable}')
