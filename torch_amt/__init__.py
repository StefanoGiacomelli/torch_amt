"""
torch_amt: PyTorch Auditory Modeling Toolbox
=============================================

A comprehensive, differentiable PyTorch implementation of computational auditory 
models from the Auditory Modeling Toolbox (AMT). Provides hardware-accelerated, 
gradient-enabled implementations of state-of-the-art psychoacoustic models and 
their constituent building blocks.

**Key Features:**
    - Hardware-accelerated with PyTorch (CUDA, MPS, CPU)
    - Differentiable for gradient-based optimization
    - Modular architecture with reusable components
    - Allignment with AMT MATLAB/Octave implementations
    - Extensively documented with examples and references

**Quick Start:**

    >>> import torch
    >>> import torch_amt
    >>> 
    >>> # Load a complete auditory model
    >>> model = torch_amt.Dau1997(fs=48000)
    >>> audio = torch.randn(1, 48000)  # 1 second of audio at 48 kHz
    >>> output = model(audio)
    >>> 
    >>> # Or use individual components
    >>> fb = torch_amt.GammatoneFilterbank(fs=48000, fc=(80, 8000))
    >>> ihc = torch_amt.IHCEnvelope(fs=48000)
    >>> 
    >>> filtered = fb(audio)
    >>> envelope = ihc(filtered)

**Package Structure:**

    torch_amt/
    ├── models/             # Complete end-to-end auditory models
    │   ├── Dau1997                 - Monaural Temporal processing model
    │   ├── Glasberg2002            - Loudness model
    │   ├── Moore2016               - Binaural loudness model
    │   ├── King2019                - FM/AM masking model      
    │   ├── Osses2021               - Binaural Temporal integration model
    │   └── Paulick2024             - Physiological CASP model
    │
    └── common/             # Reusable building blocks
        ├── filterbanks.py          - Auditory filterbanks (Gammatone, DRNL, etc.)
        ├── ihc.py                  - Inner hair cell models
        ├── adaptation.py           - Auditory nerve adaptation
        ├── modulation.py           - Modulation filterbanks
        ├── loudness.py             - Loudness processing stages
        ├── ears.py                 - Outer/middle ear filtering
        └── filters.py              - Generic signal processing utilities

**Author:**
    Stefano Giacomelli - Ph.D. candidate @ DISIM dpt. - University of L'Aquila
    
**License:**
    GNU General Public License v3.0 or later (GPLv3+)

**Citations:**
    If you use this package in your research, please cite:
    
    - Majdak, P., Hollomey, C., & Baumgartner, R. (2022). "AMT 1.x: A toolbox 
      for reproducible research in auditory modeling." Acta Acustica, 6, 19.

**References:**
    - MATLAB Auditory Modeling Toolbox v1.6.0: http://amtoolbox.org/
    - Documentation: [To be added when published]
    - GitHub: https://github.com/StefanoGiacomelli/torch_amt
    - PyPI: [To be added when published]

**Version History:**
    - 0.1.0 (2026-02): Initial release
    - 0.2.0 (2026-03): Updated documentation and added new features
"""

# ============================================================================
# Package Metadata
# ============================================================================

__version__ = "0.2.0"
__author__ = "Stefano Giacomelli"
__email__ = "stefano.giacomelli@graduate.univaq.it"
__license__ = "GPL-3.0-or-later"
__url__ = "https://github.com/StefanoGiacomelli/torch_amt" 
__description__ = "PyTorch Auditory Modeling Toolbox - Differentiable implementations of computational auditory models"

# ============================================================================
# Public API - End-to-End Auditory Models
# ============================================================================
# Complete psychoacoustic models ready for research and applications

from torch_amt.models.dau1997 import Dau1997
from torch_amt.models.glasberg2002 import Glasberg2002
from torch_amt.models.king2019 import King2019
from torch_amt.models.moore2016 import Moore2016
from torch_amt.models.osses2021 import Osses2021
from torch_amt.models.paulick2024 import Paulick2024

# ============================================================================
# Public API - Common Building Blocks
# ============================================================================
# Modular components for custom auditory processing pipelines

# --- Filterbanks & Frequency Processing ---
# Gammatone, ERB-scale, excitation patterns, DRNL
from torch_amt.common.filterbanks import (
    # Utility functions for ERB scale conversions
    audfiltbw,                          # Auditory filter bandwidth
    erb2fc,                             # ERB to center frequency
    fc2erb,                             # Center frequency to ERB
    f2erb,                              # Frequency to ERB bandwidth
    f2erbrate,                          # Frequency to ERB rate
    erbrate2f,                          # ERB rate to frequency
    erbspacebw,                         # ERB-spaced bandwidth
    
    # Filterbank implementations
    GammatoneFilterbank,                # Gammatone auditory filterbank
    ERBIntegration,                     # ERB integration across frequency
    MultiResolutionFFT,                 # Multi-resolution spectral analysis
    Moore2016Spectrum,                  # Moore et al. (2016) spectrum analysis
    ExcitationPattern,                  # Excitation pattern transformation
    Moore2016ExcitationPattern,         # Moore et al. (2016) excitation pattern
    DRNLFilterbank,                     # Dual resonance nonlinear filterbank
    FastDRNLFilterbank,                 # Optimized DRNL implementation
)

# --- Inner Hair Cell Processing ---
# Envelope extraction and physiological IHC models
from torch_amt.common.ihc import (
    IHCEnvelope,                        # Standard envelope extraction (Dau et al.)
    IHCPaulick2024,                     # Physiological IHC model (Paulick et al. 2024)
)

# --- Auditory Nerve Adaptation ---
# Temporal adaptation mechanisms
from torch_amt.common.adaptation import (
    AdaptLoop,                          # Multi-stage adaptation loops (Dau et al.)
)

# --- Modulation Analysis ---
# Temporal modulation filterbanks for AM/FM analysis
from torch_amt.common.modulation import (
    ModulationFilterbank,               # Standard modulation filterbank (Dau, Jepsen, Paulick)
    FastModulationFilterbank,           # Optimized mega-batch version (~5x faster)
    King2019ModulationFilterbank,       # King et al. (2019) modulation filterbank
    FastKing2019ModulationFilterbank,   # FFT-accelerated King2019 (~250x faster)
)

# --- Outer & Middle Ear Filtering ---
# Frequency-dependent transfer functions
from torch_amt.common.ears import (
    HeadphoneFilter,                    # Headphone frequency response
    MiddleEarFilter,                    # Middle ear transfer function
    OuterMiddleEarFilter,               # Combined outer+middle ear
)

# --- Loudness Processing ---
# Compression, specific loudness, binaural processing
from torch_amt.common.loudness import (
    # Compression stages
    BrokenStickCompression,             # Broken-stick nonlinearity (King et al.)
    PowerCompression,                   # Power-law compression
    
    # Specific loudness transformations
    SpecificLoudness,                   # Glasberg & Moore (2002) specific loudness
    Moore2016SpecificLoudness,          # Moore et al. (2016) specific loudness
    
    # Binaural processing
    SpatialSmoothing,                   # Gaussian smoothing across ERB scale
    BinauralInhibition,                 # Cross-ear inhibition
    Moore2016BinauralLoudness,          # Complete binaural loudness computation
    
    # Temporal integration
    LoudnessIntegration,                # Glasberg & Moore (2002) integration
    Moore2016AGC,                       # Automatic Gain Control
    Moore2016TemporalIntegration,       # Two-stage temporal integration
)

# --- Generic Filters & Signal Processing ---
# Low-level filtering utilities and transforms
from torch_amt.common.filters import (
    # Analysis
    torch_hilbert,                      # Analytic signal via Hilbert transform
    torch_pchip_interp,                 # PCHIP interpolation
    
    # Filter classes
    ButterworthFilter,                  # Butterworth IIR filter
    SOSFilter,                          # Second-order sections filter
    IIRFilter,                          # Generic IIR filter
    
    # Filter application functions
    apply_sos_pytorch,                  # Apply SOS filter with gradients
    apply_iir_pytorch,                  # Apply IIR filter with gradients
    
    # FIR design utilities
    torch_firwin2,                      # FIR filter design
    torch_minimum_phase,                # Convert to minimum phase
    torch_filtfilt,                     # Zero-phase filtering
)

# ============================================================================
# Package-Level Exports
# ============================================================================
# Defines what `from torch_amt import *` will expose

__all__ = [
    # ========================================================================
    # Complete Auditory Models
    # ========================================================================
    "Dau1997",              # Dau et al. (1997) - Temporal processing
    "Glasberg2002",         # Glasberg & Moore (2002) - Loudness
    "Moore2016",            # Moore et al. (2016) - Binaural loudness
    "King2019",             # King et al. (2019) - FM/AM masking
    "Osses2021",            # Osses et al. (2021) - Temporal integration
    "Paulick2024",          # Paulick et al. (2024) - Physiological CASP
    
    # ========================================================================
    # Filterbanks & Frequency Processing
    # ========================================================================
    # ERB scale utilities
    "audfiltbw",
    "erb2fc",
    "fc2erb",
    "f2erb",
    "f2erbrate",
    "erbrate2f",
    "erbspacebw",
    
    # Filterbank classes
    "GammatoneFilterbank",
    "ERBIntegration",
    "MultiResolutionFFT",
    "Moore2016Spectrum",
    "ExcitationPattern",
    "Moore2016ExcitationPattern",
    "DRNLFilterbank",
    "FastDRNLFilterbank",
    
    # ========================================================================
    # Inner Hair Cell Models
    # ========================================================================
    "IHCEnvelope",
    "IHCPaulick2024",
    
    # ========================================================================
    # Adaptation
    # ========================================================================
    "AdaptLoop",
    
    # ========================================================================
    # Modulation Analysis
    # ========================================================================
    "ModulationFilterbank",
    "FastModulationFilterbank",
    "King2019ModulationFilterbank",
    "FastKing2019ModulationFilterbank",
    
    # ========================================================================
    # Outer & Middle Ear Models
    # ========================================================================
    "HeadphoneFilter",
    "MiddleEarFilter",
    "OuterMiddleEarFilter",
    
    # ========================================================================
    # Loudness Processing
    # ========================================================================
    "BrokenStickCompression",
    "PowerCompression",
    "SpecificLoudness",
    "Moore2016SpecificLoudness",
    "SpatialSmoothing",
    "BinauralInhibition",
    "Moore2016BinauralLoudness",
    "LoudnessIntegration",
    "Moore2016AGC",
    "Moore2016TemporalIntegration",
    
    # ========================================================================
    # Signal Processing Utilities
    # ========================================================================
    "torch_hilbert",
    "torch_pchip_interp",
    "ButterworthFilter",
    "SOSFilter",
    "IIRFilter",
    "apply_sos_pytorch",
    "apply_iir_pytorch",
    "torch_firwin2",
    "torch_minimum_phase",
    "torch_filtfilt",
]

# ============================================================================
# Convenience: Group components by category for easier discovery
# ============================================================================

# Users can access grouped components via torch_amt.models or torch_amt.filterbanks
models = {
    'Dau1997': Dau1997,
    'Glasberg2002': Glasberg2002,
    'King2019': King2019,
    'Moore2016': Moore2016,
    'Osses2021': Osses2021,
    'Paulick2024': Paulick2024,
}

filterbanks = {
    'GammatoneFilterbank': GammatoneFilterbank,
    'DRNLFilterbank': DRNLFilterbank,
    'FastDRNLFilterbank': FastDRNLFilterbank,
    'ExcitationPattern': ExcitationPattern,
    'ERBIntegration': ERBIntegration,
    'Moore2016Spectrum': Moore2016Spectrum,
    'Moore2016ExcitationPattern': Moore2016ExcitationPattern,
}

ihc = {
    'IHCEnvelope': IHCEnvelope,
    'IHCPaulick2024': IHCPaulick2024,
}

adaptation = {
    'AdaptLoop': AdaptLoop,
}

ears = {
    'HeadphoneFilter': HeadphoneFilter,
    'MiddleEarFilter': MiddleEarFilter,
    'OuterMiddleEarFilter': OuterMiddleEarFilter,
}

modulation = {
    'ModulationFilterbank': ModulationFilterbank,
    'FastModulationFilterbank': FastModulationFilterbank,
    'King2019ModulationFilterbank': King2019ModulationFilterbank,
    'FastKing2019ModulationFilterbank': FastKing2019ModulationFilterbank,
}

loudness = {
    'BrokenStickCompression': BrokenStickCompression,
    'PowerCompression': PowerCompression,
    'SpecificLoudness': SpecificLoudness,
    'BinauralInhibition': BinauralInhibition,
    'LoudnessIntegration': LoudnessIntegration,
    'SpatialSmoothing': SpatialSmoothing,
    'Moore2016SpecificLoudness': Moore2016SpecificLoudness,
    'Moore2016BinauralLoudness': Moore2016BinauralLoudness,
    'Moore2016AGC': Moore2016AGC,
    'Moore2016TemporalIntegration': Moore2016TemporalIntegration,    
}
