torch_amt - PyTorch Auditory Modeling Toolbox
==============================================

**Differentiable, Hardware-accelerated PyTorch implementations of Computational Auditory models from the MATLAB Auditory Modeling Toolbox (AMT).**

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/badge/python-3.14+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.14+

.. image:: https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch 2.0+

.. figure:: ../../dev/AMT_front_image.png
   :alt: torch_amt - PyTorch Auditory Modeling Toolbox
   :align: center
   :width: 800px

Overview
--------

torch_amt provides a comprehensive collection of differentiable auditory models and building blocks
for psychoacoustic research, computational neuroscience, and audio deep learning applications.

**Key Features:**

* 🔥 **Hardware acceleration** - CUDA, MPS (Apple Silicon), and CPU support
* 📊 **Fully differentiable** - Integrate with neural networks and optimize via backpropagation
* 🧩 **Modular architecture** - Mix and match components for custom auditory pipelines
* 🎓 **Scientific adherence** - Matching MATLAB AMT v1.6.0 implementations
* 📚 **Comprehensive documentation** - Detailed API reference with equations and examples

Installation
------------

.. code-block:: bash

   pip install torch-amt

Or from source:

.. code-block:: bash

   git clone https://github.com/StefanoGiacomelli/torch_amt.git
   cd torch_amt
   pip install -e .

Quick Start
-----------

Complete Auditory Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch_amt

    # Load Dau et al. (1997) model
    model = torch_amt.Dau1997(fs=48000)

    # Process 1 second of audio
    audio = torch.randn(1, 48000)  # (batch, time)
    output = model(audio)

    print(f"Input: {audio.shape}")
    # Input: torch.Size([1, 48000])
    print(f"Output: List of {len(output)} frequency channels")
    # Output: List of 31 frequency channels
    print(f"Each channel shape: {output[0].shape}")
    # Each channel shape: torch.Size([1, 8, 48000]) - (batch, modulation_channels, time)

Custom Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch_amt

    # Build custom auditory processing chain
    filterbank = torch_amt.GammatoneFilterbank(fs=48000, fc=(80, 8000))
    ihc = torch_amt.IHCEnvelope(fs=48000)
    adaptation = torch_amt.AdaptLoop(fs=48000)

    # Process signal
    audio = torch.randn(2, 48000)     # Batch of 2 signals
    filtered = filterbank(audio)      # (2, 31, 48000) - 31 frequency channels
    envelope = ihc(filtered)          # (2, 31, 48000) - Envelope extraction
    adapted = adaptation(envelope)    # (2, 31, 48000) - Temporal adaptation

    print(f"Input: {audio.shape}")
    # Input: torch.Size([2, 48000])
    print(f"After Gammatone filterbank: {filtered.shape}")
    # After Gammatone filterbank: torch.Size([2, 31, 48000])
    print(f"After IHC envelope: {envelope.shape}")
    # After IHC envelope: torch.Size([2, 31, 48000])
    print(f"After adaptation: {adapted.shape}")
    # After adaptation: torch.Size([2, 31, 48000])

Hardware Acceleration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch_amt

    # Check available hardware
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Move model to GPU (CUDA or MPS)
    model = torch_amt.Dau1997(fs=48000)

    if torch.backends.mps.is_available():
        model = model.to('mps')  # Apple Silicon
        print(f"Using device: mps")
    elif torch.cuda.is_available():
        model = model.cuda()  # NVIDIA GPU
        print(f"Using device: cuda")
    else:
        print(f"Using device: cpu")

    # Process on accelerated hardware
    audio = torch.randn(8, 48000).to(model.gammatone_fb.fc.device)
    output = model(audio)

Learnable Front-ends for Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch_amt

    class AudioClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # Learnable auditory front-end
            self.auditory = torch_amt.King2019(fs=48000, learnable=True)
            self.classifier = nn.Linear(155, 10)  # 31 freqs × 5 mods = 155 → 10 classes
        
        def forward(self, audio):
            features = self.auditory(audio)     # (B, T, F, M) e.g., (4, 24000, 31, 5)
            pooled = features.mean(dim=1)       # (B, F, M) e.g., (4, 31, 5) - Pool over time
            flattened = pooled.flatten(1)       # (B, F×M) e.g., (4, 155)
            return self.classifier(flattened)   # (B, 10)

    # Train end-to-end with backpropagation
    model = AudioClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Example forward pass
    audio = torch.randn(4, 24000)  # Batch of 4 signals, 0.5 seconds @ 48kHz
    logits = model(audio)  # (4, 10)
    print(f"Input: {audio.shape} → Output: {logits.shape}")
    # Input: torch.Size([4, 24000]) → Output: torch.Size([4, 10])

Available Models
----------------

torch_amt includes 6 complete auditory models:

* **Dau1997** - Temporal processing model with adaptation loops
* **Glasberg2002** - Loudness model with specific loudness transformation
* **Moore2016** - Binaural loudness model with spatial processing
* **King2019** - FM/AM masking model with broken-stick compression
* **Osses2021** - Temporal integration model
* **Paulick2024** - Physiological CASP model with advanced IHC

Plus 43+ building block components organized into:

* **Ear Models** - Outer and middle ear filtering
* **Auditory Filterbanks** - Gammatone, DRNL, excitation patterns
* **Inner Hair Cell Models** - Envelope extraction, physiological models
* **Modulation Analysis** - Temporal modulation filterbanks (standard & fast)
* **Loudness Processing** - Compression, specific loudness, binaural processing
* **Signal Processing** - Filters, transforms, utilities

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/models
   api/filterbanks
   api/ihc
   api/adaptation
   api/modulation
   api/loudness
   api/ears
   api/filters

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   license
   citing

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use torch_amt in your research, please cite:

.. code-block:: bibtex

   @software{giacomelli2026torch_amt,
     author = {Giacomelli, Stefano},
     title = {torch\_amt: PyTorch Auditory Modeling Toolbox},
     year = {2026},
     url = {https://github.com/StefanoGiacomelli/torch_amt},
     version = {0.1.0}
   }

Also consider citing the original AMT paper:

.. code-block:: bibtex

   @article{majdak2022amt,
     author = {Majdak, Piotr and Hollomey, Clara and Baumgartner, Robert},
     title = {AMT 1.x: A toolbox for reproducible research in auditory modeling},
     journal = {Acta Acustica},
     volume = {6},
     pages = {19},
     year = {2022},
     doi = {10.1051/aacus/2022011},
     url = {https://amtoolbox.org/}
   }

Contact
-------

**Stefano Giacomelli**  
ICT - Ph.D. Candidate  
Department of Engineering, Information Science & Mathematics (DISIM dpt.)
University of L'Aquila, Italy

.. figure:: https://phdict.disim.univaq.it/wp-content/uploads/2024/06/logo-univaq-disim-2-2-768x283.png
   :alt: DISIM - University of L'Aquila
   :align: left
   :width: 400px
   :height: 150px

📧 Email: stefano.giacomelli@graduate.univaq.it  
🔗 GitHub: https://github.com/StefanoGiacomelli 
🆔 ORCID: https://orcid.org/0009-0009-0438-1748 
🎓 Scholar: https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=it  
💼 LinkedIn: https://www.linkedin.com/in/stefano-giacomelli-811654135

*This project is funded under the Italian National Ministry of University and Research, for the Italian National Recovery and Resilience Plan (NRRP) "Methods of Computational Auditory Scene Analysis and Synthesis supporting eXtended and Immersive Reality Services"*

License
-------

This project is licensed under the GNU General Public License v3.0 or later (GPLv3+).
