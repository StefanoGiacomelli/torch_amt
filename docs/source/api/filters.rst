Signal Processing Utilities
============================

Low-level filtering utilities and signal transforms.

.. currentmodule:: torch_amt

Classes
-------

ButterworthFilter
~~~~~~~~~~~~~~~~~

.. autoclass:: ButterworthFilter
   :members:
   :inherited-members:
   :show-inheritance:

SOSFilter
~~~~~~~~~

.. autoclass:: SOSFilter
   :members:
   :inherited-members:
   :show-inheritance:

IIRFilter
~~~~~~~~~

.. autoclass:: IIRFilter
   :members:
   :inherited-members:
   :show-inheritance:

Utility Functions
-----------------

Analysis
~~~~~~~~

.. autofunction:: torch_hilbert
.. autofunction:: torch_pchip_interp

Filtering
~~~~~~~~~

.. autofunction:: apply_sos_pytorch
.. autofunction:: apply_iir_pytorch
.. autofunction:: torch_filtfilt

Design
~~~~~~

.. autofunction:: torch_firwin2
.. autofunction:: torch_minimum_phase
