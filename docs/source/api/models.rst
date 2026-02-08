Complete Auditory Models
========================

torch_amt provides 6 complete end-to-end auditory models ready for research and applications.

.. currentmodule:: torch_amt

Model Overview
--------------

+----------------+------+----------------------------------+-------------------------------------------+
| Model          | Year | Key Features                     | Primary Applications                      |
+================+======+==================================+===========================================+
| Dau1997        | 1997 | Adaptation loops, modulation     | AM detection, temporal processing         |
+----------------+------+----------------------------------+-------------------------------------------+
| Glasberg2002   | 2002 | Specific loudness, integration   | Loudness perception, hearing aids         |
+----------------+------+----------------------------------+-------------------------------------------+
| Moore2016      | 2016 | Binaural processing, spatial     | Binaural loudness, spatial hearing        |
+----------------+------+----------------------------------+-------------------------------------------+
| King2019       | 2019 | Broken-stick compression, FM/AM  | FM masking, modulation interactions       |
+----------------+------+----------------------------------+-------------------------------------------+
| Osses2021      | 2021 | Extended temporal integration    | Speech perception, temporal resolution    |
+----------------+------+----------------------------------+-------------------------------------------+
| Paulick2024    | 2024 | Physiological IHC, CASP          | Physiological modeling, cochlear implants |
+----------------+------+----------------------------------+-------------------------------------------+

Dau1997
-------

.. autoclass:: torch_amt.Dau1997
   :members:
   :inherited-members:
   :show-inheritance:

Glasberg2002
------------

.. autoclass:: torch_amt.Glasberg2002
   :members:
   :inherited-members:
   :show-inheritance:

Moore2016
---------

.. autoclass:: torch_amt.Moore2016
   :members:
   :inherited-members:
   :show-inheritance:

King2019
--------

.. autoclass:: torch_amt.King2019
   :members:
   :inherited-members:
   :show-inheritance:

Osses2021
---------

.. autoclass:: torch_amt.Osses2021
   :members:
   :inherited-members:
   :show-inheritance:

Paulick2024
-----------

.. autoclass:: torch_amt.Paulick2024
   :members:
   :inherited-members:
   :show-inheritance:
