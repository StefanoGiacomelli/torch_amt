# 📋 **TORCH_AMT DOCUMENTATION & CODE STRUCTURE STANDARDS (MODELS)**

## Header

```python
"""
[ModelName] Auditory Model [w. Distinctive Features]
====================================================

Author:
    Name Surname - Role @ Where

License:
    GNU General Public License v3.0 or later (GPLv3+)

This module implements the [Authors (Year)] auditory model for [main purpose/
application]. The model features [distinctive characteristics: e.g., nonlinear
compression, binaural processing, temporal integration, etc.].

The implementation is ported from the MATLAB Auditory Modeling Toolbox (AMT)
and extended with PyTorch for gradient-based optimization and GPU acceleration.

[Optional: Additional paragraph about model significance, psychoacoustic basis,
or specific contributions compared to other models.]

References
----------
.. [1] [Primary model reference - full citation]

.. [2] [Secondary reference if model builds on previous work]

.. [3] [Additional references for specific components]

.. [4] [AMT repository reference]
"""
```

---

## **Model Classes (nn.Module)**

```python
class ModelName(nn.Module):
    r"""
    [Authors (Year)] auditory model [brief descriptor].
    
    [Full description paragraph 1: Model purpose, psychoacoustic phenomena
    it addresses, main features. 2-3 sentences.]
    
    [Full description paragraph 2: Implementation details, AMT compatibility,
    differentiability, GPU support. 2-3 sentences.]
    
    [Optional paragraph 3: Historical context, relation to other models,
    specific applications.]
    
    Algorithm Overview
    ------------------
    The model implements a [N]-stage auditory processing pipeline:
    
    **Stage 1: [Stage Name - e.g., Gammatone Filterbank]**
    
    [Brief description of what this stage does and why.]
    
    .. math::
        [Primary equation for this stage]
    
    [Additional explanation: parameter meanings, typical values, rationale.]
    
    **Stage 2: [Stage Name - e.g., Nonlinear Compression]**
    
    [Description with conditional behavior if applicable:]
    
    *[Option A name]* ([when used]):
    
    .. math::
        [Equation for option A]
    
    [Explanation of option A behavior]
    
    *[Option B name]* ([when used]):
    
    .. math::
        [Equation for option B]
    
    [Explanation of option B behavior]
    
    **Stage 3: [Stage Name]**
    
    [Description using inline math for simple cases: :math:`variable = expression`]
    
    .. math::
        [Equation if needed]
    
    [Bullet list of sub-steps if complex stage:]
    
    - Step 3a: [Description]
    - Step 3b: [Description with math :math:`x(t)`]
    - Step 3c: [Description]
    
    **Stage [N]: [Final Stage Name]**
    
    [Description of final processing stage]
    
    .. math::
        [Output equation with dimensionality notation]
    
    Output: [Description of output format, shape notation :math:`(B, T', F, M)`,
    explanation of each dimension.]
    
    Parameters
    ----------
    fs : float
        Sampling rate in Hz. Must match the audio sampling rate.
        Common values: [list typical values like 44100, 48000].
    
    flow : float, optional
        Lower frequency bound for [filterbank type] in Hz. Default: [value] Hz.
        [Explanation of what this controls].
    
    fhigh : float, optional
        Upper frequency bound for [filterbank type] in Hz. Default: [value] Hz.
        [Explanation and typical range].
    
    basef : float, optional
        Base frequency in Hz for centered analysis. If provided, ``flow`` and 
        ``fhigh`` are automatically computed as ``basef ± [N] ERB``, creating
        a narrow frequency range centered on ``basef``. Default: None (use
        ``flow``/``fhigh`` directly).
        
        [Explain use case: e.g., "Useful for frequency-specific analysis"]
    
    [stage]_[parameter] : type, optional
        [Parameter controlling specific stage]. Default: [value].
        
        [Detailed explanation with typical values, physiological motivation,
        effects on output, etc. Use math if needed: :math:`\\tau = 5` ms]
    
    [option]_type : {'option1', 'option2', 'option3'}, optional
        [What this parameter controls]. Default: 'option1'.
        
        - ``'option1'``: [Description with typical use case]
        - ``'option2'``: [Description with differences from option1]
        - ``'option3'``: [Description]
    
    preset : {'preset1', 'preset2', 'preset3'}, optional
        Pre-configured model variant matching published models. Default: None.
        
        - ``'preset1'``: [Model name/paper reference]
          
          * ``param1 = value1`` ([explanation])
          * ``param2 = value2``
          * [Additional configuration details]
        
        - ``'preset2'``: [Model name/paper reference]
          
          * [Configuration with differences highlighted]
        
        If None, [default behavior description].
    
    subfs : float, optional
        Desired output sampling rate in Hz for downsampling. If provided,
        output time dimension is downsampled from original sampling rate.
        Default: None (no downsampling, use ``fs``).
        
        [Explanation: e.g., "Useful for reducing computational cost when
        high temporal resolution not needed."]
    
    learnable : bool, optional
        If ``True``, [list which model components become trainable:
        e.g., "compression exponents, filter coefficients"]. Enables
        gradient-based optimization of model parameters.
        If ``False``, all parameters are fixed. Default: ``False``.
    
    return_stages : bool, optional
        If ``True``, return tuple of (output, stages_dict) where stages_dict
        contains intermediate processing stages for analysis/debugging.
        If ``False``, return only final output. Default: ``False``.
    
    dtype : torch.dtype, optional
        Data type for computations. Default: ``torch.float32``.
        Use ``torch.float64`` for higher numerical precision if needed.
    
    [component]_kwargs : dict, optional
        Additional keyword arguments passed to [ComponentClass].
        Default: None (use component defaults).
        
        [Example parameters that can be passed]
    
    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    
    [stage1] : ComponentClass
        Stage 1: [Brief description of component role].
    
    [stage2] : ComponentClass
        Stage 2: [Brief description].
    
    [stage_n] : ComponentClass
        Stage N: [Final processing module].
    
    fc : torch.Tensor
        Center frequencies of [auditory/frequency] channels, shape (num_channels,) in Hz.
        [Additional info: e.g., "Logarithmically/linearly spaced from flow to fhigh"]
    
    mfc : torch.Tensor, optional
        Center frequencies of [modulation/temporal] channels, shape (num_mod_channels,) in Hz.
        [When present and what it represents]
    
    num_channels : int
        Number of [auditory/frequency] channels.
        [How it's determined: e.g., "depends on flow, fhigh, ERB spacing"]
    
    [parameter]_config : type
        [Configuration parameters stored for reproducibility]
    
    Input Shape
    -----------
    x : torch.Tensor
        Audio signal with shape:
        
        - :math:`(B, T)` - Batch of signals
        - :math:`(C, T)` - Multi-channel input
        - :math:`(T,)` - Single signal
        
        where:
        
        - :math:`B` = batch size
        - :math:`C` = input channels (e.g., stereo L/R)
        - :math:`T` = time samples
        
        [Additional constraints: e.g., "Must be mono (single channel) or
        will be converted to mono by averaging"]
    
    Output Shape
    ------------
    When ``return_stages=False`` (default):
        torch.Tensor
            Output tensor with shape :math:`(B, T', F, ...)`:
            
            - :math:`B` = batch size
            - :math:`T'` = time samples (possibly downsampled if subfs specified)
            - :math:`F` = num_channels ([description])
            - [Additional dimensions with explanations]
            
            [Format description: e.g., "Each element represents [quantity]
            for frequency channel F at time T'"]
            
    When ``return_stages=True``:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            - First element: [final output description] (shape as above)
            - Second element: dict with keys:
              
              - ``'stage1_name'``: [Description], shape :math:`(B, F, T)`
              - ``'stage2_name'``: [Description], shape :math:`(B, F, T)`
              - ``'stage_n_name'``: [Description], shape [final shape]
              
            [Explanation of what intermediate stages contain and their use]
    
    Examples
    --------
    **Basic usage:**
    
    >>> import torch
    >>> from torch_amt.models import ModelName
    >>> 
    >>> # Create model with default parameters
    >>> model = ModelName(fs=48000)
    >>> 
    >>> # Generate [description] signal
    >>> audio = torch.randn(2, 24000) * 0.01  # [duration] @ [fs]
    >>> 
    >>> # Process through model
    >>> output = model(audio)
    >>> print(f"Input: {audio.shape}, Output: {output.shape}")
    Input: torch.Size([2, 24000]), Output: torch.Size([2, ..., ...])
    >>> 
    >>> # Inspect model configuration
    >>> print(f"Frequency channels: {model.num_channels}")
    Frequency channels: [N]
    >>> print(f"Center frequencies: {model.fc[:5]}")
    Center frequencies: tensor([...])
    
    **With intermediate stages for debugging:**
    
    >>> model_debug = ModelName(fs=48000, return_stages=True)
    >>> output, stages = model_debug(audio)
    >>> 
    >>> print(f"Available stages: {list(stages.keys())}")
    Available stages: ['stage1', 'stage2', ...]
    >>> print(f"After [stage1]: {stages['stage1'].shape}")
    After [stage1]: torch.Size([...])
    >>> print(f"After [stage2]: {stages['stage2'].shape}")
    After [stage2]: torch.Size([...])
    
    **Batch processing multiple signals:**
    
    >>> # Process batch of [N] signals
    >>> batch_audio = torch.randn(8, 48000) * 0.01
    >>> output_batch = model(batch_audio)
    >>> print(f"Batch output: {output_batch.shape}")
    Batch output: torch.Size([8, ..., ...])
    
    **Using basef for frequency-specific analysis:**
    
    >>> # Analyze around [frequency] Hz (±[N] ERB)
    >>> model_centered = ModelName(fs=48000, basef=1000)
    >>> print(f"Channels: {model_centered.num_channels}")
    Channels: [N]
    >>> print(f"Center frequencies: {model_centered.fc}")
    Center frequencies: tensor([...])
    
    **Using preset configurations:**
    
    >>> # Load published model configuration
    >>> model_preset = ModelName(fs=48000, preset='[preset_name]')
    >>> print(f"[Parameter]: {model_preset.[parameter]}")
    [Parameter]: [value]
    
    **With [specific feature/option]:**
    
    >>> # [Description of what this demonstrates]
    >>> model_feature = ModelName(
    ...     fs=48000,
    ...     [param1]=[value1],
    ...     [param2]=[value2]
    ... )
    >>> output_feature = model_feature(audio)
    
    **With downsampling for efficiency:**
    
    >>> # Downsample output to [rate] Hz
    >>> model_ds = ModelName(fs=48000, subfs=1000)
    >>> audio_long = torch.randn(1, 480000) * 0.01  # [duration]
    >>> output_ds = model_ds(audio_long)
    >>> print(f"Downsampled output: {output_ds.shape}")
    Downsampled output: torch.Size([1, ..., ...])  # T reduced significantly
    
    **Learnable model for neural network training:**
    
    >>> model_learnable = ModelName(fs=48000, learnable=True)
    >>> n_params = sum(p.numel() for p in model_learnable.parameters())
    >>> print(f"Trainable parameters: {n_params}")
    Trainable parameters: [N]
    >>> 
    >>> # Example training loop
    >>> optimizer = torch.optim.Adam(model_learnable.parameters(), lr=1e-3)
    >>> for epoch in range(num_epochs):
    ...     output = model_learnable(audio_batch)
    ...     loss = criterion(output, target)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    
    **Accessing model components:**
    
    >>> model = ModelName(fs=48000)
    >>> 
    >>> # Access filterbank characteristics
    >>> print(f"Auditory fc (Hz): {model.fc[:5]}")
    Auditory fc (Hz): tensor([...])
    >>> 
    >>> # Access specific stage parameters
    >>> print(f"[Component] config: {model.[component].get_parameters()}")
    [Component] config: {...}
    
    **[Additional example for specific use case]:**
    
    >>> # [Description]
    >>> [code example]
    
    Notes
    -----
    **Model Configuration:**
    
    The default configuration implements the [Authors (Year)] model with:
    
    - **[Component 1]**: [Specification with key parameters]
    - **[Component 2]**: [Specification]
    - **[Component N]**: [Final component specification]
    
    [Table if multiple configurations:]
    
    +------------+-------------+--------------+-------------+
    | Component  | Default     | Preset1      | Preset2     |
    +============+=============+==============+=============+
    | [param1]   | [value]     | [value]      | [value]     |
    +------------+-------------+--------------+-------------+
    | [param2]   | [value]     | [value]      | [value]     |
    +------------+-------------+--------------+-------------+
    
    **[Calibration/Scaling if applicable]:**
    
    [Explanation of calibration requirements, typical dB full scale conventions,
    importance of matching signal reference levels, etc.]
    
    Example:
    
    .. code-block:: python
    
        # For signals calibrated to [X] dB SPL at 0 dBFS
        model = ModelName(fs=48000, dboffset=[X])
        
        # For [specific convention]
        model = ModelName(fs=48000, dboffset=[Y])
    
    **basef Parameter:**
    
    When ``basef`` is specified, ``flow`` and ``fhigh`` are automatically
    computed to create a narrow frequency range:
    
    .. math::
        \\text{ERB}_{\\text{base}} = \\text{fc2erb}(\\text{basef})
    
    .. math::
        f_{\\text{low}} = \\text{erb2fc}(\\text{ERB}_{\\text{base}} - N)
    
    .. math::
        f_{\\text{high}} = \\text{erb2fc}(\\text{ERB}_{\\text{base}} + N)
    
    This creates ~[M] channels centered on ``basef``, useful for [use case].
    
    **Computational Complexity:**
    
    Processing time scales as:
    
    .. math::
        T_{\\text{compute}} \\propto T \\cdot [complexity expression]
    
    where :math:`T` = signal length, [other variables with explanations].
    
    For [typical example: 1 second @ 48 kHz]: ~[X]-[Y] seconds on CPU,
    ~[A]-[B] seconds on GPU.
    
    [Breakdown by stage if significant:]
    
    - Stage 1 ([component]): ~[X]% of total time
    - Stage N ([component]): ~[Y]% of total time
    
    **Memory Requirements:**
    
    Peak memory [with/without] intermediate stages:
    
    .. math::
        \\text{Memory} \\approx [expression in terms of B, T, F, etc.]
    
    For [example scenario]: ~[X]-[Y] MB.
    
    **Device Compatibility:**
    
    The model supports:
    
    - **CPU**: Full functionality, [performance notes]
    - **CUDA**: [X]x speedup, recommended for [use case]
    - **MPS** (Apple Silicon): [compatibility notes, known issues if any]
    
    [Transfer between devices:]
    
    .. code-block:: python
    
        model = ModelName(fs=48000).to('cuda')
        audio_gpu = audio.to('cuda')
        output_gpu = model(audio_gpu)
    
    **Psychoacoustic Applications:**
    
    The model is particularly suited for:
    
    - [Application 1 with brief explanation]
    - [Application 2]
    - [Application 3]
    - [Research area with relevant phenomena]
    
    **Comparison with [Other Models]:**
    
    [Brief comparison highlighting key differences]
    
    +------------------+------------------+------------------+
    | Feature          | This Model       | [Other Model]    |
    +==================+==================+==================+
    | [Feature 1]      | [Value/Yes/No]   | [Value/Yes/No]   |
    +------------------+------------------+------------------+
    | [Feature 2]      | [Value]          | [Value]          |
    +------------------+------------------+------------------+
    
    **Known Limitations:**
    
    [List any known limitations, assumptions, or constraints:]
    
    - [Limitation 1 with explanation]
    - [Limitation 2]
    - [Workarounds if available]
    
    See Also
    --------
    [Component1] : Stage 1 - [Brief description of role]
    [Component2] : Stage 2 - [Brief description]
    [ComponentN] : Stage N - [Final stage description]
    [RelatedModel1] : Related model with [key difference]
    [RelatedModel2] : Alternative approach for [purpose]
    
    References
    ----------
    .. [1] [Primary model citation - full bibliographic details]
    
    .. [2] [Secondary reference if model extends previous work]
    
    .. [3] [Component-specific references]
    
    .. [4] [AMT toolbox reference if applicable]
    
    .. [5] [Additional references for psychoacoustic background]
    """
    
    def __init__(self,
                 fs: float,
                 [parameters as documented above]):
        super().__init__()
        
        # Initialize components
        # ...
```

---

## **Methods**

### forward()

```python
def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Process audio through complete [model name] pipeline.
    
    [Full description: Sequential processing through all stages, automatic
    device handling, shape transformations, intermediate stage capture if
    return_stages=True, etc. 2-4 sentences.]
    
    Parameters
    ----------
    x : torch.Tensor
        Input audio signal. Shape:
        
        - :math:`(B, T)` - Batch of mono signals
        - :math:`(C, T)` - Multi-channel (will [describe conversion])
        - :math:`(T,)` - Single mono signal (batch dimension added automatically)
        
        where :math:`B` = batch size, :math:`C` = channels, :math:`T` = time samples.
        
        [Constraints: e.g., "Must contain at least [N] samples for filterbank initialization"]
    
    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        If ``return_stages=False`` (default):
            torch.Tensor with shape :math:`(B, T', F, ...)` containing
            [description of output content and meaning].
        
        If ``return_stages=True``:
            Tuple of (output, stages_dict) where:
            
            - output: torch.Tensor with final model output (shape as above)
            - stages_dict: Dict[str, torch.Tensor] with keys:
              
              - ``'[stage1_name]'``: After [stage 1], shape :math:`(B, F, T)`
              - ``'[stage2_name]'``: After [stage 2], shape :math:`(B, F, T)`
              - ``'[stage_n_name]'``: After [stage N], shape [final shape]
    
    Notes
    -----
    **Processing Steps:**
    
    1. [Brief description of each stage in order]
    2. [Stage 2]
    3. ...
    N. [Final stage]
    
    **Device Handling:**
    
    All internal components are automatically moved to match the input device.
    [Any special considerations for device transfer]
    
    **Batch Processing:**
    
    Single signals :math:`(T,)` are automatically expanded to :math:`(1, T)`.
    [Any batching optimizations or constraints]
    
    **[Special behavior if applicable]:**
    
    [E.g., "Downsampling is applied after Stage N to reduce output size",
    "Learnable parameters trigger [specific behavior]", etc.]
    """
    # Implementation
```

---

### extra_repr()

```python
def extra_repr(self) -> str:
    """
    Extra representation string for module printing.
    
    Returns
    -------
    str
        String containing key model parameters in compact format.
    """
    return (f"fs={self.fs}, num_channels={self.num_channels}, "
            f"[key_param1]={self.[param1]}, learnable={self.learnable}")
```

---

## CHECKLIST

### File Header

- [ ] Title with Model Name + Features, aligned `===`
- [ ] Author: Name Surname - Role @ Where
- [ ] License: GPLv3+
- [ ] Overview (2-3 paragraphs): Purpose, features, AMT compatibility
- [ ] References [1]-[4]: Primary paper, secondary, components, AMT

### Model Class

- [ ] `r"""` docstring (raw for LaTeX)
- [ ] One-line + Multi-paragraph description (purpose, implementation, context)
- [ ] **Algorithm Overview** with stage-by-stage breakdown
  - [ ] Each stage with name, math equations, explanations
  - [ ] Inline math `:math:` and display math `.. math::`
  - [ ] Output format description
- [ ] **Parameters** section
  - [ ] Core params: fs, flow/fhigh, basef (if applicable)
  - [ ] Stage-specific params with detailed explanations
  - [ ] Type options with bullet list (``'option'``)
  - [ ] Presets with nested bullets and config details
  - [ ] learnable, return_stages, dtype, subfs
  - [ ] Component kwargs dictionaries
  - [ ] All with proper defaults: ``None``, ``False``, ``'string'``
- [ ] **Attributes** section
  - [ ] Component references (stage1, stage2, ...)
  - [ ] fc, mfc if applicable (with shapes)
  - [ ] num_channels and derived quantities
- [ ] **Input Shape** section (separate from Output)
  - [ ] All input variations: (B,T), (C,T), (T,)
  - [ ] Dimension explanations
  - [ ] Constraints if any
- [ ] **Output Shape** section (separate from Input)
  - [ ] return_stages=False case with dimensions
  - [ ] return_stages=True case with tuple and dict keys
  - [ ] Shape explanations for each stage output
- [ ] **Examples** section (extensive)
  - [ ] Basic usage with shape printing
  - [ ] With intermediate stages
  - [ ] Batch processing
  - [ ] Using basef
  - [ ] Using presets
  - [ ] Specific features (compression types, etc.)
  - [ ] With downsampling
  - [ ] Learnable model with training loop
  - [ ] Accessing components and parameters
  - [ ] Additional use-case specific examples
- [ ] **Notes** section with multiple subsections
  - [ ] **Model Configuration**: Default specs with table if multiple configs
  - [ ] **[Calibration/Scaling]**: If applicable (dB conventions, etc.)
  - [ ] **basef Parameter**: Mathematical explanation if applicable
  - [ ] **Computational Complexity**: Time scaling with equations and examples
  - [ ] **Memory Requirements**: Peak memory with formula and example
  - [ ] **Device Compatibility**: CPU/CUDA/MPS support and notes
  - [ ] **Psychoacoustic Applications**: Use cases and phenomena
  - [ ] **Comparison with [Other Models]**: Table or bullet comparison
  - [ ] **Known Limitations**: List with explanations
- [ ] **See Also** section
  - [ ] All internal components with stage numbers
  - [ ] Related models with key differences
- [ ] **References** section
  - [ ] Primary model citation [1]
  - [ ] Secondary references [2-3]
  - [ ] Component references [4+]
  - [ ] AMT toolbox [last number]

### Methods

- [ ] `forward()`:
  - [ ] One-line + detailed description
  - [ ] Parameters with all shape variations
  - [ ] Returns with both return_stages cases
  - [ ] Notes: Processing steps, device handling, special behaviors
- [ ] `extra_repr()`:
  - [ ] Docstring with Returns section
  - [ ] Compact parameter string

### Sphinx Compatibility

- [ ] Sections with underline: `---`, `===`
- [ ] Inline math: `:math:\`...\``
- [ ] Display math: `.. math::`
- [ ] Code blocks: `.. code-block:: python`
- [ ] Code examples: `>>>`
- [ ] Cross-refs: `:class:`, `:meth:`, `:func:`
- [ ] Parameters: `\`\`param_name\`\``
- [ ] Proper escaping in LaTeX equations
