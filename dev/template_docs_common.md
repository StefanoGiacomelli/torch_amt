# 📋 **TORCH_AMT DOCUMENTATION & CODE STRUCTURE STANDARDS (/COMMON)**

## Header

```python
"""
[Module Title]
[==============]  [← Underline with '=' exactly below]

Author: 
    Name Surname - Role @ Where

License:
    GNU General Public License v3.0 or later (GPLv3+)

[Overview paragraph: general description of the module, what it contains, 
enumeration of main classes/functions if relevant]

[Second paragraph: implementation details, compatibility with AMT MATLAB/Octave, 
CASP framework if applicable]

References
----------
.. [1] [Complete reference first paper - AMT]

.. [2] [Complete reference second paper - AMT Toolbox]
       
.. [3] [Reference AMT SourceForge link]
"""
```

---

## Functions

```python
@torch.jit.script  # If JIT-compiled
def _function_name_jit(param1: torch.Tensor,
                       param2: torch.Tensor,
                       param3: float) -> torch.Tensor:
    """
    One-line summary ending with period.
    
    [Optional: More detailed description of functionality and purpose.
    Especially important for complex functions.]
    
    Parameters
    ----------
    param1 : torch.Tensor
        Description of param1. Include shape if relevant [B, F, T].
    
    param2 : torch.Tensor
        Description of param2 with units if applicable (Hz, seconds, etc).
    
    param3 : float
        Description.
    
    Returns
    -------
    torch.Tensor
        Description of return value with shape [B, F, T].
    
    Notes
    -----
    [Optional section for algorithm details, performance notes, 
    numerical considerations, etc.]
    """
    # Implementation
```

---

## **Classes (nn.Module)**

```python
class ClassName(nn.Module):
    r"""[One-line description ending with period.]
    
    [Full description paragraph: Cosa fa la classe, contesto scientifico,
    ruolo nell'auditory modeling. 2-4 frasi.]
    
    [Optional secondo paragrafo: Dettagli comportamento, varianti, note
    sull'implementazione se rilevante.]
    
    Algorithm Overview
    ------------------
    [Descrizione step-by-step dell'algoritmo implementato:]
    
    1. **[Step 1 name]**: [Description]
       
       .. math::
           [LaTeX equation]
       
       [Optional explanation]
    
    2. **[Step 2 name]**: [Description]
       
       - **[Sub-option A]**: [Detail]
       - **[Sub-option B]**: [Detail]
       
       .. math::
           [LaTeX multi-line equations]
    
    3. **[Step 3 name]**: [Description with inline math :math:`x(t)`]
    
    Parameters
    ----------
    param_name : type
        Full description of parameter. Explain what it controls, units,
        default behavior.
        
        If parameter has multiple options:
        
        * ``'option1'``: Description of option 1
        * ``'option2'``: Description of option 2 with details
        
        Default: [value] ([explanation]).
    
    another_param : type, optional
        Description with defaults and constraints.
        Include equations if needed: :math:`\tau = 5` ms.
        Default: [value].
    
    preset : {'preset1', 'preset2', 'preset3'}, optional
        Configuration preset:
        
        - **'preset1'**: [Model name] ([Authors Year])
          
          * ``param1 = value1``
          * ``param2 = value2`` ([explanation])
          * [Additional details]
        
        - **'preset2'**: [Model name] ([Authors Year])
          
          * [Configuration details in bullet list]
        
        Default: ``None`` ([behavior when None]).
    
    learnable : bool, optional
        If ``True``, [list which parameters become trainable].
        If ``False``, [fixed behavior].
        Default: ``False``.
    
    dtype : torch.dtype, optional
        Data type for computations. Default: ``torch.float32``.
    
    Attributes
    ----------
    attr1 : type
        Description of internal state attribute. Include shape if tensor.
    
    attr2 : torch.Tensor or nn.Parameter
        Description with conditional behavior (learnable vs buffer).
        Shape: ``(dim1, dim2)``.
    
    Shapes
    ------
    - Input: :math:`(B, F, T)` or :math:`(F, T)` where
        * :math:`B` = batch size
        * :math:`F` = frequency channels  
        * :math:`T` = time samples
    
    - Output: [Same shape as input / Different shape with description]
    
    Notes
    -----
    **[Section Header if multiple topics]:**
    
    [Important implementation details, computational complexity,
    connection to neuroscience/psychoacoustics, preset differences,
    numerical considerations, device handling, etc.]
    
    **[Another Section]:**
    
    [Additional notes organized by topic]
    
    **Connection to [Model Name]:**
    
    [How this relates to broader models]
    
    See Also
    --------
    OtherClass : Brief description of relation
    AnotherClass : How they interact
    some_function : Related utility function
    
    Examples
    --------
    **Basic usage with default parameters:**
    
    >>> import torch
    >>> from torch_amt.common.module import ClassName
    >>> 
    >>> # Create instance
    >>> obj = ClassName(fs=16000)
    >>> print(obj)
    ClassName(fs=16000, [key_params])
    >>> 
    >>> # Process signal
    >>> x = torch.randn(2, 31, 16000)  # [Explanation of dimensions]
    >>> y = obj(x)
    >>> print(f"Input: {x.shape} -> Output: {y.shape}")
    Input: torch.Size([2, 31, 16000]) -> Output: torch.Size([2, 31, 16000])
    
    **Using preset configurations:**
    
    >>> # Preset example
    >>> obj_preset = ClassName(fs=16000, preset='preset_name')
    >>> print(f"Parameter: {obj_preset.param_name:.1f}")
    Parameter: 10.0
    
    **Learnable parameters for training:**
    
    >>> obj_learn = ClassName(fs=16000, learnable=True)
    >>> print(f"Trainable: {sum(p.numel() for p in obj_learn.parameters())}")
    Trainable: 42
    >>> 
    >>> # In training loop
    >>> optimizer = torch.optim.Adam(obj_learn.parameters(), lr=1e-3)
    
    **[Additional examples as needed]:**
    
    >>> # More complex usage scenarios

    References
    ----------
    .. [1] [Author(s)], "[Title]," *Journal*, vol. X, no. Y, pp. ZZ-WW, Year.
    
    .. [2] [Full citation for second reference]
    
    .. [3] [Additional references specific to this class]
    
    """
    
    def __init__(self, ...):
        # Implementation
```

---

### Methods (Private & Public)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    [One-line summary of what forward does.]
    
    [Full description: cosa processa, come gestisce input, 
    comportamento speciale, refresh automatici se learnable, etc.
    2-4 frasi]
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal. Shape: :math:`(B, F, T)` or :math:`(F, T)` where
        
        - B = batch size
        - F = frequency channels
        - T = time samples
    
    Returns
    -------
    torch.Tensor
        Output signal [description]. Same shape as input.
    
    Notes
    -----
    **[Section if needed]:**
    
    [Implementation details: learnable updates, device handling,
    phase behavior, special cases, etc.]
    
    **[Another section]:**
    
    [Performance notes, gradient flow, etc.]
    """
    # Implementation
```

---

## **extra_repr()**

```python
def extra_repr(self) -> str:
    """
    "Extra representation string for module printing."
    
    Returns
    -------
    str
        String containing key module parameters [in MINIMAL format / 
        for debugging / etc.].
    """
    return (f"fs={self.fs}, param1={self.param1}, "
            f"param2={self.param2:.1f}, learnable={self.learnable}")
```

---

## CHECKLIST

### File Header

- [ ] Title with `===` aligned
- [ ] Author: Name Surname - Role @ Where
- [ ] License: GPLv3+
- [ ] Overview (2 paragraphs)
- [ ] References AMT [1], [2], [3]

### Class

- [ ] `r"""` docstring (raw for LaTeX)
- [ ] One-line + Full description
- [ ] Algorithm Overview with steps + math
- [ ] Detailed Parameters (type, description, default)
- [ ] Attributes (internal state)
- [ ] Shape section (I/O dimensions)
- [ ] Notes with sub-headings `**Topic:**`
- [ ] See Also
- [ ] Specific References
- [ ] Examples with >>> and output

### Methods

- [ ] One-line summary
- [ ] Parameters with type + description
- [ ] Returns with type + description
- [ ] Notes if necessary
- [ ] NO Examples (except forward in some cases)

### Sphinx Compatibility

- [ ] Sections with underline: `---`, `===`
- [ ] Inline math: `:math:\`...\``
- [ ] Display math: `.. math::`
- [ ] Code examples: `>>>`
- [ ] Cross-refs: `:class:`, `:meth:`
- [ ] Parameters: `\`\`param_name\`\``
