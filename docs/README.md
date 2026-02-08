# torch_amt Documentation

This directory contains the Sphinx documentation for torch_amt.

## Building Documentation Locally

### 1. Install Documentation Dependencies

```bash
# From the torch_amt root directory
pip install -e ".[docs]"
```

This will install:

- sphinx
- sphinx-rtd-theme (Read the Docs theme)
- sphinx-autodoc-typehints (better type hint rendering)
- myst-parser (Markdown support)

### 2. Build HTML Documentation

```bash
cd docs/
make html
```

### 3. View Documentation

Open `_build/html/index.html` in your web browser:

```bash
# On macOS
open _build/html/index.html

# On Linux
xdg-open _build/html/index.html

# On Windows
start _build/html/index.html
```

## Other Build Commands

```bash
# Clean previous builds
make clean

# Build PDF (requires LaTeX)
make latexpdf

# Build epub format
make epub

# Check for broken links
make linkcheck

# See all available targets
make help
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Homepage
│   ├── _static/          # Static files (CSS, images, etc.)
│   ├── _templates/       # Custom HTML templates
│   └── api/              # API reference pages
├── _build/               # Generated documentation (not in git)
├── Makefile              # Build commands (Unix/Mac)
└── make.bat              # Build commands (Windows)
```

## Updating Documentation

The API documentation is automatically generated from docstrings in the code.
To update:

1. Edit docstrings in the Python source code
2. Rebuild with `make html`
3. Refresh browser to see changes

## Publishing to Read the Docs

The documentation is configured to work with Read the Docs. Simply:

1. Connect your GitHub repository to readthedocs.org
2. RTD will automatically build docs on every commit
3. Docs will be available at https://torch-amt.readthedocs.io
