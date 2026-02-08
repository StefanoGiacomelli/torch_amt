# 🚀 GitHub Repository Setup Guide

This document provides step-by-step instructions for publishing the torch_amt package to GitHub and PyPI.

---

## ✅ Pre-Flight Checklist

Before publishing, ensure all these files are present and up-to-date:

### Core Files
- [x] `README.md` - Professional README with badges, examples, features
- [x] `LICENSE` - GPL-3.0-or-later license file
- [x] `CHANGELOG.md` - Version history following Keep a Changelog format
- [x] `CONTRIBUTING.md` - Guidelines for contributors
- [x] `CITATION.cff` - Academic citation metadata
- [x] `.gitignore` - Comprehensive ignore patterns
- [x] `pyproject.toml` - Package configuration for PyPI
- [x] `requirements.txt` - Python dependencies

### Package Structure
- [x] `torch_amt/__init__.py` - Main package entry point with full exports
- [x] `torch_amt/models/` - 6 complete auditory models
- [x] `torch_amt/common/` - 43+ building block components
- [x] `examples/` - Usage examples and tutorials
- [x] `tests/` - Test suite (add more tests if needed)

### GitHub Actions
- [x] `.github/workflows/tests.yml` - Automated testing on push/PR
- [x] `.github/workflows/publish.yml` - Auto-publish to PyPI on release

### Documentation
- [x] `dev/templates_docs_common.md` - Documentation standards for components
- [x] `dev/templates_docs_models.md` - Documentation standards for models

---

## 📋 Step-by-Step GitHub Setup

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `torch_amt`
3. Description: "PyTorch Auditory Modeling Toolbox - Differentiable implementations of computational auditory models"
4. Choose: **Public** (for open-source) or **Private** (initially)
5. **Don't** initialize with README, .gitignore, or license (you already have them)
6. Click "Create repository"

### 2. Initialize Local Git Repository

```bash
cd /Users/stefano/Documents/auditory_models/torch_amt

# Initialize git (if not already done)
git init

# Add all files
git add .

# First commit
git commit -m "feat: initial release of torch_amt v0.1.0

- 6 complete auditory models (Dau1997, Glasberg2002, King2019, Moore2016, Osses2021, Paulick2024)
- Modular components for custom pipelines
- Full Hardware support (CUDA, MPS, CPU)
- Differentiable for gradient-based optimization
- Type hints and comprehensive documentation
```

### 3. Connect to GitHub

Replace `[username]` with your GitHub username:

```bash
# Add remote
git remote add origin https://github.com/[username]/torch_amt.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Configure Repository Settings

On GitHub (https://github.com/[username]/torch_amt/settings):

#### General Settings
- [ ] Add topics: `pytorch`, `auditory-modeling`, `psychoacoustics`, `computational-neuroscience`, `deep-learning`
- [ ] Update description
- [ ] Add website (when docs are published)

#### Branches
- [ ] Set `main` as default branch
- [ ] Enable branch protection rules:
  - Require pull request reviews before merging
  - Require status checks to pass (tests)
  - Require branches to be up to date

#### GitHub Actions
- [ ] Enable "Read and write permissions" for workflows
  - Go to Settings → Actions → General → Workflow permissions
  - Select "Read and write permissions"
  - Check "Allow GitHub Actions to create and approve pull requests"

#### Secrets (for PyPI publishing)
- [ ] Add PyPI API token:
  - Go to Settings → Secrets and variables → Actions
  - Click "New repository secret"
  - Name: `PYPI_API_TOKEN`
  - Value: [Your PyPI API token - get from https://pypi.org/manage/account/token/]

---

## 🏷️ Creating Your First Release

### 1. Tag the Release

```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release version 0.1.0 - Initial public release

Features:
- 6 complete auditory models
- 43+ building blocks
- Full GPU support
- Differentiable implementations
- Comprehensive documentation"

# Push tag to GitHub
git push origin v0.1.0
```

### 2. Create GitHub Release

1. Go to https://github.com/[username]/torch_amt/releases/new
2. Choose tag: `v0.1.0`
3. Release title: `torch_amt v0.1.0 - Initial Release 🎉`
4. Description: Copy from CHANGELOG.md (v0.1.0 section)
5. Check "Set as the latest release"
6. Click "Publish release"

**This will automatically trigger the PyPI publishing workflow!**

---

## 📦 PyPI Publishing

### Option A: Automatic (via GitHub Actions) ⭐ Recommended

Once you create a GitHub release, the workflow will automatically:
1. Build the package
2. Run checks
3. Publish to PyPI

**Prerequisites:**
- PyPI account created
- API token added to GitHub secrets (see above)

### Option B: Manual Publishing

If you prefer to publish manually:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI first (recommended)
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ torch-amt

# If everything works, upload to real PyPI
twine upload dist/*
```

---

## 🔧 Post-Publishing Tasks

### 1. Update README Badges

After publishing, update these in README.md:

```markdown
[![PyPI version](https://badge.fury.io/py/torch-amt.svg)](https://badge.fury.io/py/torch-amt)
[![Downloads](https://pepy.tech/badge/torch-amt)](https://pepy.tech/project/torch-amt)
[![GitHub stars](https://img.shields.io/github/stars/[username]/torch_amt.svg)](https://github.com/[username]/torch_amt/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/[username]/torch_amt.svg)](https://github.com/[username]/torch_amt/issues)
```

### 2. Update URLs

Replace `[username]` in these files:
- `README.md` - GitHub links, star history
- `CITATION.cff` - repository-code URL
- `CONTRIBUTING.md` - Clone URL
- `pyproject.toml` - Homepage, Repository, Bug Tracker URLs
- `torch_amt/__init__.py` - __url__

### 3. Enable GitHub Features

- [ ] **Issues**: Enable issue tracker
- [ ] **Discussions**: Enable for Q&A and community
- [ ] **Wiki**: Optional documentation
- [ ] **Projects**: For roadmap tracking

### 4. Set Up Documentation (Optional but Recommended)

**Using Read the Docs:**

1. Sign up at https://readthedocs.org/
2. Import your GitHub repository
3. Configure build settings to use Sphinx
4. Documentation will be available at https://torch-amt.readthedocs.io

**Using GitHub Pages:**

```bash
cd docs/
make html
cp -r _build/html/* ../docs_gh_pages/
cd ../docs_gh_pages
git push
```

### 5. Add Shields/Badges

Update README.md with additional badges:
- CI/CD status
- PyPI version
- Downloads count  
- Code coverage
- Documentation status
- DOI (after publishing paper)

---

## 📢 Announcement Checklist

After everything is published:

- [ ] Announce on Twitter/X with #PyTorch hashtag
- [ ] Post on Reddit r/MachineLearning
- [ ] Share in relevant Discord/Slack communities
- [ ] Email to colleagues and collaborators
- [ ] Submit to PyTorch Hub (optional)
- [ ] Write blog post about the release
- [ ] Submit paper to JOSS (Journal of Open Source Software)

---

## 🔄 Maintenance Workflow

### For Bug Fixes

```bash
git checkout -b bugfix/fix-issue-123
# Make changes
git commit -m "fix(filterbanks): correct ERB spacing calculation"
git push origin bugfix/fix-issue-123
# Create PR on GitHub
```

### For New Features

```bash
git checkout -b feature/add-zilany2014
# Implement feature
git commit -m "feat(models): add Zilany2014 auditory nerve model"
git push origin feature/add-zilany2014
# Create PR on GitHub
```

### For New Releases

1. Update version in `torch_amt/__init__.py` and `pyproject.toml`
2. Update `CHANGELOG.md` with changes
3. Commit: `git commit -m "chore: bump version to X.Y.Z"`
4. Tag: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
5. Push: `git push && git push --tags`
6. Create release on GitHub

---

## 📊 Monitoring

### GitHub Insights

Check regularly:
- Traffic (views, clones)
- Community (issues, PRs, discussions)
- Dependency graph
- Security advisories

### PyPI Statistics

Monitor at https://pypi.org/project/torch-amt/:
- Download statistics
- Supported versions
- Dependencies

---

## 🆘 Troubleshooting

### "Permission denied" when pushing

```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:[username]/torch_amt.git
```

### GitHub Actions failing

- Check workflow logs in Actions tab
- Verify secrets are set correctly
- Test locally first: `pytest`, `black --check`, `isort --check`

### PyPI upload fails

- Verify API token is correct
- Check package name isn't taken
- Try TestPyPI first
- Ensure version number is unique (can't re-upload same version)

---

## 📚 Additional Resources

- **GitHub Docs**: https://docs.github.com/
- **PyPI Guide**: https://packaging.python.org/tutorials/packaging-projects/
- **Semantic Versioning**: https://semver.org/
- **Conventional Commits**: https://www.conventionalcommits.org/
- **Keep a Changelog**: https://keepachangelog.com/

---

## ✅ Final Checklist Before Publishing

- [ ] All tests passing locally
- [ ] Code formatted with black and isort
- [ ] Documentation complete and accurate
- [ ] Examples work correctly
- [ ] Version numbers updated everywhere
- [ ] CHANGELOG.md updated
- [ ] LICENSE file present
- [ ] README.md polished and professional
- [ ] .gitignore comprehensive
- [ ] No API keys or secrets in code
- [ ] All URLs updated with correct username
- [ ] GitHub repository created
- [ ] Local changes committed
- [ ] Code pushed to GitHub
- [ ] GitHub Actions enabled
- [ ] PyPI API token added to secrets
- [ ] Release created on GitHub

---

**You're ready to publish! 🚀**

Good luck with your release!
