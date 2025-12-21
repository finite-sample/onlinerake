# Contributing to OnlineRake

We welcome contributions to `onlinerake`! This guide will help you get started.

## Development Setup

**Requirements**
- Python 3.11 or later
- uv (recommended) or pip

**Clone and Setup**
```bash
git clone https://github.com/finite-sample/onlinerake.git
cd onlinerake
uv sync --group dev --group test
```

## Development Workflow

**Running Tests**
```bash
# Run comprehensive test suite
uv run pytest tests/test_onlinerake.py -v --cov=onlinerake --cov-report=term

# Test interactive tutorials
jupyter notebook docs/notebooks/
```

**Code Quality**
```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking  
uv run mypy onlinerake/
```

## Types of Contributions

**Bug Reports**
- Clear description of the problem
- Minimal code example to reproduce
- Your Python version and package versions

**Feature Requests**
- Clear description of the proposed feature
- Use case and motivation
- Consider if it fits the project scope

**Code Contributions**
- Fork the repository
- Create a feature branch
- Write tests for new functionality
- Ensure all tests pass
- Submit a pull request

## Code Guidelines

- Follow existing code style (ruff formatting)
- Add type hints for new functions
- Include docstrings for public APIs
- Write tests for new features
- Keep changes focused and atomic

## Questions?

Feel free to open an issue for questions about contributing!