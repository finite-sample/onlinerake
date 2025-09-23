Contributing
============

We welcome contributions to ``onlinerake``! This guide will help you get started.

Development Setup
-----------------

**Requirements**

- Python 3.10 or later
- Git

**Setup Steps**

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/your-username/onlinerake.git
   cd onlinerake

3. Install in development mode:

.. code-block:: bash

   pip install -e .
   pip install pytest black flake8 sphinx sphinx-rtd-theme myst-parser

Code Style
----------

We use ``black`` for code formatting and ``flake8`` for linting:

.. code-block:: bash

   # Format code
   black onlinerake/
   
   # Check formatting
   black --check onlinerake/
   
   # Run linting
   flake8 onlinerake/

Running Tests
-------------

Run the test suite to make sure everything works:

.. code-block:: bash

   # Run all tests
   pytest test_onlinerake.py -v
   
   # Run with coverage
   pytest test_onlinerake.py --cov=onlinerake --cov-report=term
   
   # Run simulation tests
   python -m onlinerake.simulation
   python realistic_examples.py

Types of Contributions
----------------------

**Bug Reports**
   Found a bug? Please open an issue with:
   - Clear description of the problem
   - Minimal code example to reproduce
   - Your Python version and package versions

**Feature Requests**
   Have an idea for improvement? Open an issue with:
   - Description of the proposed feature
   - Use case and motivation
   - Proposed API (if applicable)

**Code Contributions**
   Ready to contribute code? Here's the process:
   
   1. Check existing issues or open a new one
   2. Create a feature branch: ``git checkout -b feature-name``
   3. Make your changes with tests
   4. Run the test suite and style checks
   5. Commit with descriptive messages
   6. Push and create a pull request

Documentation
-------------

Documentation improvements are always welcome:

.. code-block:: bash

   # Build docs locally
   cd docs/
   make html
   
   # View in browser
   open _build/html/index.html

Documentation is written in reStructuredText and built with Sphinx.

Areas for Contribution
----------------------

Here are some areas where contributions would be especially valuable:

**Algorithm Enhancements**
   - Support for multi-level categorical variables
   - Adaptive learning rate schedules
   - Alternative loss functions
   - Regularization techniques

**Performance Optimizations**
   - Vectorized operations for batch processing
   - Memory-efficient implementations
   - GPU acceleration (optional)

**Additional Features**
   - Integration with popular survey platforms
   - Visualization tools for weight evolution
   - Export functionality for different formats
   - Real-time monitoring dashboards

**Testing & Quality**
   - More edge case tests
   - Performance benchmarks
   - Integration tests
   - Property-based testing

Pull Request Guidelines
-----------------------

To ensure smooth review process:

1. **Focus**: Keep changes focused and atomic
2. **Tests**: Add tests for new functionality
3. **Documentation**: Update docs for API changes
4. **Style**: Follow existing code conventions
5. **Commit Messages**: Use descriptive commit messages

Example commit message:

.. code-block:: text

   Add support for custom weight initialization
   
   - Allow users to provide initial weights via new parameter
   - Add validation for weight dimensions and positivity
   - Update documentation and examples
   - Add comprehensive tests for edge cases

Review Process
--------------

All contributions go through code review:

1. Automated checks (CI/CD) must pass
2. Manual review by maintainers
3. Discussion and iteration as needed
4. Merge when approved

Questions?
----------

Feel free to:

- Open an issue for questions
- Start a discussion for broader topics
- Reach out to maintainers directly

Thank you for contributing to ``onlinerake``!