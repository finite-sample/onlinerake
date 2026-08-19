"""Sphinx configuration — fleet standard via py-canon, plus notebooks."""

import os
from pathlib import Path

import pypandoc
from py_canon.sphinx import configure

ns = globals()
configure(ns)

# nbsphinx converts each notebook's markdown cells by spawning a `pandoc`
# binary, and the shared docs workflow's runner has none. pypandoc-binary
# carries one inside its wheel, so point PATH at it before the build starts.
os.environ["PATH"] = os.pathsep.join(
    [str(Path(pypandoc.get_pandoc_path()).parent), os.environ["PATH"]]
)

# configure() sets these; the standard has no notebook support, so extend
# rather than replace (its docstring asks for exactly this).
ns["extensions"] += ["nbsphinx", "sphinx.ext.mathjax"]
ns["exclude_patterns"] += ["**.ipynb_checkpoints"]
ns["intersphinx_mapping"].update(
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    }
)

# The notebooks are committed without outputs, so "auto" executes them during
# the build. That is deliberate: it is the only thing standing between a broken
# example and the published docs, and `-W` turns a failure into a red build.
nbsphinx_execute = "auto"
nbsphinx_allow_errors = False
nbsphinx_kernel_name = "python3"
