"""Sphinx configuration — fleet standard via py-canon, plus the examples gallery."""

from py_canon.sphinx import configure

ns = globals()
configure(ns)

# configure() sets these; the standard has no gallery, so extend rather than
# replace (its docstring asks for exactly this).
ns["extensions"] += ["sphinx_gallery.gen_gallery", "sphinx.ext.mathjax"]
ns["intersphinx_mapping"].update(
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    }
)

# Each example runs at build time; an example that raises fails the build.
sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": "/plot_",
    "download_all_examples": False,
}
