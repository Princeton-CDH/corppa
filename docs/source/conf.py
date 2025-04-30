# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from corppa import __version__

project = "corppa"
copyright = "2024,2025 Center for Digital Humanities, Princeton University"
author = "Center for Digital Humanities RSE Team, Princeton University"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    #    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

source_suffix = [".rst", ".md"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = []
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Ignore annotation-related modules
autodoc_mock_imports = [
    "prodigy",
    "pyarrow",
    "rapidfuzz",
    "spacy",
    "unidecode",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_theme_options = {
    # "description": "",
    "github_user": "Princeton-CDH",
    "github_repo": "corppa",
    # "codecov_button": True,  # enable once code coverage is higher
}

# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "localtoc.html",
        "searchbox.html",
        "sidebar_footer.html",
    ],
}



