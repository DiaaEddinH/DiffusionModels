# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DiffusionModels'
copyright = '2026, Diaa Eddin Habibi'
author = 'Diaa Eddin Habibi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"myst_parser",
	"sphinx.ext.autodoc",
	"sphinx.ext.napoleon",
	"sphinx.ext.mathjax",

]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = {'.rst' : 'restructuredtext', '.md' : 'markdown'}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
