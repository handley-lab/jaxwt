"""Sphinx configuration for jaxwavelets documentation."""
project = 'jaxwavelets'
copyright = '2026, Will Handley'
author = 'Will Handley'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'

autodoc_member_order = 'bysource'
numpydoc_show_class_members = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
