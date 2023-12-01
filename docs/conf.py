# type: ignore

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from os.path import exists

from openpilot.common.basedir import BASEDIR
from openpilot.system.version import get_version

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

VERSION = get_version()


# -- Project information -----------------------------------------------------

project = 'openpilot docs'
copyright = '2021, comma.ai' # noqa: A001
author = 'comma.ai'
version = VERSION
release = VERSION
language = 'en'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',   # Auto-generate docs
  'sphinx.ext.viewcode',  # Add view code link to modules
  'sphinx_rtd_theme',     # Read The Docs theme
  'myst_parser',          # Markdown parsing
  'breathe',              # Doxygen C/C++ integration
  'sphinx_sitemap',       # sitemap generation for SEO
]

myst_html_meta = {
  "description": "openpilot docs",
  "keywords": "op, openpilot, docs, documentation",
  "robots": "all,follow",
  "googlebot": "index,follow,snippet,archive",
  "property=og:locale": "en_US",
  "property=og:site_name": "docs.comma.ai",
  "property=og:url": "https://docs.comma.ai",
  "property=og:title": "openpilot Documentation",
  "property=og:type": "website",
  "property=og:image:type": "image/jpeg",
  "property=og:image:width": "400",
  "property=og:image": "https://docs.comma.ai/_static/logo.png",
  "property=og:image:url": "https://docs.comma.ai/_static/logo.png",
  "property=og:image:secure_url": "https://docs.comma.ai/_static/logo.png",
  "property=og:description": "openpilot Documentation",
  "property=twitter:card": "summary_large_image",
  "property=twitter:logo": "https://docs.comma.ai/_static/logo.png",
  "property=twitter:title": "openpilot Documentation",
  "property=twitter:description": "openpilot Documentation"
}

html_baseurl = 'https://docs.comma.ai/'
sitemap_filename = "sitemap.xml"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- c docs configuration ---------------------------------------------------

# Breathe Configuration
# breathe_default_project = "c_docs"
breathe_build_directory = f"{BASEDIR}/build/docs/html/xml"
breathe_separate_member_pages = True
breathe_default_members = ('members', 'private-members', 'undoc-members')
breathe_domain_by_extension = {
  "h": "cc",
}
breathe_implementation_filename_extensions = ['.c', '.cc']
breathe_doxygen_config_options = {}
breathe_projects_source = {}

# only document files that have accompanying .cc files next to them
print("searching for c_docs...")
for root, _, files in os.walk(BASEDIR):
  found = False
  breath_src = {}
  breathe_srcs_list = []

  for file in files:
    ccFile = os.path.join(root, file)[:-2] + ".cc"

    if file.endswith(".h") and exists(ccFile):
      f = os.path.join(root, file)

      parent_dir_abs = os.path.dirname(f)
      parent_dir = parent_dir_abs[len(BASEDIR) + 1:]
      parent_project = parent_dir.replace('/', '_')
      print(f"\tFOUND: {f} in {parent_project}")

      breathe_srcs_list.append(file)
      found = True

    if found:
      breath_src[parent_project] = (parent_dir_abs, breathe_srcs_list)
      breathe_projects_source.update(breath_src)

print(f"breathe_projects_source: {breathe_projects_source.keys()}")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_show_copyright = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'
html_theme_options = {
  'logo_only': False,
  'display_version': True,
  'vcs_pageview_mode': 'blob',
  'style_nav_header_background': '#000000',
}
html_extra_path = ['_static']
