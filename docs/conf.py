# type: ignore
# pylint: skip-file

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
from os.path import exists
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
OPENPILOT_ROOT = os.path.abspath(r'../../') # from openpilot/build/docs


# -- Project information -----------------------------------------------------

project = 'openpilot'
copyright = '2021, comma.ai'
author = 'comma.ai'


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
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- c docs configuration ---------------------------------------------------

# Breathe Configuration
# breathe_default_project = "c_docs"
breathe_build_directory = f"{OPENPILOT_ROOT}/build/docs/html/xml"
breathe_separate_member_pages = True
breathe_default_members = ('members', 'private-members', 'undoc-members')
breathe_domain_by_extension = {
        "h" : "cc",
        }
breathe_implementation_filename_extensions = ['.c', '.cc', '.cpp']
breathe_doxygen_config_options = {}
breathe_projects_source  = {
        # "loggerd" : ("../../../selfdrive/loggerd", ["logger.h"])
        }

# only document files that have accompanying .cc files next to them
print("searching for c_docs...")
for root, dirs, files in os.walk(OPENPILOT_ROOT):
        found = False
        breath_src = {}
        breathe_srcs_list = []

        for file in files:
                ccFile = os.path.join(root, file)[:-2] +".cc"

                if file.endswith(".h") and exists(ccFile):
                        f = os.path.join(root, file)
                        parent_dir = os.path.basename(os.path.dirname(f))
                        parent_dir_abs = os.path.dirname(f)
                        print(f"\tFOUND: {f} in {parent_dir} ({parent_dir_abs})")

                        breathe_srcs_list.append(file)
                        # breathe_srcs_list.append(ccFile)
                        found = True

                # print(f"\tbreathe_srcs_list: {breathe_srcs_list}")

                if found:
                        breath_src[parent_dir] = (parent_dir_abs, breathe_srcs_list)
                        breathe_projects_source.update(breath_src)

print(f"breathe_projects_source: {breathe_projects_source.keys()}")
# input("Press Enter to continue...")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
