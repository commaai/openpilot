#!/usr/bin/env python3
"""Build self-extracting zipapp installers for openpilot.

Produces one executable per variant in installers/ by bundling:
  - __main__.py (extraction + installer UI)
  - config.py   (generated per-variant: branch, internal flag, git URL)
  - raylib/, pyray/, cffi/ packages + _cffi_backend.*.so
  - assets: fonts + continue_openpilot.sh

Each output is a shebang + zip file, directly executable with system Python.
"""
import os
import shutil
import sys
import tempfile
import zipapp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "installers")

GIT_URL = "https://github.com/commaai/openpilot.git"

INSTALLERS = [
    # (name,       branch,             internal)
    ("openpilot",          "release3",         False),
    ("openpilot_test",     "release3-staging", False),
    ("openpilot_nightly",  "nightly",          False),
    ("openpilot_internal", "nightly-dev",      True),
]


def find_package(name):
    """Return the directory of an installed Python package."""
    for d in sys.path:
        p = os.path.join(d, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
            return p
    raise FileNotFoundError(f"Package '{name}' not found in sys.path")


def find_cffi_backend():
    """Return the path to _cffi_backend.*.so."""
    import _cffi_backend
    return _cffi_backend.__file__


def stage_common(staging):
    """Copy shared files into the staging directory (everything except config.py)."""
    # Main entry point
    shutil.copy2(os.path.join(SCRIPT_DIR, "__main__.py"), staging)

    # Python packages (pure + native)
    for pkg in ("raylib", "pyray", "cffi"):
        shutil.copytree(find_package(pkg), os.path.join(staging, pkg),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Top-level cffi backend .so
    cffi_so = find_cffi_backend()
    shutil.copy2(cffi_so, os.path.join(staging, os.path.basename(cffi_so)))

    # Assets
    assets = os.path.join(staging, "assets")
    os.makedirs(assets)
    fonts_dir = os.path.join(ROOT_DIR, "selfdrive", "assets", "fonts")
    for ttf in ("inter-ascii.ttf", "Inter-Bold.ttf", "Inter-Light.ttf"):
        shutil.copy2(os.path.join(fonts_dir, ttf), assets)
    shutil.copy2(os.path.join(SCRIPT_DIR, "continue_openpilot.sh"), assets)


def write_config(staging, branch, internal):
    """Generate config.py with padding for fork patching (binary search-and-replace)."""
    # The '?' sentinel + spaces let fork tools patch the URL/branch in-place
    # inside the uncompressed zip, matching the C++ installer's pattern.
    pad = " " * 64
    with open(os.path.join(staging, "config.py"), "w") as f:
        f.write(f'GIT_URL = "{GIT_URL}?{pad}".split("?")[0]\n')
        f.write(f'BRANCH = "{branch}?{pad}".split("?")[0]\n')
        f.write(f"INTERNAL = {internal}\n")


def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, branch, internal in INSTALLERS:
        with tempfile.TemporaryDirectory() as staging:
            stage_common(staging)
            write_config(staging, branch, internal)

            output = os.path.join(OUTPUT_DIR, f"installer_{name}")
            zipapp.create_archive(staging, output, interpreter="/usr/bin/env python3")
            os.chmod(output, 0o755)
            print(f"  {output}")


if __name__ == "__main__":
    build()
