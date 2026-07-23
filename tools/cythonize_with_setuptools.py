#!/usr/bin/env python3
"""Run Cython with setuptools' Python 3.12 distutils compatibility shim."""

import setuptools  # noqa: F401
from Cython.Build.Cythonize import main


if __name__ == "__main__":
  raise SystemExit(main())
