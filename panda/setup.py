"""
    Panda CAN Controller Dongle
    ~~~~~

    Setup
    `````

    $ pip install . # or python setup.py install
"""

import codecs
import os
import re
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
  """Taken from pypa pip setup.py:
  intentionally *not* adding an encoding option to open, See:
  https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
  """
  return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
  version_file = read(*file_paths)
  version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                            version_file, re.M)
  if version_match:
    return version_match.group(1)
  raise RuntimeError("Unable to find version string.")

setup(
  name='pandacan',
  version=find_version("python", "__init__.py"),
  url='https://github.com/commaai/panda',
  author='comma.ai',
  author_email='',
  packages=[
    'panda',
  ],
  package_dir={'panda': 'python'},
  platforms='any',
  license='MIT',
  install_requires=[
    'libusb1',
  ],
  extras_require = {
    'dev': [
      "scons",
      "pycryptodome >= 3.9.8",
      "cffi",
      "flaky",
      "pytest",
      "pytest-mock",
      "pytest-xdist",
      "pytest-timeout",
      "pytest-randomly",
      "parameterized",
      "pre-commit",
      "numpy",
      "ruff",
      "spidev",
      "setuptools", # for setup.py
    ],
  },
  ext_modules=[],
  description="Code powering the comma.ai panda",
  long_description='See https://github.com/commaai/panda',
  classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    "Natural Language :: English",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: System :: Hardware",
  ],
)
