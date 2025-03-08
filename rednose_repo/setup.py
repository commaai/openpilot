import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

setup(
  name='rednose',
  version='0.0.1',
  url='https://github.com/commaai/rednose',
  author='comma.ai',
  author_email='harald@comma.ai',
  packages=find_packages(),
  platforms='any',
  license='MIT',
  package_data={'': ['helpers/chi2_lookup_table.npy', 'templates/*']},
  install_requires=[
    'sympy',
    'numpy',
    'scipy',
    'tqdm',
    'cffi',
  ],
  ext_modules=[],
  description="Kalman filter library",
  long_description='See https://github.com/commaai/rednose',
)
