#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

testing_minimal = [
  "numpy",
  "torch==2.8.0",
  "pytest",
  "pytest-xdist",
  "pytest-timeout",
  "hypothesis",
  "z3-solver",
]

setup(name='tinygrad',
      version='0.11.0',
      description='You like pytorch? You like micrograd? You love tinygrad! <3',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = [
        'tinygrad',
        'tinygrad.apps',
        'tinygrad.codegen',
        'tinygrad.codegen.opt',
        'tinygrad.codegen.late',
        'tinygrad.engine',
        'tinygrad.frontend',
        'tinygrad.nn',
        'tinygrad.renderer',
        'tinygrad.runtime',
        'tinygrad.runtime.autogen',
        'tinygrad.runtime.autogen.am',
        'tinygrad.runtime.autogen.nv',
        'tinygrad.runtime.graph',
        'tinygrad.runtime.support',
        'tinygrad.runtime.support.am',
        'tinygrad.runtime.support.nv',
        'tinygrad.schedule',
        'tinygrad.shape',
        'tinygrad.uop',
        'tinygrad.viz',
      ],
      package_data = {'tinygrad': ['py.typed'], 'tinygrad.viz': ['index.html', 'assets/**/*', 'js/*']},
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=[],
      python_requires='>=3.10',
      extras_require={
        'arm': ["unicorn"],
        'triton': ["triton-nightly>=2.1.0.dev20231014192330"],
        'linting': [
            "pylint",
            "mypy==1.18.1",
            "typing-extensions",
            "pre-commit",
            "ruff",
            "numpy",
            "typeguard",
        ],
        #'mlperf': ["mlperf-logging @ git+https://github.com/mlperf/logging.git@5.0.0-rc3"],
        'testing_minimal': testing_minimal,
        'testing_unit': testing_minimal + [
            "tqdm",
            "safetensors",
            "tabulate",  # for sz.py
        ],
        'testing': testing_minimal + [
            "pillow",
            "onnx==1.18.0",
            "onnx2torch",
            "onnxruntime",
            "opencv-python",
            "tabulate",
            "tqdm",
            "safetensors",
            "transformers",
            "sentencepiece",
            "tiktoken",
            "blobfile",
            "librosa",
            "numba>=0.55",  # librosa needs numba but uv ignores python upper bounds and some numba versions require <python3.10
            "networkx",
            "nibabel",
            "bottle",
            "ggml-python",
            "capstone",
            "pycocotools",
            "boto3",
            "pandas",
            "influxdb3-python"
        ],
        'docs': [
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings[python]",
            "markdown-callouts",
            "markdown-exec[ansi]",
            "black",
            "numpy",
        ],
      },
      include_package_data=True)
