#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinygrad',
      version='0.10.0',
      description='You like pytorch? You like micrograd? You love tinygrad! <3',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['tinygrad', 'tinygrad.runtime.autogen', 'tinygrad.codegen', 'tinygrad.nn', 'tinygrad.renderer', 'tinygrad.engine',
                  'tinygrad.runtime', 'tinygrad.runtime.support', 'tinygrad.runtime.graph', 'tinygrad.shape'],
      package_data = {'tinygrad': ['py.typed']},
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=[],
      python_requires='>=3.10',
      extras_require={
        'llvm': ["llvmlite"],
        'arm': ["unicorn"],
        'triton': ["triton-nightly>=2.1.0.dev20231014192330"],
        'linting': [
            "pylint",
            "mypy==1.13.0",
            "typing-extensions",
            "pre-commit",
            "ruff",
            "types-tqdm",
        ],
        #'mlperf': ["mlperf-logging @ git+https://github.com/mlperf/logging.git@4.1.0-rc3"],
        'testing': [
            "numpy",
            "torch",
            "jax",
            "pillow",
            "pytest",
            "pytest-xdist",
            "onnx==1.16.0",
            "onnx2torch",
            "opencv-python",
            "tabulate",
            "tqdm",
            "safetensors",
            "transformers",
            "sentencepiece",
            "tiktoken",
            "blobfile",
            "librosa",
            "networkx",
            "hypothesis",
            "nibabel",
            "bottle",
            "ggml-python"
        ],
        'webgpu': ["wgpu"],
        'docs': [
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings[python]",
            "markdown-callouts",
            "markdown-exec[ansi]",
            "black",
            "numpy",
        ],
        'testing_tf': [
            "tensorflow==2.15.1",
            "tensorflow_addons",
        ]
      },
      include_package_data=True)
