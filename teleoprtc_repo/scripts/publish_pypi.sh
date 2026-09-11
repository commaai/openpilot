#!/usr/bin/env bash

set -e

if [[ -z "$1" ]]; then
  echo "Usage: $0 <PyPI token>"
  exit 1
fi
PYPI_TOKEN="$1"

# install required packages
pip install --upgrade twine build

# build the package
python3 -m build

# upload to PyPI
REPOSITORY=""
if [[ -n "$TEST_UPLOAD" ]]; then
    REPOSITORY="--repository testpypi"
fi

python3 -m twine upload $REPOSITORY --username __token__ --password "$PYPI_TOKEN" dist/*
