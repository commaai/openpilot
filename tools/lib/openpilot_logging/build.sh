#!/usr/bin/env bash
set -e

uv venv
source .venv/bin/activate
uv sync

rm -rf cereal/
cp -avr ../../../cereal .


# quick test
# TODO: fix pythonpath
export PYTHONPATH=/home/batman/openpilot/tools/lib/openpilot_logging/
python openpilot_logging/logreader.py

rm -rf dist/
uv python -c "import setuptools; setuptools.setup()" sdist

# twine upload dist/* --verbose
