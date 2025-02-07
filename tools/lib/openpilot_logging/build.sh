#!/usr/bin/env bash
set -e

uv venv
source .venv/bin/activate
uv sync

rm -rf cereal/
cp -vrL ../../../cereal .


# quick test
# TODO: fix pythonpath
export PYTHONPATH=/home/batman/openpilot/tools/lib/openpilot_logging/
python openpilot_logging/logreader.py 1d3dc3e03047b0c7/000000dd--455f14369d/0/q > /dev/null

rm -rf dist/
python -c "import setuptools; setuptools.setup()" sdist

# twine upload dist/* --verbose
