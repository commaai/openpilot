#!/usr/bin/env bash
set -eo xtrace

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# TODO: why doesn't uv do this?
export PYTHONPATH=$DIR
echo $PYTHONPATH

uv venv
source .venv/bin/activate
uv sync

# TODO: just copy log.capnp and car.capnp with a simple loader file
#  the code might want to be structured to pull the latest definitions and store to cache (every time? if too old? by flag?)
mkdir -p openpilot_logging/cereal/include/
cp -rL ../../../cereal/*.capnp openpilot_logging/cereal/
cp -rL ../../../cereal/include/*.capnp openpilot_logging/cereal/include/

# quick test
python openpilot_logging/logreader.py 1d3dc3e03047b0c7/000000dd--455f14369d/0/q > /dev/null

rm -rf dist/
python -c "import setuptools; setuptools.setup()" sdist

#twine upload dist/* --verbose
