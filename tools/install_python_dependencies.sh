#!/usr/bin/env bash
set -e

# Increase the pip timeout to handle TimeoutError
export PIP_DEFAULT_TIMEOUT=200

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

# updating uv on macOS results in 403 sometimes
function update_uv() {
  for i in $(seq 1 5);
  do
    if uv self update; then
      return 0
    else
      sleep 2
    fi
  done
  echo "Failed to update uv 5 times!"
}

if ! command -v "uv" > /dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN='$HOME/.cargo/env'
  ADD_PATH_CMD=". \"$UV_BIN\""
  eval $ADD_PATH_CMD
fi

echo "updating uv..."
update_uv

# TODO: remove --no-cache once this is fixed: https://github.com/astral-sh/uv/issues/4378
echo "installing python packages..."
ARCH=$(uname -m)
if [[ $ARCH == "aarch64" ]]; then
	apt install python3 python3-pip cmake curl -y
	curl https://files.pythonhosted.org/packages/5a/97/ca40c4d7d36162ddfd0bb96a89206469a95b925faf67046ba6e4b5b78283/casadi-3.6.5.tar.gz -o casadi-3.6.5.tar.gz
	tar -xvf casadi-3.6.5.tar.gz

	cd casadi-3.6.5
	# build casadi wheel
	python3 setup.py bdist_wheel
	echo "built casadi wheel"
	# update the uv.lock file
	uv add casadi-3.6.5/dist/casadi-3.6.5-cp312-cp312-linux_aarch64.whl
	cd ..
fi

uv --no-cache sync --all-extras
source .venv/bin/activate

echo "PYTHONPATH=${PWD}" > $ROOT/.env
if [[ "$(uname)" == 'Darwin' ]]; then
  echo "# msgq doesn't work on mac" >> $ROOT/.env
  echo "export ZMQ=1" >> $ROOT/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> $ROOT/.env
fi

if [ "$(uname)" != "Darwin" ] && [ -e "$ROOT/.git" ]; then
  echo "pre-commit hooks install..."
  pre-commit install
  git submodule foreach pre-commit install
fi
