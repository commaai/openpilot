export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export PYTHONPATH="$HOME/openpilot"
export PATH="$PATH:$HOME/openpilot/external/capnp/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/openpilot/external/capnp/lib"

export OPENPILOT_ENV=1
