if [ -z "$OPENPILOT_ENV" ]; then
  export PYTHONPATH="$HOME/openpilot"

  unamestr=`uname`
  if [[ "$unamestr" == 'Linux' ]]; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    export PATH="$PATH:$HOME/openpilot/external/capnp/bin"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/openpilot/external/capnp/lib"
  elif [[ "$unamestr" == 'Darwin' ]]; then
    # msgq doesn't work on mac
    export ZMQ=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  fi

  export OPENPILOT_ENV=1
fi

