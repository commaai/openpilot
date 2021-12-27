OP_ROOT=$(git rev-parse --show-toplevel)
if [ -z "$OPENPILOT_ENV" ]; then
  export PYTHONPATH="$OP_ROOT:$PYTHONPATH"
  export PATH="$HOME/.pyenv/bin:$PATH"

  # Pyenv suggests we place the below two lines in .profile before we source
  # .bashrc, but there is no simple way to guarantee we do this correctly
  # programmatically across heterogeneous systems. For end-user convenience,
  # we add the lines here as a workaround.
  # https://github.com/pyenv/pyenv/issues/1906
  export PYENV_ROOT="$HOME/.pyenv"

  unamestr=`uname`
  if [[ "$unamestr" == 'Linux' ]]; then
    eval "$(pyenv init --path)"

    eval "$(pyenv virtualenv-init -)"
  elif [[ "$unamestr" == 'Darwin' ]]; then
    # msgq doesn't work on mac
    export ZMQ=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  fi
  eval "$(pyenv init -)"

  export OPENPILOT_ENV=1
fi
