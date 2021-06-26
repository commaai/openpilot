if [ -z "$OPENPILOT_ENV" ]; then
  export PYTHONPATH="$HOME/openpilot"

  unamestr=`uname`
  if [[ "$unamestr" == 'Linux' ]]; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv virtualenv-init -)"
  elif [[ "$unamestr" == 'Darwin' ]]; then
    # msgq doesn't work on mac
    export ZMQ=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  fi
  eval "$(pyenv init -)"

  export OPENPILOT_ENV=1
fi
