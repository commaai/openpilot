#!/bin/bash -e

echo "Updating Homebrew"
brew update

brew install capnp \
             czmq \
             coreutils \
             eigen \
             ffmpeg \
             glfw \
             libarchive \
             libusb \
             libtool \
             llvm \
             pyenv \
             qt5 \
             zeromq

# Detect shell and pick correct RC file.
if [[ $SHELL == "/bin/zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == "/bin/bash" ]]; then
  RC_FILE="$HOME/.bash_profile"
else
  echo "-------------------------------------------------------------"
  echo "Unsupported shell: \"$SHELL\", cannot install to RC file."
  echo "Please run: echo \"source $OP_DIR/tools/openpilot_env.sh\" >> %YOUR SHELL's RC file%"
  echo "-------------------------------------------------------------"
fi

# Install to RC file (only non-CI).
if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
  OP_DIR=$(git rev-parse --show-toplevel)
  echo "source $OP_DIR/tools/openpilot_env.sh" >> $RC_FILE
  source $RC_FILE
  echo "Added openpilot_env to RC file: $RC_FILE"
else
  echo "Skipped RC file installation"
fi

# Install python.
pyenv install -s 3.8.2
pyenv global 3.8.2
pyenv rehash
eval "$(pyenv init -)" # CI doesn't use .bash_profile, and will use python2.7 if this line isn't here.

pip install pipenv==2020.8.13
pipenv install --system --deploy
