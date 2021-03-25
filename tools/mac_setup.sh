#!/bin/bash -e

# Install brew if required
if [[ $(command -v brew) == "" ]]; then
  echo "Installing Hombrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
fi

brew bundle --file=- <<-EOS
brew "capnp"
brew "coreutils"
brew "eigen"
brew "ffmpeg"
brew "glfw"
brew "libarchive"
brew "libusb"
brew "libtool"
brew "llvm"
brew "openssl"
brew "pyenv"
brew "qt@5"
brew "zeromq"
cask "gcc-arm-embedded"
EOS

if [[ $SHELL == "/bin/zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == "/bin/bash" ]]; then
  RC_FILE="$HOME/.bash_profile"
fi

if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
  OP_DIR=$(git rev-parse --show-toplevel)
  echo "source $OP_DIR/tools/openpilot_env.sh" >> $RC_FILE
  source $RC_FILE
  echo "Added openpilot_env to RC file: $RC_FILE"
fi

pyenv install -s 3.8.5
pyenv global 3.8.5
pyenv rehash
eval "$(pyenv init -)"

pip install pipenv==2020.8.13
pipenv install --system --deploy
