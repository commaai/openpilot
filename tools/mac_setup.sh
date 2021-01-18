#!/bin/bash -e

# Install brew if required.
if [[ $(command -v brew) == "" ]]; then
  echo "Installing Hombrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
fi

brew install capnp@0.8.0 \
             coreutils@8.32 \
             eigen@3.3.9 \
             ffmpeg@4.3.1_4 \
             glfw@3.3.2 \
             libarchive@3.5.1_1 \
             libusb@1.0.24 \
             libtool@2.4.6_2 \
             llvm@11.0.0_1 \
             openssl@1.1 \
             pyenv@1.2.22 \
             qt@5.15.2 \
             zeromq@4.3.3_1

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

pyenv install -s 3.8.2
pyenv global 3.8.2
pyenv rehash
eval "$(pyenv init -)"

pip install pipenv==2020.8.13
pipenv install --system --deploy
