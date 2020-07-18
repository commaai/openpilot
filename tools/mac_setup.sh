#!/bin/bash -e

# install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

brew install capnp \
             czmq \
             coreutils \
             eigen \
             ffmpeg \
             glfw \
             libarchive \
             libtool \
             llvm \
             pyenv \
             zeromq

# for use in CI after restoring cache
if [ ! -z "$BREW_LINK" ]; then
  brew link $(brew list) || true
fi

# install python
pyenv install -s 3.8.2
pyenv global 3.8.2
pyenv rehash
eval "$(pyenv init -)"

pip install --no-cache-dir pipenv==2018.11.26
pipenv install --system --deploy --clear

