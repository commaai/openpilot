#!/bin/bash -e

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

brew install scons capnp czmq coreutils pyenv

pyenv install 3.8.2
pyenv global 3.8.2
pyenv rehash
pipenv install --system --deploy

