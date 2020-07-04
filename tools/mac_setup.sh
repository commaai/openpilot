#!/bin/bash -e

# install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

brew install capnp czmq coreutils pyenv

# install python
pyenv install 3.8.2
pyenv global 3.8.2
pyenv rehash

pip install --no-cache-dir pipenv==2018.11.26
pipenv install --system --deploy

