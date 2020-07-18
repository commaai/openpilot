#!/bin/bash -e

# install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

brew install capnp \
             czmq \
             coreutils \
             eigen \
             llvm \
             libarchive \
             pyenv \
             sdl2_gfx \
             sdl2_image \
             sdl2_mixer \
             sdl2_ttf

# install python
pyenv install 3.8.2
pyenv global 3.8.2
pyenv rehash
eval "$(pyenv init -)"

pip install --no-cache-dir pipenv==2018.11.26
pipenv install --system --deploy --clear

