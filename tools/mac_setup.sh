#!/bin/bash -e
PYTHON_VERSION=3.8.10
PYTHON_VER=3.8
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd $DIR/../ && pwd)"

# Install brew if required
if [[ $(command -v brew) == "" ]]; then
  echo "Installing Hombrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
fi

brew bundle --file=- <<-EOS
brew "cmake"
brew "git-lfs"
brew "zlib"
brew "bzip2"
brew "unzip"
brew "wget"
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
brew "protobuf"
brew "protobuf-c"
cask "gcc-arm-embedded"
EOS

if [[ $SHELL == "/bin/zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == "/bin/bash" ]]; then
  RC_FILE="$HOME/.bash_profile"
fi

export LDFLAGS="$LDFLAGS -L/usr/local/opt/zlib/lib"
export LDFLAGS="$LDFLAGS -L/usr/local/opt/bzip2/lib"
export LDFLAGS="$LDFLAGS -L/usr/local/opt/openssl@1.1/lib"
export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/zlib/include"
export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/bzip2/include"
export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/openssl@1.1/include"
export PATH="$PATH:/usr/local/opt/openssl@1.1/bin"
export PATH="$PATH:/usr/local/bin"

# openpilot environment
if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
  echo "source $ROOT/tools/openpilot_env.sh" >> $RC_FILE
  source "$ROOT/tools/openpilot_env.sh"
  echo "Added openpilot_env to RC file: $RC_FILE"
fi

# install python dependencies
$ROOT/update_requirements.sh || true

# install casadi
echo "-- casadi manual install"
VENV=`pipenv --venv`
cd /tmp/ && mkdir -p casadi
wget https://github.com/casadi/casadi/releases/download/3.5.5/casadi3.5.5_source.zip
unzip -qq casadi3.5.5_source.zip -d casadi
cd casadi && mkdir -p build && cd build
cmake .. \
  -DWITH_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=$VENV \
  -DPYTHON_PREFIX:PATH=$VENV/lib/python$PYTHON_VER/site-packages \
  -DPYTHON_LIBRARY:FILEPATH=$HOME/.pyenv/versions/$PYTHON_VERSION/lib/libpython3.8.dylib \
  -DPYTHON_EXECUTABLE:FILEPATH=$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python \
  -DPYTHON_INCLUDE_DIR:PATH=$HOME/.pyenv/versions/$PYTHON_VERSION/include/python3.8
make -j$(nproc) && make install
cd $ROOT

echo
echo "----   FINISH OPENPILOT SETUP   ----"
echo "Configure your active shell env by running:"
echo "source $RC_FILE"
