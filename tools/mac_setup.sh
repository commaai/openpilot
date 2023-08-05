#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd $DIR/../ && pwd)"
ARCH=$(uname -m)

if [[ $SHELL == "/bin/zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == "/bin/bash" ]]; then
  RC_FILE="$HOME/.bash_profile"
fi

# Install brew if required
if [[ $(command -v brew) == "" ]]; then
  echo "Installing Hombrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
  echo "[ ] installed brew t=$SECONDS"

  # make brew available now
  if [[ $ARCH == "x86_64" ]]; then
      echo 'eval "$(/usr/local/homebrew/bin/brew shellenv)"' >> $RC_FILE
      eval "$(/usr/local/homebrew/bin/brew shellenv)"
  else
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $RC_FILE
      eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
fi

# TODO: remove protobuf,protobuf-c,swig when casadi can be pip installed
brew bundle --file=- <<-EOS
brew "catch2"
brew "cmake"
brew "cppcheck"
brew "git-lfs"
brew "zlib"
brew "bzip2"
brew "capnp"
brew "coreutils"
brew "eigen"
brew "ffmpeg"
brew "glfw"
brew "libarchive"
brew "libusb"
brew "libtool"
brew "llvm"
brew "openssl@3.0"
brew "pyenv"
brew "qt@5"
brew "zeromq"
brew "protobuf"
brew "protobuf-c"
brew "swig"
brew "gcc@12"
cask "gcc-arm-embedded"
brew "portaudio"
EOS

echo "[ ] finished brew install t=$SECONDS"

BREW_PREFIX=$(brew --prefix)

# archive backend tools for pip dependencies
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/zlib/lib"
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/bzip2/lib"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/zlib/include"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/bzip2/include"

# pycurl curl/openssl backend dependencies
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/openssl@3/lib"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/openssl@3/include"
export PYCURL_CURL_CONFIG=/usr/bin/curl-config
export PYCURL_SSL_LIBRARY=openssl

# openpilot environment
if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
  echo "source $ROOT/tools/openpilot_env.sh" >> $RC_FILE
  source "$ROOT/tools/openpilot_env.sh"
  echo "Added openpilot_env to RC file: $RC_FILE"
fi

# install python dependencies
$ROOT/update_requirements.sh
eval "$(pyenv init --path)"
echo "[ ] installed python dependencies t=$SECONDS"

echo
echo "----   OPENPILOT SETUP DONE   ----"
echo "Open a new shell or configure your active shell env by running:"
echo "source $RC_FILE"
