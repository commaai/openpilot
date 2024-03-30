#!/usr/bin/env bash

set -e

if [ -z "$SKIP_PROMPT" ]; then
  echo "---------------   macOS support   ---------------"
  echo "Running openpilot natively on macOS is not officially supported."
  echo "It might build, some parts of it might work, but it's not fully tested, so there might be some issues."
  echo 
  echo "Check out devcontainers for a seamless experience (see tools/README.md)."
  echo "-------------------------------------------------"
  echo -n "Are you sure you want to continue? [y/N] "
  read -r response
  if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    exit 1
  fi
fi

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
brew "pyenv-virtualenv"
brew "qt@5"
brew "zeromq"
brew "gcc@13"
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

# install python dependencies
$DIR/install_python_dependencies.sh
echo "[ ] installed python dependencies t=$SECONDS"

# brew does not link qt5 by default
# check if qt5 can be linked, if not, prompt the user to link it
QT_BIN_LOCATION="$(command -v lupdate || :)"
if [ -n "$QT_BIN_LOCATION" ]; then
  # if qt6 is linked, prompt the user to unlink it and link the right version
  QT_BIN_VERSION="$(lupdate -version | awk '{print $NF}')"
  if [[ ! "$QT_BIN_VERSION" =~ 5\.[0-9]+\.[0-9]+ ]]; then
    echo
    echo "lupdate/lrelease available at PATH is $QT_BIN_VERSION"
    if [[ "$QT_BIN_LOCATION" == "$(brew --prefix)/"* ]]; then
      echo "Run the following command to link qt5:"
      echo "brew unlink qt@6 && brew link qt@5"
    else
      echo "Remove conflicting qt entries from PATH and run the following command to link qt5:"
      echo "brew link qt@5"
    fi
  fi
else
  brew link qt@5
fi

echo
echo "----   OPENPILOT SETUP DONE   ----"
echo "Open a new shell or configure your active shell env by running:"
echo "source $RC_FILE"
