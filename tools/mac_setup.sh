#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd $DIR/../ && pwd)"

# homebrew update is slow
export HOMEBREW_NO_AUTO_UPDATE=1

if [[ $SHELL == "/bin/zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == "/bin/bash" ]]; then
  RC_FILE="$HOME/.bash_profile"
fi

# Install brew if required
if [[ $(command -v brew) == "" ]]; then
  echo "Installing Homebrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  echo "[ ] installed brew t=$SECONDS"

  # make brew available now
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $RC_FILE
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
    brew up
fi

brew bundle --file=- <<-EOS
brew "git-lfs"
brew "capnp"
brew "coreutils"
brew "eigen"
brew "ffmpeg"
brew "glfw"
brew "libusb"
brew "libtool"
brew "llvm"
brew "qt@5"
brew "zeromq"
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
