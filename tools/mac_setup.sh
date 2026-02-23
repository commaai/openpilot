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
brew "coreutils"
brew "eigen"
brew "llvm"
EOS

echo "[ ] finished brew install t=$SECONDS"

# install python dependencies
$DIR/install_python_dependencies.sh
echo "[ ] installed python dependencies t=$SECONDS"

echo
echo "----   OPENPILOT SETUP DONE   ----"
echo "Open a new shell or configure your active shell env by running:"
echo "source $RC_FILE"
