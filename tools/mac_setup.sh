#!/bin/bash -e

# Install brew if required.
if [[ $(command -v brew) == "" ]]; then
  echo "Installing Hombrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
fi

# (du -shc /usr/local/Homebrew/Library/* | sort -h) || true
# (du -shc /usr/local/Cellar/* | sort -h) || true
# (du -shc /usr/local/Caskroom/* | sort -h) || true
(du -shc ~/Library/Caches/Homebrew/downloads/* | sort -h) || true

before=$(ls ~/Library/Caches/Homebrew/downloads)

# rm -rf /usr/local/Homebrew/*
# find /usr/local/Homebrew \! -regex ".+\.git.+" -delete
# rm -rf /usr/local/Cellar/*
# rm -rf /usr/local/Caskroom/*
# rm -rf ~/Library/Caches/Homebrew/*

if [[ $(command -v ffmpeg) != "" ]]; then
  echo "Cache worked!"
fi

brew install zeromq
#  capnp \
            #  coreutils \
            #  eigen \
            #  ffmpeg \
            #  glfw \
            #  libarchive \
            #  libusb \
            #  libtool \
            #  llvm \
            #  openssl \
            #  pyenv \
            #  qt5 \
            #  zeromq

# (du -shc /usr/local/Homebrew/Library/* | sort -h) || true
# (du -shc /usr/local/Cellar/* | sort -h) || true
# (du -shc /usr/local/Caskroom/* | sort -h) || true
(du -shc ~/Library/Caches/Homebrew/downloads/* | sort -h) || true
after=$(ls ~/Library/Caches/Homebrew/downloads)

echo "NEW FILES"
new=$(comm -13 <(echo $before) <(echo $after))
echo $new
diff <(echo $before) <(echo $after)

if [[ $(command -v ffmpeg) != "" ]]; then
  echo "ffmpeg exists"
fi

# if [[ $SHELL == "/bin/zsh" ]]; then
#   RC_FILE="$HOME/.zshrc"
# elif [[ $SHELL == "/bin/bash" ]]; then
#   RC_FILE="$HOME/.bash_profile"
# fi

# if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
#   OP_DIR=$(git rev-parse --show-toplevel)
#   echo "source $OP_DIR/tools/openpilot_env.sh" >> $RC_FILE
#   source $RC_FILE
#   echo "Added openpilot_env to RC file: $RC_FILE"
# fi

# pyenv install -s 3.8.2
# pyenv global 3.8.2
# pyenv rehash
# eval "$(pyenv init -)"

# pip install pipenv==2020.8.13
# pipenv install --system --deploy
