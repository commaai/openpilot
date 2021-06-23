#!/bin/bash -e

usage() {
  echo "Usage: $0 [openpilot install directory]"
  echo
  echo "Options:"
  echo " -h | --help - Shows basic about/usage"
  echo
  echo "If no install directory is provided, a best-guess is used."
}

# Validate syntax / parse args
for argval in "$@"; do
  if [ "$argval" == --help ] || [ "$argval" == -h ]; then
    usage
    exit 0
  fi
done
if [[ ! "$1" =~ (^\/.+$|^~.*$|^\.$|^$) ]]; then
  echo "[FAIL] Invalid syntax"
  echo
  usage
  exit 1
else
  if [ -z "$1" ]; then
    OP_HOME=$(git -C "$PWD" rev-parse --show-toplevel)
    if [[ "$OP_HOME" == fatal* ]]; then
      echo "[FAIL] OP_HOME: $OP_HOME"
      echo
      echo "Please first change to a directory within the openpilot repository "
      echo "or specify a folder when invoking $0."
      echo
      echo "Git folder result: $OP_HOME"
      echo
      exit 1
    fi
  else
    OP_HOME="$1"
  fi
fi
GIT_REMOTE=$(git -C "$OP_HOME" remote get-url origin)
GIT_ROOT=$(git -C "$OP_HOME" rev-parse --show-toplevel)

PLATFORM=$(uname)
check_platform() {
  if [[ $PLATFORM == Linux ]] || [[ $PLATFORM == Darwin ]]; then
    echo "[PASS] PLATFORM: $PLATFORM"
  else
    echo "[FAIL] PLATFORM: $PLATFORM"
    echo
    echo "Today, openpilot can only be set up on Mac & Linux. Tomorrow, with"
    echo "help of your generous contribution, we may support more platforms."
    echo 
    exit 1
  fi
}
check_platform

check_shell() {
  if [[ $SHELL == "/bin/zsh" ]]; then
    RC_FILE="$HOME/.zshrc"
  elif [[ $SHELL =~ .+\/bash$ ]]; then
    RC_FILE="$HOME/.bash_profile"
  else
    echo "[FAIL] SHELL: $SHELL"
    echo
    echo "Error: Shell in use ($SHELL) is not compatible with the openpilot "
    echo "environment (yet)"
    echo
    echo "If you wish to keep a different default shell than bash (not "
    echo "explicitly supported), you can invoke the bash shell by simply "
    echo "typing the command `bash` then rerunning this script."
    exit 1
  fi
  echo "[PASS] SHELL: $SHELL"
}

check_op_home() {
  if [[ ! "$GIT_REMOTE" =~ .+\/openpilot.git$ ]] || [[ ! "$GIT_ROOT" == "$OP_HOME" ]]; then
    echo "[FAIL] OP_HOME: $OP_HOME"
    echo
    echo "$OP_HOME should be where openpilot is cloned."
    echo
    echo "Expected: origin url should end in openpilot.git"
    echo "Actual: $GIT_REMOTE"
    echo
    echo "Expected: git top level directory (root) should equal $OP_HOME"
    echo "Actual: $GIT_ROOT"
    echo
    exit 1
  else
    echo "[PASS] OP_HOME: $OP_HOME"
  fi
}

check_shell
check_op_home

install_deps_apt() {
  APT_VERSION=$(apt-get --version | awk -F "." '/apt/ {print $1}')
  if [[ $APT_VERSION != "apt 2" ]]; then
    echo "[FAIL] apt-get version"
    echo
    echo "Today, this script requires apt-get version 2. Tomorrow, with the"
    echo "help of your generous contribution, we may support more package"
    echo "managers."
    echo
    exit 1
  else
    echo "[PASS] APT: $APT_VERSION"
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
      autoconf \
      build-essential \
      bzip2 \
      capnproto \
      cppcheck \
      libcapnp-dev \
      clang \
      cmake \
      curl \
      ffmpeg \
      git \
      libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev \
      libarchive-dev \
      libbz2-dev \
      libcurl4-openssl-dev \
      libeigen3-dev \
      libffi-dev \
      libglew-dev \
      libgles2-mesa-dev \
      libglfw3-dev \
      libglib2.0-0 \
      liblzma-dev \
      libomp-dev \
      libopencv-dev \
      libpng16-16 \
      libssl-dev \
      libstdc++-arm-none-eabi-newlib \
      libsqlite3-dev \
      libtool \
      libusb-1.0-0-dev \
      libzmq3-dev \
      libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
      libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev \
      libsystemd-dev \
      locales \
      ocl-icd-libopencl1 \
      ocl-icd-opencl-dev \
      opencl-headers \
      python-dev \
      python3-pip \
      qml-module-qtquick2 \
      qt5-default \
      qtmultimedia5-dev \
      qtwebengine5-dev \
      qtlocation5-dev \
      qtpositioning5-dev \
      libqt5sql5-sqlite \
      libqt5svg5-dev \
      screen \
      sudo \
      vim \
      wget \
      gcc-arm-none-eabi \
      libqt5x11extras5-dev \
      libreadline-dev
  fi
  # install git lfs
  if ! command -v "git-lfs" > /dev/null 2>&1; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs pull
  fi

  # install pyenv
  if ! command -v "pyenv" > /dev/null 2>&1; then
    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
  fi
}

install_deps_brew() {
  if [[ $(command -v brew) == "" ]]; then
    echo "Installing Hombrew"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
  fi

brew bundle --file=- <<-EOS
brew "cmake"
brew "zlib"
brew "bzip2"
brew "rust"
brew "rustup-init"
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
cask "gcc-arm-embedded"
EOS

  # Build requirements for macOS
  # https://github.com/pyenv/pyenv/issues/1740
  # https://github.com/pyca/cryptography/blob/main/docs/installation.rst
  rustup-init -y

  export LDFLAGS="$LDFLAGS -L/usr/local/opt/zlib/lib"
  export LDFLAGS="$LDFLAGS -L/usr/local/opt/bzip2/lib"
  export LDFLAGS="$LDFLAGS -L/usr/local/opt/openssl@1.1/lib"
  export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/zlib/include"
  export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/bzip2/include"
  export CPPFLAGS="$CPPFLAGS -I/usr/local/opt/openssl@1.1/include"
  export PATH="$PATH:/usr/local/opt/openssl@1.1/bin"
  export PATH="$PATH:/usr/local/bin"

}

if [[ $PLATFORM == 'Linux' ]]; then
  install_deps_apt
elif [[ $PLATFORM == 'Darwin' ]]; then
  install_deps_brew
fi

if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
  if [[ $PLATFORM == 'Darwin' ]]; then
    echo "export PATH=\"\$PATH:$HOME/.cargo/bin\"" >> $RC_FILE
    export PATH="$PATH:\"\$HOME/.cargo/bin\""
  fi
  echo "source $OP_HOME/tools/openpilot_env.sh" >> $RC_FILE
  source $RC_FILE
  echo "Added openpilot_env to RC file: $RC_FILE"
else
  echo "[WARN] Openpilot not added to $RC_FILE"
fi

pyenv install -s 3.8.5
pyenv global 3.8.5
pyenv rehash
eval "$(pyenv init -)"
pip install --upgrade pip==20.2.4
pip install pipenv==2020.8.13
pipenv install --system --deploy
git -C $OP_HOME submodule update --init
pre-commit install

if [[ `git branch --show-current` == "release2" ]]; then
  echo
  echo '[WARN] Your current branch is "release2". You should not do any'
  echo '       development work on this branch.'
fi

echo
echo
echo "----   FINISH OPENPILOT SETUP   ----"
echo
echo "Congratulations, your dev environment has been set up successfully!"
echo 
echo "Useful Commands:"
echo "  Here are some useful commands you may want to"
echo "  consider adding as aliases to your $RC_FILE:"
echo
echo '    alias op-pch="pre-commit run --all"'
echo '    alias op-init="git submodule update --init"'
echo '    alias op-build="cd $OP_HOME && scons -j$(nproc)"'
echo '    alias op-mypy="pre-commit run mypy --all""'
echo
echo "Additional Notes:"
echo "  Please note that some tools, such as [PlotJuggler](https://github.com/commaai/openpilot/tree/master/tools/plotjuggler)"
echo "  require additional installations and that information can be found in"
echo "  their respective repositories."
echo 
echo "Final Steps"
echo "  Activate environment via: \`source $RC_FILE\`"
echo "  Build openpilot: \`cd $OP_HOME && scons -j$(nproc)\`"

