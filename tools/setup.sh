#!/usr/bin/bash -e

function detect_shell_config() {
    local BASH_PROFILE="$HOME/.bash_profile"
    local BASHRC="$HOME/.bashrc"
    local ZSHRC="$HOME/.zshrc"
    if [[ $SHELL == "/bin/zsh" ]]; then
        RC_FILE="$ZSHRC"
    elif [[ $SHELL == "/bin/bash" ]]; then
        if [ -a "$BASH_PROFILE" ]; then
            RC_FILE="$BASH_PROFILE"
        else
            RC_FILE="$BASHRC"
        fi
    else
        return 1
    fi
}

function command_exists() {
    if ! command -v "$1" > /dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

function install_macos_requirements() {
    # Install brew if required
    if ! command_exists "brew"; then
        echo "Installing Hombrew"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    fi
    
    brew bundle --file=- <<-EOS
brew "git-lfs"
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
    
    # Add Rust to PATH
    if ! command_exists "cargo" && [ -a "$RC_FILE" ] && [ -z "$CI" ]; then
        echo "export PATH=\"\$PATH:$HOME/.cargo/bin\"" >> $RC_FILE
        export PATH="$PATH:$HOME/.cargo/bin"
    fi
}

function install_ubuntu_latest_requirements() {
    sudo apt-get update -qq && sudo apt-get install -qq -y --no-install-recommends \
        autoconf \
        build-essential \
        bzip2 \
        capnproto \
        cppcheck \
        libcapnp-dev \
        clang \
        cmake \
        make \
        curl \
        ffmpeg \
        git \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev \
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
        libsdl1.2-dev  libportmidi-dev libavformat-dev libavcodec-dev libfreetype6-dev \
        libsystemd-dev \
        locales \
        ocl-icd-libopencl1 \
        ocl-icd-opencl-dev \
        opencl-headers \
        python-dev \
        python3-pip \
        qml-module-qtquick2 \
        qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
        qtmultimedia5-dev \
        qtwebengine5-dev \
        qtlocation5-dev \
        qtpositioning5-dev \
        libqt5sql5-sqlite \
        libqt5svg5-dev \
        wget \
        gcc-arm-none-eabi \
        libqt5x11extras5-dev \
        libreadline-dev

    # install git lfs
    if ! command_exists "git-lfs"; then
        # Note: the line below should be uncommented when there's a build available for the latest version of Ubuntu
        # curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs -qq
    fi

    # install pyenv
    if ! command_exists "pyenv" && [ -z "$NO_PYENV" ] && [ ! -d "$HOME/.pyenv" ]; then
        curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
    fi
}

function install_ubuntu_lts_requirements() {
    sudo apt-get update -qq && sudo apt-get install -qq -y --no-install-recommends \
        autoconf \
        build-essential \
        bzip2 \
        capnproto \
        cppcheck \
        libcapnp-dev \
        clang \
        cmake \
        make \
        curl \
        ffmpeg \
        git \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavresample-dev libavfilter-dev \
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
        libsdl1.2-dev  libportmidi-dev libavformat-dev libavcodec-dev libfreetype6-dev \
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
        wget \
        gcc-arm-none-eabi \
        libqt5x11extras5-dev \
        libreadline-dev


    # install git lfs
    if ! command_exists "git-lfs"; then
      curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
      sudo apt-get install git-lfs -qq
    fi

    # install pyenv
    if ! command_exists "pyenv" && [ -z "$NO_PYENV" ] && [ ! -d "$HOME/.pyenv" ]; then
      curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
    fi

}

function setup_pyenv() {
    if ! command_exists "pyenv"; then
        echo "setup_pyenv: pyenv not found"
        exit 1
    elif ! pyenv prefix ${PYENV_PYTHON_VERSION} &> /dev/null; then
        export MAKEFLAGS="-j$(nproc)"
        eval "$(pyenv init -)"
        PYENV_PYTHON_VERSION=$(cat $OP_ROOT/.python-version)
        CONFIGURE_OPTS=--enable-shared pyenv install -f ${PYENV_PYTHON_VERSION}
        pyenv global ${PYENV_PYTHON_VERSION}
        pyenv rehash
    fi
}

function setup_packages() {
    echo "Upgrading pip"
    pip install --upgrade pip --quiet
    echo "Installing pipenv"
    pip install pipenv --quiet
    [ -d "./xx" ] && export PIPENV_PIPFILE=./xx/Pipfile
    echo "Installing packages"
    pipenv install --dev --deploy --system
    if command_exists "pyenv"; then
        pyenv rehash
    fi
    
    echo "Installing pre-commit"
    pre-commit install
    # For internal comma repos
    [ -d "./xx" ] && (cd xx && pre-commit install)
    [ -d "./notebooks" ] && (cd notebooks && pre-commit install)
}

function setup_help() {
    echo "Setup script for openpilot environment"
    echo "By default it does the following things:"
    echo -e "\t1. Download required libraries and tools for Ubuntu 20.04 LTS."
    echo -e "\t2. Save openpilot environment script in shell configuration."
    echo -e "\t3. Setup pyenv to build and use configured Python version."
    echo -e "\t4. Download packages from Pipfile using pipenv."
    echo 
    echo "Syntax: $0 [OPTIONS...]"
    echo "Options:"
    echo -e "\t[-h|--help]:     Print help message."
    echo -e "\t--os <OS>:       Operating system (ubuntu-lts by default)."
    echo -e "\t--skip-reqs:     Do not download dependencies."
    echo -e "\t--skip-rc:       Do not setup shell evironment."
    echo -e "\t--no-pyenv:      Skip pyenv setup. Will use default Python if pyenv is not installed."
    echo -e "\t--skip-packages: Do not install pipenv and its requirements from Pipfile."
    echo -e "Supported OSes: ubuntu-lts, ubuntu-latest, macos (not tested)"
}

SELECTED_OS="ubuntu-lts"

# Parse arguments
while [ ! -z "$1" ]; do
    case "$1" in
        -h|--help)
            setup_help 
            exit 0
            ;;
        --os)
            shift
            SELECTED_OS="$1"
            
            if [ -z "$SELECTED_OS" ]; then
                echo "Selected OS is empty."
                exit 1
            fi

            shift
            ;;
        --skip-reqs)
            SKIP_REQS=1
            shift
            ;;
        --no-pyenv)
            NO_PYENV=1
            shift
            ;;
        --skip-rc)
            SKIP_RC=1
            shift
            ;;
        --skip-packages)
            SKIP_PACKAGES=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
    esac 
done


# Try to detect shell configuration file if it was not specified
if [ -z "$RC_FILE" ]; then
    detect_shell_config 
fi

if [ -n "$RC_FILE" ]; then
    echo "Shell config detected: $RC_FILE"
fi

if [ -z "$SKIP_REQS" ]; then
    echo "Selected OS: $SELECTED_OS"
    echo "Installing required system dependencies"
    case $SELECTED_OS in
        ubuntu-lts)
            install_ubuntu_lts_requirements
            ;;
        ubuntu-latest)
            install_ubuntu_latest_requirements
            ;;
        macos)
            install_macos_requirements
            ;;
        *)
            echo "OS not supported. Install dependencies manually or skip this step with --skip-reqs."
            exit 1
            ;;
    esac
else
    echo "Skipped downloading system dependencies." 
fi

OP_ROOT=$(git rev-parse --show-toplevel)
cd $OP_ROOT

if [ -z "$SKIP_RC" ]; then
    OP_ENV_SCRIPT="$OP_ROOT/tools/openpilot_env.sh"

    # If the env script is not present, add it to shell
    if [ -z "$OPENPILOT_ENV" ] && [ -f "$RC_FILE" ]; then
        echo -e "\n# openpilot environment" >> $RC_FILE
        echo "export OP_ROOT=$OP_ROOT" >> $RC_FILE
        echo "source $OP_ENV_SCRIPT" >> $RC_FILE
        echo "Added openpilot environment script to shell config: $RC_FILE"
    elif [ -z "$OPENPILOT_ENV" ] && [ ! -f "$RC_FILE" ]; then
        echo "No valid shell configuration file provided."
        echo "Cannot install openpilot environment script."
        echo "Add the following code to your shell manually:"
        echo -e "\n# openpilot environment"
        echo "export OP_ROOT=$OP_ROOT"
        echo "source $OP_ENV_SCRIPT"
    else
        echo "openpilot environment script already installed"
    fi

    source $OP_ENV_SCRIPT

    if [ -z "$OPENPILOT_ENV" ]; then
        echo "Environment not set correctly."
        exit 1
    fi
else
    echo "Skipped installation of openpilot environment script."
fi


echo "Doing the rest of git checkout."
git lfs pull
git submodule init
git submodule update

if [ -z "$NO_PYENV" ]; then
    echo "Setting up pyenv."
    setup_pyenv
else
    echo "Skipped pyenv setup."
fi

if [ -z "$SKIP_PACKAGES" ]; then
    echo "Installing packages."
    setup_packages
else
    echo "Skipping installation of packages."
fi



echo -e "\n---- OPENPILOT ENVIRONMENT SETUP FINISHED ----"
echo "Your environment is set up!"
if [ -f "$RC_FILE " ]; then
    echo "Use 'source $RC_FILE to reload your shell"
fi
echo "Go to $OP_ROOT and compile OP with 'scons -j\$(nproc)'."
echo "Then try executing these commands in separate windows:"
echo -e "\t* $OP_ROOT/selfdrive/ui/replay/replay --demo"
echo -e "\t* $OP_ROOT/selfdrive/ui/ui"
