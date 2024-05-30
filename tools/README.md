# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 20.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Running natively on any other system is not recommended and will require modifications. On Windows you can use WSL, and on macOS or incompatible Linux systems, it is recommended to use the dev containers.

## Native setup on Ubuntu 20.04

**1. Clone openpilot**

NOTE: This repository uses Git LFS for large files. Ensure you have [Git LFS](https://git-lfs.com/) installed and set up before cloning or working with it.

Either do a partial clone for faster download:
``` bash
git clone --filter=blob:none --recurse-submodules --also-filter-submodules https://github.com/commaai/openpilot.git
```

or do a full clone:
``` bash
git clone --recurse-submodules https://github.com/commaai/openpilot.git
```

Now you can use the automatic script or manually setup your environment:

### Automatic setup

**2. Run the setup script**

``` bash
cd openpilot
git lfs pull
tools/ubuntu_setup.sh
```

Activate a shell with the Python dependencies installed:
``` bash
poetry shell
```

**3. Build openpilot**

``` bash
scons -u -j$(nproc)
```

### Manual setup

**2. Setup pyenv**

Run the pyenv installer:

```bash
curl https://pyenv.run | bash
```

Then follow [the instructions from pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#set-up-your-shell-environment-for-pyenv) to setup the shell environment.

After you're done, restart your shell:

```bash
exec "$SHELL"
```

**3. Install runtime dependencies**

Install the common requirements:

```bash
sudo apt install ca-certificates clang cppcheck build-essential gcc-arm-none-eabi liblzma-dev capnproto libcapnp-dev curl libcurl4-openssl-dev git git-lfs ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libbz2-dev libeigen3-dev libffi-dev libglew-dev libgles2-mesa-dev libglfw3-dev libglib2.0-0 libqt5charts5-dev libncurses5-dev libssl-dev libusb-1.0-0-dev libzmq3-dev libsqlite3-dev libsystemd-dev locales opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev portaudio19-dev qtmultimedia5-dev qtlocation5-dev qtpositioning5-dev qttools5-dev-tools libqt5svg5-dev libqt5serialbus5-dev libqt5x11extras5-dev libqt5opengl5-dev
```

After that, if you're using Ubuntu 24.04 LTS:

```bash
sudo apt install g++-12 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools python3-dev
```

If you're using Ubuntu 20.04 ( Focal Fossa ):

```bash
sudo apt install libavresample-dev qt5-default python-dev
```

**4. Install development dependencies ( optional )**

```bash
sudo apt install casync cmake make clinfo libqt5sql5-sqlite libreadline-dev libdw1 autoconf libtool bzip2 libarchive-dev libncursesw5-dev libportaudio2 locales
```

**5. Install development tools ( optional )**

```bash
sudo apt install valgrind
```

**6. Install python through pyenv**

Update pyenv and install python:
```bash
pyenv update
CONFIGURE_OPTS="--enable-shared" pyenv install -f 3.11.4
```

**7. Setup poetry**

Update pip and install poetry:

```bash
pip install pip==24.0
pip install poetry==1.7.0
```

Configure poetry virtualenvs and install dotenv plugin:
```bash
poetry config virtualenvs.prefer-active-python true --local
poetry config virtualenvs.in-project true --local

poetry self add poetry-dotenv-plugin@^0.1.0
```

**8. Install the python dependencies**

```bash
poetry install --no-cache --no-root
pyenv rehash
```

After that, configure your active shell env and activate poetry:

```bash
source ~/.bashrc
poetry shell
```

**9. Build openpilot**

``` bash
scons -u -j$(nproc)
```


## Dev Container on any Linux or macOS

openpilot supports [Dev Containers](https://containers.dev/). Dev containers provide customizable and consistent development environment wrapped inside a container. This means you can develop in a designated environment matching our primary development target, regardless of your local setup.

Dev containers are supported in [multiple editors and IDEs](https://containers.dev/supporting), including Visual Studio Code. Use the following [guide](https://code.visualstudio.com/docs/devcontainers/containers) to start using them with VSCode.

#### X11 forwarding on macOS

GUI apps like `ui` or `cabana` can also run inside the container by leveraging X11 forwarding. To make use of it on macOS, additional configuration steps must be taken. Follow [these](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088) steps to setup X11 forwarding on macOS.

## WSL on Windows

[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-20.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

**NOTE**: If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after following the steps above, you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.

## CTF
Learn about the openpilot ecosystem and tools by playing our [CTF](/tools/CTF.md).

## Directory Structure

```
├── ubuntu_setup.sh     # Setup script for Ubuntu
├── mac_setup.sh        # Setup script for macOS
├── cabana/             # View and plot CAN messages from drives or in realtime
├── camerastream/       # Cameras stream over the network
├── joystick/           # Control your car with a joystick
├── lib/                # Libraries to support the tools and reading openpilot logs
├── plotjuggler/        # A tool to plot openpilot logs
├── replay/             # Replay drives and mock openpilot services
├── scripts/            # Miscellaneous scripts
├── serial/             # Tools for using the comma serial
├── sim/                # Run openpilot in a simulator
├── ssh/                # SSH into a comma device
└── webcam/             # Run openpilot on a PC with webcams
```
