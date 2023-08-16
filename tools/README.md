# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 20.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car). We also have a CI test to verify that openpilot builds on macOS, but the tools are untested. For the best experience, stick to Ubuntu 20.04, otherwise openpilot and the tools should work with minimal to no modifications on macOS and other Linux systems.

## Setup your PC

First, clone openpilot:
``` bash
cd ~
git clone https://github.com/commaai/openpilot.git

cd openpilot
git submodule update --init
```

Then, run the setup script:

``` bash
# for Ubuntu 20.04 LTS
tools/ubuntu_setup.sh

# for macOS
tools/mac_setup.sh
```

Activate a shell with the Python dependencies installed:

``` bash
cd openpilot && poetry shell
```

Build openpilot with this command:
``` bash
scons -u -j$(nproc)
```

### Dev Container

openpilot supports [Dev Containers](https://containers.dev/). Dev containers provide customizable and consistent development environment wrapped inside a container. This means you can develop in a designated environment matching our primary development target, regardless of your local setup.

Dev containers are supported in [multiple editors and IDEs](https://containers.dev/supporting), including [Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers).

#### X11 forwarding on macOS

GUI apps like `ui` or `cabana` can also run inside the container by leveraging X11 forwarding. To make use of it on macOS, additional configuration steps must be taken. First of all, install [XQuartz](https://formulae.brew.sh/cask/xquartz#default):
```sh
brew install --cask xquartz
```

Open XQuartz, go to `XQuartz` > `Settings...` > `Security` and check "Authenticate connections" and "Allow connections from network clients" boxes. Then allow for connections from localhost with:
```sh
xhost +localhost
```
Note that this is temporary and only affects current XQuartz session. To make it permanent, add the above line to shell rc file.

`.Xauthority` file is used for authentication with X11 server. It's typically stored in user's `$HOME` directory. Openpilot's dev container is using `XAUTHORITY` env var to locate such file in host file system. If it's not set already, run following lines to set it up (and again, to make it permanent, add it to rc file):

```sh
XAUTH_FILE_ROW=($(xauth info | head -n 1))
export XAUTHORITY="${XAUTH_FILE_ROW[@]:2}"
```

### Windows

Neither openpilot nor any of the tools are developed or tested on Windows, but the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-20.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

**NOTE**: If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after following the steps above, you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.

## CTF
Learn about the openpilot ecosystem and tools by playing our [CTF](/tools/CTF.md).

## Directory Structure

```
├── ubuntu_setup.sh     # Setup script for Ubuntu
├── mac_setup.sh        # Setup script for macOS
├── cabana/             # View and plot CAN messages from drives or in realtime
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
