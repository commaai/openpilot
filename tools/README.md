# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 24.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Running natively on any other system is not recommended and will require modifications. On Windows you can use WSL, and on macOS or incompatible Linux systems, it is recommended to use the dev containers.

## Native setup on Ubuntu 24.04

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

**2. Run the setup script**

``` bash
cd openpilot
tools/ubuntu_setup.sh
```

**3. Git LFS**

``` bash
git lfs pull
```

**4. Activate a python shell**

Activate a shell with the Python dependencies installed:
``` bash
source .venv/bin/activate
```

**5. Build openpilot**

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

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-24.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

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
└── webcam/             # Run openpilot on a PC with webcams
```
