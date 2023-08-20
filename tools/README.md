# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 20.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car). We also have a CI test to verify that openpilot builds on macOS, but the tools are untested. For the best experience, stick to Ubuntu 20.04, otherwise openpilot and the tools should work with minimal to no modifications on macOS and other Linux systems.

## Setup your PC

First, clone openpilot:
``` bash
cd ~
git clone --recurse-submodules https://github.com/commaai/openpilot.git

# or do a partial clone instead for a faster clone and smaller repo size
git clone --filter=blob:none --recurse-submodules --also-filter-submodules https://github.com/commaai/openpilot.git

cd openpilot
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
