# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 24.04** with [modern hardware](https://github.com/commaai/openpilot/wiki/Requirements#hardware), which is the primary development target aside from the [supported embedded hardware](https://comma.ai/shop/comma-3x).

Most of openpilot should work natively on macOS. On Windows you can use WSL2 for a nearly native Ubuntu experience [with some exceptions](https://github.com/commaai/openpilot/wiki/WSL). Running natively on any other system is not currently recommended and will likely require modifications.

## Native setup on Ubuntu 24.04 and macOS

**1. Clone openpilot**

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
tools/op.sh setup
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

## CTF
Learn about the openpilot ecosystem and tools by playing our [CTF](/tools/CTF.md) and [reading the wiki](https://github.com/commaai/openpilot/wiki/Introduction-to-openpilot).

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
