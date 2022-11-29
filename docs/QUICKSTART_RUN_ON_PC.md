# Quick Start: How to Run on PC

This document explains how to run openpilot, run the UI, and run demo video with logged messages using Replay. To run it with the CARLA simulator, visit the [CARLA Documentation](https://github.com/commaai/openpilot/blob/master/tools/sim/README.md) and the [CARLA wiki](https://github.com/commaai/openpilot/wiki/CARLA). This is a quick guide to get started. It takes key information from other documents and is summarized here. We encourage you to explore the documents found in our repository. 

## Requirements
Openpilot may not work with your operating system. 

Window users need to install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and then install [Ubuntu 20.4](https://apps.microsoft.com/store/detail/ubuntu-20045-lts/9MTTCL66CPXJ) from the Microsoft store. 

Note that GUI applications do not work with WSL. You will then need to either upgrade to Windows 11 and follow the [WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) steps, or you can set up an [Xorg server](https://techcommunity.microsoft.com/t5/windows-dev-appconsult/running-wsl-gui-apps-on-windows-10/ba-p/1493242)

We use Ubuntu 20.4 and MacOS(10.15). Other versions may not work

You also need a GPU with openCL support. Nvidia is used for development and has worked.


## Setup your PC and Run openpilot

You may encounter errors when entering these commands. There are resources online that can help solve these issues. Unfortunately we cannot help all of you. Remember Google is your friend. 

Clone openpilot:
``` bash
cd ~
git clone https://github.com/commaai/openpilot.git

cd openpilot
git submodule update --init
```

Run the setup script: 
``` bash
# for Ubuntu 20.04 LTS
cd openpilot
tools/ubuntu_setup.sh

# for macOS
cd openpilot
tools/mac_setup.sh
```

Run the update script:
``` bash
cd openpilot
./update_requirements.sh
```

Activate a shell with the Python dependencies installed:

``` bash
cd ~
cd openpilot && poetry shell
```

Build openpilot with this command:
``` bash
cd openpilot
scons -u -j$(nproc)
```

The build should pass without issues  
``` bash
cd openpilot/selfdrive/ui
./ui
```

*if you get an error, make sure to run the command below then run ./ui again to get more information. 
``` bash
cd openpilot/selfdrive/ui
export QT_DEBUG_PLUGINS=1
```

You should now see openpilot UI running on your computer.

## Replay 
Once you finished setting up your PC and can now run openpilot, lets explore replay. Replay replays all the messages logged while running openpilot. We are going to replay our default route called demo. Then we are going to use replay with Watch three to watch the three cameras simultaneously. 
```bash
# start a replay
cd openpilot/tools/replay
./replay --demo --dcam --ecam

# start watch3 in a different terminal
cd openpilot/selfdrive/ui
./watch3
```
You should now be able to See both the UI and video footage
![](https://i.imgur.com/IeaOdAb.png)

## Video 
YouTube video by George Hotz. It is a good resource, especially if you are a visual learner.

[![comma ai | George Hotz | Do openpilot tools work? Let's find out! | github.com/commaai](https://i.ytimg.com/an_webp/ixfAdv9sL30/mqdefault_6s.webp?du=3000&sqp=CNrdmZwG&rs=AOn4CLDOLgygFuJ7AmG0F2m1YB0qJaZ-0g)](https://www.youtube.com/watch?v=ixfAdv9sL30)

## Directory Structure

```
├── ubuntu_setup.sh     # Setup script for Ubuntu
├── mac_setup.sh        # Setup script for macOS
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
