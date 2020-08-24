openpilot in simulator
=====================

## Setup

Checkout openpilot
```
cd ~/ && git clone https://github.com/commaai/openpilot.git
```

## Running the simulator

First, start the CARLA server.
```
cd ~/openpilot/tools/sim
./start_carla.sh
```


The next steps can either run on your local machine or in the openpilot-sim docker container.
Use `start_openpilot_docker.sh` to start the docker container and start a shell inside of it.

Next, start the bridge.
```
cd ~/openpilot/tools/sim
./bridge.py
```

Then, start the bridge.
```
cd ~/openpilot/tools/sim
./launch_openpilot.sh
```

## Controls

Now put the focus on the terminal running bridge.py and you can control
openpilot driving in the simulation with the following keys

|  key  |   functionality   |
| :---: | :---------------: |
|   1   |  Cruise up 5 mph  |
|   2   | Cruise down 5 mph |
|   3   |   Cruise cancel   |
|   q   |     Exit all      |

