openpilot in simulator
=====================

There are docker images for both CARLA (the simulator we're using) and openpilot


## Setup

Checkout openpilot
```
cd ~/ && git clone https://github.com/commaai/openpilot.git
```

Install the dependencies


## Running the simulator

First, start the CARLA server.
```
cd ~/openpilot/tools/sim
./start_carla.sh
```

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
|   1   | Cruise up 5 mph |
|   2   | Cruise down 5 mph |
|   3   |   Cruise cancel   |
|   Up Arrow   |  Throttle  |
|  Down Arrow    |  Brake  |
|   Left Arrow   |  Left Turn  |
|   Right Arrow   |  Right Turn  |
|   q   |     Exit all      |

