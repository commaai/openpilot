openpilot in simulator
=====================
Needs Ubuntu 16.04

## Checkout openpilot
```
cd ~/
git clone https://github.com/commaai/openpilot.git

# Add export PYTHONPATH=$HOME/openpilot to your bashrc
# Have a working tensorflow+keras in python3.7.3 (with [packages] in openpilot/Pipfile)
```
## Install (in terminal 1)
```
cd ~/openpilot/tools/sim
./start_carla.sh  # install CARLA 0.9.7 and start the server
```
## openpilot (in terminal 2)
```
cd ~/openpilot/selfdrive/
PASSIVE=0 NOBOARD=1 ./manager.py
```
## bridge (in terminal 3)
```
# links carla to openpilot, will "start the car" according to manager
cd ~/openpilot/tools/sim
./bridge.py
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


