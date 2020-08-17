openpilot in simulator
=====================
There are docker images for both Carla (the simulator we're using) and openpilot

## Checkout openpilot
```
cd ~/
git clone https://github.com/commaai/openpilot.git
cd ~/openpilot/tools/sim
```
## Run Carla (in terminal 1)
```
./start_carla_docker.sh  # run the CARLA 0.9.7 docker image
```
## Run openpilot (in terminal 2)
Run the openpilot image and two scripts : `selfdrive/manager.py` and `tools/sim/bridge.py`
```
./build_container # pull openpilot-sim image
./run_op_sim.sh # run openpilot-sim image
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


--- 

## Develop Locally
In order to develop locally, you can build the images locally and mount the openpilot folder in the docker image as a volume

## openpilot (in terminal 2)
```
BUILD=1 ./build_op_sim.sh

DEVELOP=1 ./run_op_sim.sh --develop=1
```
