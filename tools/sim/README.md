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
## openpilot (in terminal 2)
```
./run_op_sim.sh # run openpilot-sim docker image
```
## Controls
Now put the focus on the terminal running bridge.py and you can control
openpilot driving in the simulation with the following keys

|  key  |   functionality   |
| :---: | :---------------: |
|   Up Arrow   |  Throttle  |
|  Down Arrow    |  Brake  |
|   2   | Cruise down 5 mph |
|   3   |   Cruise cancel   |
|   q   |     Exit all      |


--- 

## Develop Locally
In order to develop locally, you can build the images locally and mount the openpilot folder in the docker image as a volume

## openpilot (in terminal 2)
```
export DEV=1
./build_op_sim.sh

./run_op_sim.sh --develop=1
```
