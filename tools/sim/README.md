openpilot in simulator
=====================

openpilot implements a [bridge](bridge.py) that allows it to run in the [CARLA simulator](https://carla.org/). 

## System Requirements

openpilot doesn't have any extreme hardware requirements, however CARLA requires an NVIDIA graphics card and is very resource-intensive and may not run smoothly on your system. For this case, we have a low quality mode you can activate by running:
```
./start_openpilot_docker.sh --low_quality
```

You can also check out the [CARLA python documentation](https://carla.readthedocs.io/en/latest/python_api/) to find more parameters to tune that might increase performance on your system.

## Running the simulator

First, start the CARLA server in one terminal.
```
./start_carla.sh
```

Then, start the bridge and openpilot in another terminal.
```
./start_openpilot_docker.sh
```

To engage openpilot press 1 a few times while focused on bridge.py to increase the cruise speed.

## Controls

You can control openpilot driving in the simulation with the following keys

|  key  |   functionality   |
| :---: | :---------------: |
|   1   |  Cruise up 5 mph  |
|   2   | Cruise down 5 mph |
|   3   |   Cruise cancel   |
|   q   |     Exit all      |

To see the options for changing the environment, such as the town, spawn point or precipitation, you can run `./start_openpilot_docker.sh --help`.
This will print the help output inside the docker container. You need to exit the docker container before running `./start_openpilot_docker.sh` again.

## Further Reading

The following resources contain more details and troubleshooting tips.
* [CARLA on the openpilot wiki](https://github.com/commaai/openpilot/wiki/CARLA)
