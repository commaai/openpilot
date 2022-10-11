openpilot in simulator
=====================

openpilot implements a [bridge](bridge.py) that allows it to run in the [CARLA simulator](https://carla.org/).

## System Requirements

openpilot doesn't have any extreme hardware requirements, however CARLA requires an NVIDIA graphics card and is very resource-intensive and may not run smoothly on your system.
For this case, we have a the simulator in low quality by default.

You can also check out the [CARLA python documentation](https://carla.readthedocs.io/en/latest/python_api/) to find more parameters to tune that might increase performance on your system.

## Running the simulator
Start Carla simulator, openpilot and bridge processes located in tools/sim:
``` bash
# Terminal 1
./start_carla.sh

# Terminal 2 - Run openpilot and bridge in one Docker:
./start_openpilot_docker.sh

# Running the latest local code execute
    # Terminal 2:
    ./launch_openpilot.sh
    # Terminal 3
    ./bridge.py
```

### Bridge usage
_Same commands hold for start_openpilot_docker_
```
$ ./bridge.py -h
Usage: bridge.py [options]
Bridge between CARLA and openpilot.

Options:
  -h, --help            show this help message and exit
  --joystick            Use joystick input to control the car
  --high_quality        Set simulator to higher quality (requires good GPU)
  --town TOWN           Select map to drive in
  --spawn_point NUM     Number of the spawn point to start in
  --host HOST           Host address of Carla client (127.0.0.1 as default)
  --port PORT           Port of Carla client (2000 as default)
```

To engage openpilot press 1 a few times while focused on bridge.py to increase the cruise speed.
All inputs:

| key  |   functionality   |
|:----:|:-----------------:|
|  1   |  Cruise up 1 mph  |
|  2   | Cruise down 1 mph |
|  3   |   Cruise cancel   |
|  q   |     Exit all      |
| wasd | Control manually  |

## Further Reading

The following resources contain more details and troubleshooting tips.
* [CARLA on the openpilot wiki](https://github.com/commaai/openpilot/wiki/CARLA)
