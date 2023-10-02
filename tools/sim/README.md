openpilot in simulator
=====================

openpilot implements a [bridge](run_bridge.py) that allows it to run in the [MetaDrive simulator](https://github.com/metadriverse/metadrive) or [CARLA simulator](https://carla.org/).

## Launching openpilot
First, start openpilot.
``` bash
# Run locally
./tools/sim/launch_openpilot.sh
```

## Bridge usage
```
$ ./run_bridge.py -h
usage: run_bridge.py [-h] [--joystick] [--high_quality] [--dual_camera] [--simulator SIMULATOR] [--town TOWN] [--spawn_point NUM_SELECTED_SPAWN_POINT] [--host HOST] [--port PORT]

Bridge between the simulator and openpilot.

options:
  -h, --help            show this help message and exit
  --joystick
  --high_quality
  --dual_camera
  --simulator SIMULATOR
  --town TOWN
  --spawn_point NUM_SELECTED_SPAWN_POINT
  --host HOST
  --port PORT
```

#### Bridge Controls:
- To engage openpilot press 2, then press 1 to increase the speed and 2 to decrease.
- To disengage, press "S" (simulates a user brake)

#### All inputs:

```
| key  |   functionality       |
|------|-----------------------|
|  1   | Cruise Resume / Accel |
|  2   | Cruise Set    / Decel |
|  3   | Cruise Cancel         |
|  r   | Reset Simulation      |
|  i   | Toggle Ignition       |
|  q   | Exit all              |
| wasd | Control manually      |
```

## MetaDrive

### Launching Metadrive
Start bridge processes located in tools/sim:
``` bash
./run_bridge.py --simulator metadrive
```

## Carla

CARLA is also partially supported, though the performance is not great. openpilot doesn't have any extreme hardware requirements, however CARLA requires an NVIDIA graphics card and is very resource-intensive and may not run smoothly on your system.
For this case, we have the simulator in low quality by default.

You can also check out the [CARLA python documentation](https://carla.readthedocs.io/en/latest/python_api/) to find more parameters to tune that might increase performance on your system.

### Launching Carla
Start Carla simulator and bridge processes located in tools/sim:
``` bash
# Terminal 1
./start_carla.sh

# Terminal 2
./run_bridge.py --simulator carla
```

## Further Reading

The following resources contain more details and troubleshooting tips.
* [CARLA on the openpilot wiki](https://github.com/commaai/openpilot/wiki/CARLA)
