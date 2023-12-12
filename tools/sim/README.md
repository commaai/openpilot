openpilot in simulator
=====================

openpilot implements a [bridge](run_bridge.py) that allows it to run in the [MetaDrive simulator](https://github.com/metadriverse/metadrive).

## Launching openpilot
First, start openpilot.
``` bash
# Run locally
./tools/sim/launch_openpilot.sh
```

## Bridge usage
```
$ ./run_bridge.py -h
usage: run_bridge.py [-h] [--joystick] [--high_quality] [--dual_camera]
Bridge between the simulator and openpilot.

options:
  -h, --help            show this help message and exit
  --joystick
  --high_quality
  --dual_camera
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
./run_bridge.py
```