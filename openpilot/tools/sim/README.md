openpilot in simulator
=====================

openpilot implements a [bridge](run_bridge.py) that allows it to run in the [MetaDrive simulator](https://github.com/metadriverse/metadrive).

## Launching openpilot
First, start openpilot.
``` bash
# Run locally
./openpilot/tools/sim/launch_openpilot.sh
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

### Installing MetaDrive
MetaDrive isn't part of the default dependencies yet, so install it into the openpilot venv:
``` bash
uv pip install "metadrive-simulator @ git+https://github.com/commaai/metadrive.git@minimal"
```

### Launching Metadrive
Start bridge processes located in openpilot/tools/sim:
``` bash
./run_bridge.py
```

### macOS
The bridge runs on Apple silicon. Two things to know:

* Apple's OpenGL driver only exposes 4.1 through a core profile, so the bridge asks panda3d
  for `gl-version 4 1 core` with the `pandagl` display. MetaDrive's shaders have to compile
  under GLSL 3.30+ for the camera feeds to render; if you get black frames, make sure you're
  on a MetaDrive revision with the macOS shader fixes.
* The simulator processes are started with the `spawn` start method everywhere, since an
  OpenGL context can't be inherited through `fork()`. Anything you add that's shared with a
  sim process must be created from `openpilot.tools.sim.lib.common.SIM_MP_CTX`.
