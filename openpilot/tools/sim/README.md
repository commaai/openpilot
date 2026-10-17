openpilot in simulator
=====================

The simulator drives openpilot through a logged route. It replays the segment until openpilot engages, then a simulated car follows openpilot's controls. Each logged camera frame is reprojected to where the simulated car is, so openpilot sees the consequences of its own driving.

The reprojection assumes a flat road, so it's only accurate for small deviations from the logged path. The episode ends if the car gets more than 2 m or 8° off the logged path. Lane changes in the log end it too, since openpilot stays in its lane. Segments must be from a comma 3X (1928x1208 cameras).

## Usage
``` bash
./openpilot/tools/sim/run.py
```

This starts openpilot and the simulator, and shows their status until you quit. openpilot engages on its own once it's ready.

```
$ ./run.py -h
usage: run.py [-h] [--route ROUTE] [--start-sec START_SEC]

options:
  --route ROUTE          segment to drive
  --start-sec START_SEC  seconds into the segment to start from (default: 15 s on the default route)
```

| key  | functionality                |
|------|------------------------------|
|  2   | engage / set speed down      |
|  1   | resume / set speed up        |
|  3   | cancel                       |
| w s  | gas / brake                  |
| a d  | steer                        |
| z x  | left / right blinker         |
|  r   | restart the episode          |
|  i   | toggle ignition              |
|  q   | quit                         |

openpilot's output goes to `~/.comma/sim.log`.
