Replay driving data
-------------

**Hardware needed**: none

`unlogger.py` replays data collected with [dashcam](https://github.com/commaai/openpilot/tree/dashcam) or [openpilot](https://githucommaai/openpilot).

Unlogger with remote data:

```
# Log in via browser
python lib/auth.py

# Start unlogger
python replay/unlogger.py <route-name>
#Example:
#python replay/unlogger.py '3533c53bb29502d1|2019-12-10--01-13-27'

# In another terminal you can run a debug visualizer:
python replay/ui.py   # Define the environmental variable HORIZONTAL is the ui layout is too tall
```

Unlogger with local data downloaded from device or https://my.comma.ai:

```
python replay/unlogger.py <route-name> <path-to-data-directory>

#Example:

#python replay/unlogger.py '99c94dc769b5d96e|2018-11-14--13-31-42' /home/batman/unlogger_data

#Within /home/batman/unlogger_data:
#  99c94dc769b5d96e|2018-11-14--13-31-42--0--fcamera.hevc
#  99c94dc769b5d96e|2018-11-14--13-31-42--0--rlog.bz2
#  ...
```
![Imgur](https://i.imgur.com/Yppe0h2.png)

LogReader with remote data

```python
from tools.lib.logreader import LogReader
from tools.lib.route import Route
route = Route('3533c53bb29502d1|2019-12-10--01-13-27')
log_paths = route.log_paths()
events_seg0 = list(LogReader(log_paths[0]))
print(len(events_seg0), 'events logged in first segment')
```

Stream replayed CAN messages to EON
-------------

**Hardware needed**: 2 x [panda](panda.comma.ai), [debug board](https://comma.ai/shop/products/panda-debug-board/), [EON](https://comma.ai/shop/products/eon-gold-dashcam-devkit/).

It is possible to replay CAN messages as they were recorded and forward them to a EON. 
Connect 2 pandas to the debug board. A panda connects to the PC, the other panda connects to the EON.

Usage:
```
# With MOCK=1 boardd will read logged can messages from a replay and send them to the panda.
MOCK=1 selfdrive/boardd/tests/boardd_old.py

# In another terminal:
python replay/unlogger.py <route-name> <path-to-data-directory>

```
![Imgur](https://i.imgur.com/AcurZk8.jpg)
