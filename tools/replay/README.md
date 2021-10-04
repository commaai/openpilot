Stream CAN messages to your device
-------------

Replay CAN messages as they were recorded using a [panda jungle](https://comma.ai/shop/products/panda-jungle). The jungle has 6x OBD-C ports for connecting all your comma devices.

`can_replay.py` is a convenient script for when any CAN data will do.

In order to replay specific route:
```bash
MOCK=1 selfdrive/boardd/tests/boardd_old.py

# In another terminal:
selfdrive/ui/replay/replay <route-name>
```

Replay driving data
-------------

`replay` replays all the messages logged while running openpilot.

Replay with remote data:

```bash
# Log in via browser to have access to non public route
python lib/auth.py

# Start replay
selfdrive/ui/replay/replay <route-name>
# Example:
# selfdrive/ui/replay/replay '4cf7a6ad03080c90|2021-09-29--13-46-36'
# or use --demo to replay the default demo route:
# selfdrive/ui/replay/replay --demo

# In another terminal you can run a debug visualizer:
python replay/ui.py   # Define the environmental variable HORIZONTAL is the ui layout is too tall

# Or run the normal openpilot UI
cd selfdrive/ui && ./ui
```

![Imgur](https://i.imgur.com/Yppe0h2.png)

## usage
``` bash
$ selfdrive/ui/replay/replay -h
Usage: selfdrive/ui/replay/replay [options] route
Mock openpilot components by publishing logged messages.

Options:
  -h, --help             Displays this help.
  -a, --allow <allow>    whitelist of services to send
  -b, --block <block>    blacklist of services to send
  -s, --start <seconds>  start from <seconds>
  --demo                 use a demo route instead of providing your own
  --dcam                 load driver camera
  --ecam                 load wide road camera

Arguments:
  route                  the drive to replay. find your drives at
                         connect.comma.ai
```
