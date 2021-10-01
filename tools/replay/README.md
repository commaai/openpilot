Stream CAN messages to your device
-------------

Replay CAN messages as they were recorded using a [panda jungle](https://comma.ai/shop/products/panda-jungle). The jungle has 6x OBD-C ports for connecting all your comma devices.

`can_replay.py` is a convenient script for when any CAN data will do.

In order to replay specific route:
```bash
MOCK=1 selfdrive/boardd/tests/boardd_old.py

# In another terminal:
python replay/unlogger.py <route-name> <path-to-data-directory>
```

Replay driving data
-------------

`unlogger.py` replays all the messages logged while running openpilot.

Unlogger with remote data:

```bash
# Log in via browser
python lib/auth.py

# Start unlogger
python replay/unlogger.py <route-name>
# Example:
# python replay/unlogger.py '4cf7a6ad03080c90|2021-09-29--13-46-36'

# In another terminal you can run a debug visualizer:
python replay/ui.py   # Define the environmental variable HORIZONTAL is the ui layout is too tall

# Or run the normal openpilot UI
cd selfdrive/ui && ./ui
```

![Imgur](https://i.imgur.com/Yppe0h2.png)

## usage
``` bash
$ ./unlogger.py -h
usage: unlogger.py [-h] [--no-loop] [--min | --enabled ENABLED] [--disabled DISABLED] [--tl PUBLISH_TIME_LENGTH] [--no-realtime]
                   [--no-interactive] [--bind-early] [--no-visionipc] [--start-time START_TIME]
                   [route_name] [data_dir] [address_mapping [address_mapping ...]]

Mock openpilot components by publishing logged messages.

positional arguments:
  route_name            The route whose messages will be published. (default: None)
  data_dir              Path to directory in which log and camera files are located. (default: None)
  address_mapping       Pairs <service>=<zmq_addr> to publish <service> on <zmq_addr>. (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --no-loop             Stop at the end of the replay. (default: False)
  --min
  --enabled ENABLED
  --disabled DISABLED
  --tl PUBLISH_TIME_LENGTH
                        Length of interval in event time for which messages should be published. (default: None)
  --no-realtime         Publish messages as quickly as possible instead of realtime. (default: True)
  --no-interactive      Disable interactivity. (default: True)
  --bind-early          Bind early to avoid dropping messages. (default: False)
  --no-visionipc        Do not output video over visionipc (default: False)
  --start-time START_TIME
                        Seek to this absolute time (in seconds) upon starting playback. (default: 0.0)
```
