# Replay

## Replay driving data

`replay` replays all the messages logged while running openpilot.

```bash
# Log in via browser to have access to routes from your comma account
python tools/lib/auth.py

# Start a replay
tools/replay/replay <route-name>

# Example:
tools/replay/replay 'a2a0ccea32023010|2023-07-27--13-01-19'
# or use --demo to replay the default demo route:
tools/replay/replay --demo

# watch the replay with the normal openpilot UI
cd selfdrive/ui && ./ui

# or try out a debug visualizer:
python replay/ui.py
```

## usage

``` bash
$ tools/replay/replay -h
Usage: tools/replay/replay [options] route
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

## watch3

watch all three cameras simultaneously from your comma three routes with watch3

simply replay a route using the `--dcam` and `--ecam` flags:

```bash
# start a replay
cd tools/replay && ./replay --demo --dcam --ecam

# then start watch3
cd selfdrive/ui && ./watch3
```

![](https://i.imgur.com/IeaOdAb.png)

## Stream CAN messages to your device

Replay CAN messages as they were recorded using a [panda jungle](https://comma.ai/shop/products/panda-jungle). The jungle has 6x OBD-C ports for connecting all your comma devices. Check out the [jungle repo](https://github.com/commaai/panda_jungle) for more info.

In order to run your device as if it was in a car:
* connect a panda jungle to your PC
* connect a comma device or panda to the jungle via OBD-C
* run `can_replay.py`

``` bash
batman:replay$ ./can_replay.py -h
usage: can_replay.py [-h] [route_or_segment_name]

Replay CAN messages from a route to all connected pandas and jungles
in a loop.

positional arguments:
  route_or_segment_name
                        The route or segment name to replay. If not
                        specified, a default public route will be
                        used. (default: None)

optional arguments:
  -h, --help            show this help message and exit
```
