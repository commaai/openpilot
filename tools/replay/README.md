# Replay

`replay` allows you to simulate a driving session by replaying all messages logged during the use of openpilot. This provides a way to analyze and visualize system behavior as if it were live.

## Setup

Before starting a replay, you need to authenticate with your comma account using `auth.py`. This will allow you to access your routes from the server.

```bash
# Authenticate to access routes from your comma account:
python3 tools/lib/auth.py
```

## Replay a Remote Route
You can replay a route from your comma account by specifying the route name.

```bash
# Start a replay with a specific route:
tools/replay/replay <route-name>

# Example:
tools/replay/replay 'a2a0ccea32023010|2023-07-27--13-01-19'

# Replay the default demo route:
tools/replay/replay --demo
```

## Replay a Local Route
To replay a route stored locally on your machine, specify the route name and provide the path to the directory where the route files are stored.

```bash
# Replay a local route
tools/replay/replay <route-name> --data_dir="/path_to/route"

# Example:
# If you have a local route stored at /path_to_routes with segments like:
# a2a0ccea32023010|2023-07-27--13-01-19--0
# a2a0ccea32023010|2023-07-27--13-01-19--1
# You can replay it like this:
tools/replay/replay "a2a0ccea32023010|2023-07-27--13-01-19" --data_dir="/path_to_routes"
```

## Send Messages via ZMQ
By default, replay sends messages via MSGQ. To switch to ZMQ, set the ZMQ environment variable.

```bash
# Start replay and send messages via ZMQ:
ZMQ=1 tools/replay/replay <route-name>
```

## Usage
For more information on available options and arguments, use the help command:

``` bash
$ tools/replay/replay -h
Usage: tools/replay/replay [options] route
Mock openpilot components by publishing logged messages.

Options:
  -h, --help             Displays this help.
  -a, --allow <allow>    whitelist of services to send (comma-separated)
  -b, --block <block>    blacklist of services to send (comma-separated)
  -c, --cache <n>        cache <n> segments in memory. default is 5
  -s, --start <seconds>  start from <seconds>
  -x <speed>             playback <speed>. between 0.2 - 3
  --demo                 use a demo route instead of providing your own
  --data_dir <data_dir>  local directory with routes
  --prefix <prefix>      set OPENPILOT_PREFIX
  --dcam                 load driver camera
  --ecam                 load wide road camera
  --no-loop              stop at the end of the route
  --no-cache             turn off local cache
  --qcam                 load qcamera
  --no-hw-decoder        disable HW video decoding
  --no-vipc              do not output video
  --all                  do output all messages including uiDebug, userFlag.
                         this may causes issues when used along with UI

Arguments:
  route                  the drive to replay. find your drives at
                         connect.comma.ai
```

## Visualize the Replay in the openpilot UI
To visualize the replay within the openpilot UI, run the following commands:

```bash
tools/replay/replay <route-name>
cd selfdrive/ui && ./ui
```

## Try Radar Point Visualization with Rerun
To visualize radar points, run rp_visualization.py while tools/replay/replay is active.

```bash
tools/replay/replay <route-name>
python3 replay/rp_visualization.py
```

## Work with plotjuggler
If you want to use replay with plotjuggler, you can stream messages by running:

```bash
tools/replay/replay <route-name>
tools/plotjuggler/juggle.py --stream
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
