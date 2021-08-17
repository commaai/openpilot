# PlotJuggler

We've extended [PlotJuggler](https://github.com/facontidavide/PlotJuggler) to plot all of your openpilot logs. Check out our plugins: https://github.com/commaai/PlotJuggler.

## Installation

Once you've cloned and are in openpilot, download PlotJuggler and install our plugins with this command:

`cd tools/plotjuggler && ./install.sh`

## Usage

```
$ ./juggle.py -h
usage: juggle.py [-h] [--qlog] [--can] [--stream] [--layout [LAYOUT]] [route_name] [segment_number] [segment_count]

PlotJuggler plugin for reading openpilot logs

positional arguments:
  route_name         The route name to plot (cabana share URL accepted) (default: None)
  segment_number     The index of the segment to plot (default: None)
  segment_count      The number of segments to plot (default: 1)

optional arguments:
  -h, --help         show this help message and exit
  --qlog             Use qlogs (default: False)
  --can              Parse CAN data (default: False)
  --stream           Start PlotJuggler without a route to stream data using Cereal (default: False)
  --layout [LAYOUT]  Run PlotJuggler with a pre-defined layout (default: None)
```

Example:

`./juggle.py "0982d79ebb0de295|2021-01-17--17-13-08"`

## Streaming

Explore live data from your car! (If running openpilot locally, skip to `./juggle.py --stream`.)

- On comma device:
  - Enable wifi tethering
- On laptop:
  - `export ZMQ=1` in plotjuggler's environment, to tell the streaming plugin to use ZMQ.
  - Connect to comma device wifi
- [ssh into comma device](https://github.com/commaai/openpilot/wiki/SSH):
  - Run `./cereal/messaging/bridge` to re-broadcast openpilot's MSGQ to ZMQ over the network.

Start PlotJuggler with `./juggle.py --stream`, find the `Cereal Subscriber` plugin in the dropdown under Streaming, and click Start. Observe live messages from [openpilot's services](https://github.com/commaai/cereal/blob/master/services.py)!

## Demo

For a quick demo, go through the installation step and run this command:

`./juggle.py "https://commadataci.blob.core.windows.net/openpilotci/d83f36766f8012a5/2020-02-05--18-42-21/0/rlog.bz2" --layout=demo_layout.xml`


![screenshot](https://i.imgur.com/cizHCH3.png)
