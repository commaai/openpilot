# PlotJuggler

[PlotJuggler](https://github.com/facontidavide/PlotJuggler) is a tool to quickly visualize time series data, and we've written plugins to parse openpilot logs. Check out our plugins: https://github.com/commaai/PlotJuggler.

## Installation

Once you've [set up the openpilot environment](../README.md), this command will download PlotJuggler and install our plugins:

`cd tools/plotjuggler && ./juggle.py --install`

## Usage

```
$ ./juggle.py -h
usage: juggle.py [-h] [--demo] [--can] [--stream] [--layout [LAYOUT]] [--install] [--dbc DBC]
                 [route_or_segment_name] [segment_count]

A helper to run PlotJuggler on openpilot routes

positional arguments:
  route_or_segment_name
                        The route or segment name to plot (cabana share URL accepted) (default: None)
  segment_count         The number of segments to plot (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --demo                Use the demo route instead of providing one (default: False)
  --can                 Parse CAN data (default: False)
  --stream              Start PlotJuggler in streaming mode (default: False)
  --layout [LAYOUT]     Run PlotJuggler with a pre-defined layout (default: None)
  --install             Install or update PlotJuggler + plugins (default: False)
  --dbc DBC             Set the DBC name to load for parsing CAN data. If not set, the DBC will be automatically
                        inferred from the logs. (default: None)

```

Examples using route name:

`./juggle.py "a2a0ccea32023010/2023-07-27--13-01-19"`

Examples using segment range:

`./juggle.py "a2a0ccea32023010/2023-07-27--13-01-19/1"`

`./juggle.py "a2a0ccea32023010/2023-07-27--13-01-19/1/q" # use qlogs`

## Streaming

Explore live data from your car! Follow these steps to stream from your comma device to your laptop:
- Enable wifi tethering on your comma device
- [SSH into your device](https://github.com/commaai/openpilot/wiki/SSH) and run `cd /data/openpilot && ./cereal/messaging/bridge`
- On your laptop, connect to the device's wifi hotspot
- Start PlotJuggler with `ZMQ=1 ./juggle.py --stream`, find the `Cereal Subscriber` plugin in the dropdown under Streaming, and click `Start`.

If streaming to PlotJuggler from a replay on your PC, simply run: `./juggle.py --stream` and start the cereal subscriber.

## Demo

For a quick demo, go through the installation step and run this command:

`./juggle.py --demo --layout=layouts/demo.xml`

## Layouts

If you create a layout that's useful for others, consider upstreaming it.

### Tuning

Use this layout to improve your car's tuning and generate plots for tuning PRs. Also see the [tuning wiki](https://github.com/commaai/openpilot/wiki/Tuning) and tuning PR template.

`--layout layouts/tuning.xml`


![screenshot](https://i.imgur.com/cizHCH3.png)
