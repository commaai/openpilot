# Rerun.io

[ReRun](https://rerun.io) is a powerful tool for visualizing time series data, offering features tailored for analyzing openpilot logs. Explore the possibilities of rerun.io and its seamless integration with openpilot.

## Installation

Once you've [set up the openpilot environment](../README.md), this command will download PlotJuggler and install our plugins:

`cd tools/rerun && ./run.py --install`

## Usage

```
$ ./run.py -h
usage: run.py [-h] [--demo] [route_or_segment_name]

A helper to run run on openpilot routes

positional arguments:
  route_or_segment_name  The route or segment name to plot (cabana share URL accepted) (default: None)

optional arguments:
  -h, --help             show this help message and exit
  --demo                 Use the demo route instead of providing one (default: False)

```

Examples using route name:

`./run.py "a2a0ccea32023010/2023-07-27--13-01-19"`


## Streaming

Explore live data from your car! Follow these steps to stream from your comma device to your laptop:
- Enable wifi tethering on your comma device
- [SSH into your device](https://github.com/commaai/openpilot/wiki/SSH) and run `cd /data/openpilot && ./cereal/messaging/bridge`
- On your laptop, connect to the device's wifi hotspot
- Start rerun.io with ./run.py --demo.


## Demo

For a quick demo, go through the installation step and run this command:

`./run.py --demo`


## Layouts

If you create a layout that's useful for others, consider upstreaming it.

### Tuning

Use this layout to improve your car's tuning and generate plots for tuning PRs. Also see the [tuning wiki](https://github.com/commaai/openpilot/wiki/Tuning) and tuning PR template.

`--layout layouts/tuning.xml`


![screenshot](https://i.imgur.com/cizHCH3.png)