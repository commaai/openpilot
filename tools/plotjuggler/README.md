# PlotJuggler

We've extended [PlotJuggler](https://github.com/facontidavide/PlotJuggler) to plot all of your openpilot logs. Check out our plugin: https://github.com/commaai/PlotJuggler.

## Installation

Once you've cloned openpilot, install our plugin with this command:

`cd tools/plotjuggler && ./install.sh`

Usage requires an installation of PlotJuggler. On systems with snap (e.g. Ubuntu), you can install PlotJuggler with this command:

 `sudo snap install plotjuggler`

## Usage

```
batman@z840-openpilot:~/openpilot/tools/plotjuggler$ ./juggle.py -h
usage: juggle.py [-h] [--qlog] [--layout [LAYOUT]] [route_name] [segment_number]

PlotJuggler plugin for reading rlogs

positional arguments:
  route_name         The name of the route that will be plotted. (default: None)
  segment_number     The index of the segment that will be plotted (default: None)

optional arguments:
  -h, --help         show this help message and exit
  --qlog             Use qlogs (default: False)
  --layout [LAYOUT]  Run PlotJuggler with a pre-defined layout (default: None)
```

Example:

`./juggle.py "0982d79ebb0de295|2021-01-17--17-13-08"`

## Demo:

For a quick demo, go through the installation step and run this command:

`./juggle.py "https://commadataci.blob.core.windows.net/openpilotci/d83f36766f8012a5/2020-02-05--18-42-21/0/rlog.bz2" --layout=demo_layout.xml`


![screenshot](https://i.imgur.com/cizHCH3.png)
