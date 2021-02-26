# PlotJuggler
We've extended [PlotJuggler](https://github.com/facontidavide/PlotJuggler) to plot all of your openpilot logs

Here's our fork: https://github.com/commaai/PlotJuggler 

## Installation

Once you've cloned openpilot, run this command inside this directory:

`./install.sh`

## Usage

```
batman@z840-openpilot:~/openpilot/tools/plotjuggler$ ./juggle.py -h
usage: juggle.py [-h] [route_name] [segment_number]

PlotJuggler plugin for reading rlogs

positional arguments:
  route_name      The name of the route that will be plotted. (default: None)
  segment_number  The index of the segment that will be plotted (default: None)

optional arguments:
  -h, --help      show this help message and exit
```

Example:

`./juggle.py "0982d79ebb0de295|2021-01-17--17-13-08"`
