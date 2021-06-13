# PlotJuggler

We've extended [PlotJuggler](https://github.com/facontidavide/PlotJuggler) to plot all of your openpilot logs. Check out our plugins: https://github.com/commaai/PlotJuggler.

## Installation

Once you've cloned and are in openpilot, download PlotJuggler and install our plugins with this command:

`cd tools/plotjuggler && ./install.sh`

## Usage

```
batman@z840-openpilot:~/openpilot/tools/plotjuggler$ ./juggle.py -h
usage: juggle.py [-h] [--qlog] [--can] [--stream] [--layout [LAYOUT]] [route_name] [segment_number]

PlotJuggler plugin for reading rlogs

positional arguments:
  route_name         The name of the route that will be plotted. (default: None)
  segment_number     The index of the segment that will be plotted (default: None)

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

To get started exploring and plotting data live in your car, you can start PlotJuggler in streaming mode: `./juggle.py --stream`.

For this to work, you'll need a few things:
- Enable tethering on your comma device and connect your laptop.
- Run `export ZMQ=1` on the computer, which tells the streaming plugin backend to use ZMQ. If you're streaming locally, you can omit this step as ZMQ is used to transport data over the network.
- Most importantly: openpilot by default uses the MSGQ backend, so you'll need to [ssh into your device](https://github.com/commaai/openpilot/wiki/SSH) and run bridge. This simply re-broadcasts each message over ZMQ: `./cereal/messaging/bridge`.

Now start PlotJuggler using the above `juggle.py` command, and find the `Cereal Subscriber` plugin in the dropdown under Streaming. Click Start and enter the IP address of the comma two. You should now be seeing all the messages for each [service in openpilot](https://github.com/commaai/cereal/blob/master/services.py) received live from your car!

## Demo

For a quick demo, go through the installation step and run this command:

`./juggle.py "https://commadataci.blob.core.windows.net/openpilotci/d83f36766f8012a5/2020-02-05--18-42-21/0/rlog.bz2" --layout=demo_layout.xml`


![screenshot](https://i.imgur.com/cizHCH3.png)
