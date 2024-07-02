# Cabana

Cabana is a tool developed to view raw CAN data. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

## Usage Instructions

```bash
$ ./cabana -h
Usage: ./cabana [options] route

Options:
  -h, --help                     Displays help on commandline options.
  --help-all                     Displays help including Qt specific options.
  --demo                         use a demo route instead of providing your own
  --qcam                         load qcamera
  --ecam                         load wide road camera
  --stream                       read can messages from live streaming
  --panda                        read can messages from panda
  --panda-serial <panda-serial>  read can messages from panda with given serial
  --socketcan <socketcan>        read can messages from given SocketCAN device
  --zmq <zmq>                    the ip address on which to receive zmq
                                 messages
  --data_dir <data_dir>          local directory with routes
  --no-vipc                      do not output video
  --dbc <dbc>                    dbc file to open

Arguments:
  route                          the drive to replay. find your drives at
                                 connect.comma.ai
```

## Segment Ranges

you can specify which segments from a route to load

```bash
# the 4th segment
cabana 'a2a0ccea32023010|2023-07-27--13-01-19/4'

# the 4th, 5th and 6th segment
cabana 'a2a0ccea32023010|2023-07-27--13-01-19/4:6'

# the last segment
cabana 'a2a0ccea32023010|2023-07-27--13-01-19/-1'

# the first 5 segments
cabana 'a2a0ccea32023010|2023-07-27--13-01-19/:5'

# all except first segment
cabana 'a2a0ccea32023010|2023-07-27--13-01-19/1:'
```

See [openpilot wiki](https://github.com/commaai/openpilot/wiki/Cabana)
