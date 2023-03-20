# Cabana

<img src="https://cabana.comma.ai/img/cabana.jpg" width="640" height="267" />

Cabana is a tool developed to view raw CAN data. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

## Usage Instructions

```bash
$ ./cabana -h
Usage: ./_cabana [options] route

Options:
  -h, --help             Displays this help.
  --demo                 use a demo route instead of providing your own
  --qcam                 load qcamera
  --ecam                 load wide road camera
  --stream               read can messages from live streaming
  --zmq <zmq>            the ip address on which to receive zmq messages
  --data_dir <data_dir>  local directory with routes
  --no-vipc              do not output video
  --dbc <dbc>            dbc file to open

Arguments:
  route                  the drive to replay. find your drives at
                         connect.comma.ai
```

See [openpilot wiki](https://github.com/commaai/openpilot/wiki/Cabana)
