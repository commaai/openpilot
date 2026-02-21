# Cabana

Cabana is a tool developed to view raw CAN data. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

## Building

Cabana requires Qt5. Install Qt5 before building:

**Ubuntu:**
```bash
sudo apt-get install -y --no-install-recommends \
  qtbase5-dev \
  qtbase5-dev-tools \
  qttools5-dev-tools \
  libqt5charts5-dev \
  libqt5svg5-dev \
  libqt5serialbus5-dev \
  libqt5x11extras5-dev \
  libqt5opengl5-dev
```

**macOS:**
```bash
brew install qt@5
brew link qt@5
```

Once Qt5 is installed, the `cabana` wrapper script will handle building and running automatically:
```bash
# from the openpilot root
tools/cabana/cabana
```

Or build manually with scons and run the binary directly:
```bash
scons -j$(nproc) tools/cabana/_cabana
tools/cabana/_cabana
```

## Usage Instructions

```bash
$ ./cabana -h
Usage: ./cabana [options] route

Options:
  -h, --help                     Displays help on commandline options.
  --help-all                     Displays help including Qt specific options.
  --demo                         use a demo route instead of providing your own
  --auto                         Auto load the route from the best available source (no video):
                                 internal, openpilotci, comma_api, car_segments, testing_closet
  --qcam                         load qcamera
  --ecam                         load wide road camera
  --msgq                         read can messages from msgq
  --panda                        read can messages from panda
  --panda-serial <panda-serial>  read can messages from panda with given serial
  --socketcan <socketcan>        read can messages from given SocketCAN device
  --zmq <ip-address>             read can messages from zmq at the specified ip-address
                                 messages
  --data_dir <data_dir>          local directory with routes
  --no-vipc                      do not output video
  --dbc <dbc>                    dbc file to open

Arguments:
  route                          the drive to replay. find your drives at
                                 connect.comma.ai
```

## Examples

### Running Cabana in Demo Mode
To run Cabana using a built-in demo route, use the following command:

```shell
cabana --demo
```

### Loading a Specific Route

To load a specific route for replay, provide the route as an argument:

```shell
cabana "a2a0ccea32023010|2023-07-27--13-01-19"
```

Replace "0ccea32023010|2023-07-27--13-01-19" with your desired route identifier.


### Running Cabana with multiple cameras
To run Cabana with multiple cameras, use the following command:

```shell
cabana "a2a0ccea32023010|2023-07-27--13-01-19" --dcam --ecam
```

### Streaming CAN Messages from a comma Device

[SSH into your device](https://github.com/commaai/openpilot/wiki/SSH) and start the bridge with the following command:

```shell
cd /data/openpilot/cereal/messaging/
./bridge &
```

Then Run Cabana with the device's IP address:

```shell
cabana --zmq <ipaddress>
```

Replace &lt;ipaddress&gt; with your comma device's IP address.

While streaming from the device, Cabana will log the CAN messages to a local directory. By default, this directory is ~/cabana_live_stream/. You can change the log directory in Cabana by navigating to menu -> tools -> settings.

After disconnecting from the device, you can replay the logged CAN messages from the stream selector dialog -> browse local route.

### Streaming CAN Messages from Panda

To read CAN messages from a connected Panda, use the following command:

```shell
cabana --panda
```

### Using the Stream Selector Dialog

If you run Cabana without any arguments, a stream selector dialog will pop up, allowing you to choose the stream.

```shell
cabana
```

## Additional Information

For more information, see the [openpilot wiki](https://github.com/commaai/openpilot/wiki/Cabana)
