openpilot tools
============

tools to facilitate development and debugging of openpilot

![Imgur](https://i.imgur.com/IdfBgwK.jpg)


Table of Contents
============

<!--ts-->
 * [Requirements](#requirements)
 * [Setup](#setup)
 * [Tool examples](#tool-examples)
   * [Replay driving data](#replay-driving-data)
   * [Debug car controls](#debug-car-controls)
   * [Stream replayed CAN messages to EON](#stream-replayed-can-messages-to-eon)
   * [Stream EON video data to a PC](#stream-eon-video-data-to-a-pc)
 * [Welcomed contributions](#welcomed-contributions)
<!--te-->


Requirements
============

openpilot tools and the following setup steps are developed and tested on Ubuntu 20.04, MacOS 10.14.2 and Python 3.8.2.

Setup
============
1. Run `ubuntu_setup.sh` or `mac_setup.sh`, make sure everything completed correctly

2. Compile openpilot by running ```scons``` in the openpilot directory
   or alternatively run ```./openpilot_build.sh``` (uses a pre-configured docker container)

3. Try out some tools!


Tool examples
============


Replay driving data
-------------

**Hardware needed**: none

`unlogger.py` replays data collected with [dashcam](https://github.com/commaai/openpilot/tree/dashcam) or [openpilot](https://github.com/commaai/openpilot).

Unlogger with remote data:

```
# Log in via browser
python lib/auth.py

# Start unlogger
python replay/unlogger.py <route-name>
#Example:
#python replay/unlogger.py '3533c53bb29502d1|2019-12-10--01-13-27'

# In another terminal you can run a debug visualizer:
python replay/ui.py   # Define the environmental variable HORIZONTAL is the ui layout is too tall
```

Unlogger with local data downloaded from device or https://my.comma.ai:

```
python replay/unlogger.py <route-name> <path-to-data-directory>

#Example:

#python replay/unlogger.py '99c94dc769b5d96e|2018-11-14--13-31-42' /home/batman/unlogger_data

#Within /home/batman/unlogger_data:
#  99c94dc769b5d96e|2018-11-14--13-31-42--0--fcamera.hevc
#  99c94dc769b5d96e|2018-11-14--13-31-42--0--rlog.bz2
#  ...
```
![Imgur](https://i.imgur.com/Yppe0h2.png)

LogReader with remote data

```python
from tools.lib.logreader import LogReader
from tools.lib.route import Route
route = Route('3533c53bb29502d1|2019-12-10--01-13-27')
log_paths = route.log_paths()
events_seg0 = list(LogReader(log_paths[0]))
print(len(events_seg0), 'events logged in first segment')
```

Debug car controls
-------------

**Hardware needed**: [panda](panda.comma.ai), [giraffe](https://comma.ai/shop/products/giraffe/), joystick

Use the panda's OBD-II port to connect with your car and a usb cable to connect the panda to your pc.
Also, connect a joystick to your pc.

`joystickd.py` runs a deamon that reads inputs from a joystick and publishes them over zmq.
`boardd` sends the CAN messages from your pc to the panda.
`debug_controls` is a mocked version of `controlsd.py` and uses input from a joystick to send controls to your car.

Make sure the conditions are met in the panda to allow controls (e.g. cruise control engaged). You can also make a modification to the panda code to always allow controls.

Usage:
```
python carcontrols/joystickd.py

# In another terminal:
BASEDIR=$(pwd) selfdrive/boardd/boardd

# In another terminal:
python carcontrols/debug_controls.py

```
![Imgur](steer.gif)


Stream replayed CAN messages to EON
-------------

**Hardware needed**: 2 x [panda](panda.comma.ai), [debug board](https://comma.ai/shop/products/panda-debug-board/), [EON](https://comma.ai/shop/products/eon-gold-dashcam-devkit/).

It is possible to replay CAN messages as they were recorded and forward them to a EON. 
Connect 2 pandas to the debug board. A panda connects to the PC, the other panda connects to the EON.

Usage:
```
# With MOCK=1 boardd will read logged can messages from a replay and send them to the panda.
MOCK=1 selfdrive/boardd/tests/boardd_old.py

# In another terminal:
python replay/unlogger.py <route-name> <path-to-data-directory>

```
![Imgur](https://i.imgur.com/AcurZk8.jpg)


Stream EON video data to a PC
-------------

**Hardware needed**: [EON](https://comma.ai/shop/products/eon-gold-dashcam-devkit/), [comma Smays](https://comma.ai/shop/products/comma-smays-adapter/).

You can connect your EON to your pc using the Ethernet cable provided with the comma Smays and you'll be able to stream data from your EON, in real time, with low latency. A useful application is being able to stream the raw video frames at 20fps, as captured by the EON's camera.

Usage:
```
# ssh into the eon and run loggerd with the flag "--stream". In ../selfdrive/manager.py you can change:
# ...
# "loggerd": ("selfdrive/loggerd", ["./loggerd"]),
# ...
# with:
# ...
# "loggerd": ("selfdrive/loggerd", ["./loggerd", "--stream"]),
# ...

# On the PC:
# To receive frames from the EON and re-publish them. Set PYGAME env variable if you want to display the video stream
python streamer/streamerd.py
```

![Imgur](stream.gif)


Welcomed contributions
=============

* Documentation: code comments, better tutorials, etc..
* Support for other platforms other than Ubuntu 20.04.
* Performance improvements: the tools have been developed on high-performance workstations (12+ logical cores with 32+ GB of RAM), so they are not optimized for running efficiently. For example, `ui.py` might not be able to run real-time on most PCs.
* More tools: anything that you think might be helpful to others.
