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

openpilot tools and the following setup steps are developed and tested on Ubuntu 16.04, MacOS 10.14.2 and Python 3.7.3.

Setup
============
<!-- 
TODO: These instructions maybe outdated, follow ubuntu_setup.sh setup instructions

1. Install native dependencies (Mac and Ubuntu sections listed below)

    **Ubuntu**

    - core tools
      ```bash
      sudo apt install git curl python-pip
      sudo pip install --upgrade pip>=18.0 pipenv
      ```

    - ffmpeg (tested with 3.3.2)
      ```bash
      sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev
      ```

    - build tools
      ```bash
      sudo apt install autoconf automake clang clang-3.8 libtool pkg-config build-essential
      ```

    - libarchive-dev (tested with 3.1.2-11ubuntu0.16.04.4)
      ```bash
      sudo apt install libarchive-dev
      ```

    - qt python binding (tested with python-qt4, 4.11.4+dfsg-1build4)
      ```bash
      sudo apt install python-qt4
      ```

    - zmq 4.2.3 (required for replay)
      ```bash
      curl -LO https://github.com/zeromq/libzmq/releases/download/v4.2.3/zeromq-4.2.3.tar.gz
      tar xfz zeromq-4.2.3.tar.gz
      cd zeromq-4.2.3
      ./autogen.sh
      ./configure CPPFLAGS=-DPIC CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS=-fPIC --disable-shared --enable-static
      make
      sudo make install
      ```

    **Mac**

    - brew
      ``` bash
      /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
      ```

    - core tools
      ``` bash
      brew install git
      sudo pip install --upgrade pip pipenv
      xcode-select --install
      ```

    - ffmpeg (tested with 3.4.1)
        ```bash
        brew install ffmpeg
        ```

    - build tools
        ```bash
        brew install autoconf automake libtool llvm pkg-config
        ```

    - libarchive-dev (tested with 3.3.3)
        ```bash
        brew install libarchive
        ```

    - qt for Mac
        ```bash
        brew install qt
        ```

    - zmq 4.3.1 (required for replay)
        ```bash
        brew install zeromq
        ```

2. Install Cap'n Proto

    ```bash
    curl -O https://capnproto.org/capnproto-c++-0.6.1.tar.gz
    tar xvf capnproto-c++-0.6.1.tar.gz
    cd capnproto-c++-0.6.1
    ./configure --prefix=/usr/local CPPFLAGS=-DPIC CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS=-fPIC --disable-shared --enable-static
    make -j4
    sudo make install

    cd ..
    git clone https://github.com/commaai/c-capnproto.git
    cd c-capnproto
    git submodule update --init --recursive
    autoreconf -f -i -s
    CFLAGS="-fPIC" ./configure --prefix=/usr/local
    make -j4
    sudo make install
    ```

2. Clone openpilot if you haven't already

    ```bash
    git clone https://github.com/commaai/openpilot.git
    cd openpilot
    pipenv install # Install dependencies in a virtualenv
    pipenv shell # Activate the virtualenv
    ```

    **For Mac users**

    Recompile longitudinal_mpc for mac

    Navigate to:
    ``` bash
    cd selfdrive/controls/lib/longitudinal_mpc
    make clean
    make
    ```

3. Install tools dependencies

    ```bash
    cd tools
    pip install -r requirements.txt # Install tools dependencies in virtualenv
    ```

4. Add openpilot to your `PYTHONPATH`.

    For bash users:
    ```bash
    echo 'export PYTHONPATH="$PYTHONPATH:<path-to-openpilot>"' >> ~/.bashrc
    source ~/.bashrc
    ```
-->
1. Run ubuntu_setup.sh, make sure everything completed correctly

2. Compile openpilot by running ```scons``` in the openpilot directory

3. Add some folders to root
    ```bash
    sudo mkdir /data
    sudo mkdir /data/params
    sudo chown $USER /data/params
    ```
    

4. Try out some tools!


Tool examples
============


Replay driving data
-------------

**Hardware needed**: none

`unlogger.py` replays data collected with [chffrplus](https://github.com/commaai/chffrplus) or [openpilot](https://github.com/commaai/openpilot).

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
`boardd.py` sends the CAN messages from your pc to the panda.
`debug_controls` is a mocked version of `controlsd.py` and uses input from a joystick to send controls to your car.

Usage:
```
python carcontrols/joystickd.py

# In another terminal:
selfdrive/boardd/tests/boardd_old.py # Make sure the safety setting is hardcoded to ALL_OUTPUT

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
* Support for other platforms other than Ubuntu 16.04.
* Performance improvements: the tools have been developed on high-performance workstations (12+ logical cores with 32+ GB of RAM), so they are not optimized for running efficiently. For example, `ui.py` might not be able to run real-time on most PCs.
* More tools: anything that you think might be helpful to others.
