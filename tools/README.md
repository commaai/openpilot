openpilot tools
============

tools to facilitate development and debugging of openpilot

![Imgur](https://i.imgur.com/IdfBgwK.jpg)


Table of Contents
============

<!--ts-->
 * [System requirements](#requirements)
 * [Setup](#setup)
 * [Tool examples](#tool-examples)
   * [Replay driving data](#replay-driving-data)
   * [Debug car controls](#debug-car-controls)
   * [Juggle plots](#juggle-plots)
   * [Run openpilot in a simulator](#run-openpilot-in-a-simulator)
 * [Welcomed contributions](#welcomed-contributions)
<!--te-->


System requirements
============

openpilot tools and the following setup steps are developed and tested on:
* Ubuntu 20.04
* MacOS 10.14.2 (only building is tested on MacOS)
* Python 3.8.2


Setup
============
1. Run `ubuntu_setup.sh` or `mac_setup.sh`, and make sure that everything completed correctly

2. Compile openpilot by running ```scons``` in the openpilot directory
   or alternatively run ```./openpilot_build.sh``` (uses a pre-configured docker container)

3. Try out some tools!


Tool examples
============


[Replay driving data](replay)
-------------

Review video and log data from routes and stream CAN messages to your EON.


[Debug car controls](carcontrols)
-------------

Use a joystick to control your car.


[Juggle plots](plotjuggler)
-------------

Easily plot openpilot logs with [PlotJuggler](https://github.com/facontidavide/PlotJuggler), an open source tool for visualizing time series data.


[Run openpilot in a simulator](sim)
-------------

Test openpilots performance in a simulated environment. The [CARLA simulator](https://github.com/carla-simulator/carla) allows you to set a variety of features like:
* Weather
* Environment physics
* Cars
* Traffic and pedestrians


Welcomed contributions
=============

* Documentation: code comments, better tutorials, etc..
* Support for other platforms other than Ubuntu 20.04.
* Performance improvements: the tools have been developed on high-performance workstations (12+ logical cores with 32+ GB of RAM), so they are not optimized for running efficiently. For example, `ui.py` might not be able to run real-time on most PCs.
* More tools: anything that you think might be helpful to others.
