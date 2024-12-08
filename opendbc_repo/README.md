<div align="center" style="text-align: center;">

<h1>opendbc</h1>
<p>
  <b>opendbc is a Python API for your car.</b>
  <br>
  Control the gas, brake, steering, and more. Read the speed, steering angle, and more.
</p>

<h3>
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

---

Most cars since 2016 have electronically-actuatable steering, gas, and brakes thanks to [LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system#Lane_keeping_and_next_technologies) and [ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control).
The goal of this project is to support controlling the steering, gas, and brakes on every single one of those cars.

While the primary focus is on supporting ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), we're also interested in reading and writing as many things as we can (EV charge status, lock/unlocking doors, etc) such that we can build the best vehicle management app ever.

---

This README and the [supported cars list](docs/CARS.md) are all the docs for the opendbc project.
Everything you need to know to use, contribute, and extend opendbc are in these docs.

## Quick start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
pre-commit run --all-files       # run the linter
./test.sh                        # all-in-one for setup, build, lint, and test
```

[`examples/`](examples/) contains small example programs that can read state from the car and control the steering, gas, and brakes.
[`examples/joystick.py`](examples/joystick.py) allows you to control a car with a joystick.

### Project Structure
* [`opendbc/dbc/`](opendbc/dbc/) is a repository of [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files
* [`opendbc/can/`](opendbc/can/) is a library for parsing and building CAN messages from DBC files
* [`opendbc/car/`](opendbc/car/) is a high-level library for interfacing with cars using Python

## How to Port a Car

This guide covers everything from adding support to a new car all the way to improving existing cars (e.g. adding longitudinal control or radar parsing). If similar cars to yours are already compatible, most of this work is likely already done for you.

At its most basic, a car port will control the steering on a car. A "complete" car port will have all of: lateral control, longitudinal control, good tuning for both lateral and longitudinal, radar parsing (if equipped), fuzzy fingerprinting, and more. The new car support docs will clearly communicate each car's support level.

### Connect to the Car

The first step is to get connected to the car with a comma 3X and a car harness.
The car harness gets you connected to two different CAN buses and splits one of those buses to send our own actuation messages.

If you're lucky, a harness compatible with your car will already be designed and sold on comma.ai/shop. 
If you're not so lucky, start with a "developer harness" from comma.ai/shop and crimp on whatever connector you need.

### Structure of a port

Depending on , most of this basic structure will already be in place.

The entirery of a car port lives in `opendbc/car/<brand>/`:
* `carstate.py`: parses out the relevant information from the CAN stream using the car's DBC file
* `carcontroller.py`: outputs CAN messages to control the car 
* `<brand>can.py`: thin Python helpers around the DBC file to build CAN messages
* `fingerprints.py`: database of ECU firmware versions for identifying car models
* `interface.py`: high level class for interfacing with the car
* `radar_interface.py`: parses out the radar 
* `values.py`: enumerates the brand's supported cars

### Reverse Engineer CAN messages

Start off by recording a route with lots of interesting events: enable LKAS and ACC, turn the steering wheel both extremes, etc. Then, load up that route in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

#### Longitudinal

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control and tune it.

## Contributing

All opendbc development is coordinated on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section. 

### Roadmap

Short term
- [ ] `pip install opendbc`
- [ ] 100% type coverage
- [ ] 100% line coverage
- [ ] Make car ports easier: refactors, tools, tests, and docs
- [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

Longer term
- [ ] Extend support to every car with LKAS + ACC interfaces
- [ ] Automatic lateral and longitudinal control/tuning evaluation
- [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
- [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

Contributions towards anything here are welcome.

### Bounties

Every car port is eligible for a bounty:
* $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
* $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
* $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

In addition to the standard bounties, we also offer higher value bounties for more popular cars. See those at [comma.ai/bounties](comma.ai/bounties).

## FAQ

***How do I use this?*** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.

***Which cars are supported?*** See the [supported cars list](docs/CARS.md).

***Can I add support for my car?*** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

***Which cars can be supported?*** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

***How does this work?*** In short, we designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

***Is there a timeline or roadmap for adding car support?*** No, most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate. 

### Terms

* **port**: refers to the integration and support of a specific car
* **lateral control**: aka steering control
* **longitudinal control**: aka gas/brakes control
* **fingerprinting**: automatic process for identifying the car
* **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
* **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
* **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware to attach to the car and intercept the ADAS messages
* **[panda](https://github.com/commaai/panda)**: hardware used to get on a car's CAN bus
* **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: computers or control modules inside the car
* **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: a bus that connects the ECUs in a car
* **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: our tool for reverse engineering CAN messages
* **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: contains definitions for messages on a CAN bus
* **[openpilot](https://github.com/commaai/openpilot)**: an ADAS system for cars supported by opendbc
* **[comma](https://github.com/commaai)**: the company behind opendbc
* **[comma 3X](https://comma.ai/shop/comma-3x)**: the hardware used to run openpilot

### More resources

* [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
* [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
* [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
* [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
* [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS 
* [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
* [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors.
