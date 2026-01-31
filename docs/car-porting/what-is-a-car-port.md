# What is a car port?

A car port enables openpilot support on a particular car. Each car model openpilot supports needs to be individually ported. The complexity of a car port varies depending on many factors including:

* existing openpilot support for similar cars
* architecture and APIs available in the car


# Structure of a car port

Virtually all car-specific code is contained in two other repositories: [opendbc](https://github.com/commaai/opendbc) and [panda](https://github.com/commaai/panda).

## opendbc

Each car brand is supported by a standard interface structure in `opendbc/car/[brand]`:

* `interface.py`: Interface for the car, defines the CarInterface class
* `carstate.py`: Reads CAN messages from the car and builds openpilot CarState messages
* `carcontroller.py`: Control logic for executing openpilot CarControl actions on the car
* `[brand]can.py`: Composes CAN messages for carcontroller to send
* `values.py`: Limits for actuation, general constants for cars, and supported car documentation
* `radar_interface.py`: Interface for parsing radar points from the car, if applicable

## panda

* `board/safety/safety_[brand].h`: Brand-specific safety logic
* `tests/safety/test_[brand].py`: Brand-specific safety CI tests

## openpilot

For historical reasons, openpilot still contains a small amount of car-specific logic. This will eventually be migrated to opendbc or otherwise removed.

* `selfdrive/car/car_specific.py`: Brand-specific event logic

# Overview

[Jason Young](https://github.com/jyoung8607) gave a talk at COMMA_CON with an overview of the car porting process. The talk is available on YouTube:

https://www.youtube.com/watch?v=XxPS5TpTUnI
