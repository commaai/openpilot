# What is a car port?

A car port enables openpilot support on a particular car. Each car model openpilot supports needs to be individually ported. All car ports live in `openpilot/selfdrive/car/car_specific.py` and `opendbc_repo/opendbc/car`.

The complexity of a car port varies depending on many factors including:
* existing openpilot support for similar cars
* architecture and APIs available in the car


# Structure of a car port
* `interface.py`: Interface for the car, defines the CarInterface class
* `carstate.py`: Reads CAN from car and builds openpilot CarState message
* `carcontroller.py`: Builds CAN messages to send to car
* `values.py`: Limits for actuation, general constants for cars, and supported car documentation
* `radar_interface.py`: Interface for parsing radar points from the car


# Overiew

[Jason Young](https://github.com/jyoung8607) gave a talk at COMMA_CON with an overview of the car porting process. The talk is available on YouTube:

https://youtu.be/KcfzEHB6ms4?si=5szh1PX6TksOCKmM
