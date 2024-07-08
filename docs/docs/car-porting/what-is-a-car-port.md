# What is a car port?

All car ports live in `openpilot/selfdrive/car/`.

* interface.py: Interface for the car, defines the CarInterface class
* carstate.py: Reads CAN from car and builds openpilot CarState message
* carcontroller.py: Builds CAN messages to send to car
* values.py: Limits for actuation, general constants for cars, and supported car documentation
* radar_interface.py: Interface for parsing radar points from the car
