# What is openpilot?

[openpilot](http://github.com/commaai/openpilot) is an open source driver assistance system. Currently, openpilot performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW), and Lane Departure Warning (LDW) for a growing variety of [supported car makes, models, and model years](docs/CARS.md). In addition, while openpilot is engaged, a camera-based Driver Monitoring (DM) feature alerts distracted and asleep drivers. See more about [the vehicle integration](docs/INTEGRATION.md) and [limitations](docs/LIMITATIONS.md).


## How do I use it?

openpilot is designed to be used on the comma 3X.

## How does it work?

In short, openpilot uses the car's existing APIs for the built-in [ADAS](https://en.wikipedia.org/wiki/Advanced_driver-assistance_system) system and simply provides better acceleration, braking, and steering inputs than the stock system.
