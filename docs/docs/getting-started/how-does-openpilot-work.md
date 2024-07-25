# How does openpilot work?

A standard openpilot setup has three main components: a compatible car, a comma 3X, and a car harness.

## 1. A compatible car

openpilot uses the existing LKAS and ACC interfaces in the car.

## 2. A comma 3X

The comma 3X is a specialized device that has all the necessary parts to run openpilot:

* sensors: two road-facing cameras, a driver-facing camera, an [IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit), and GPS
* compute to run the driving model
* hardware to talk to the car

The car harness:

* connects the comma 3X to the car's CAN bus

Note that the comma 3X can be replaced with a normal PC with some webcams and a CAN interface to the car.

## 3. A car harness


<!--

concepts:
- CAN bus
- panda
- sensors

-->
