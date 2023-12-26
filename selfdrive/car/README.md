# selfdrive/car

Check out [this blog post](https://blog.comma.ai/how-to-write-a-car-port-for-openpilot/) for a high-level overview of porting a car.

## Useful car porting utilities

Testing car ports in your car is very time-consuming. Check out these utilities to do basic checks on your work before running it in your car.

### [Cabana](/tools/cabana/README.md)

View your car's CAN signals through DBC files, which openpilot uses to parse and create messages that talk to the car.

Example:
```bash
> tools/cabana/cabana '1bbe6bf2d62f58a8|2022-07-14--17-11-43'
```

### [selfdrive/debug/auto_fingerprint.py](/selfdrive/debug/auto_fingerprint.py)

Given a route and platform, automatically inserts FW fingerprints from the platform into the correct place in values.py

Example:
```bash
> python selfdrive/debug/auto_fingerprint.py '1bbe6bf2d62f58a8|2022-07-14--17-11-43' 'SUBARU OUTBACK 6TH GEN'
Attempting to add fw version for:  SUBARU OUTBACK 6TH GEN
```

### [selfdrive/car/tests/test_car_interfaces.py](/selfdrive/car/tests/test_car_interfaces.py)

Finds common bugs for car interfaces, without even requiring a route.


#### Example: Typo in signal name
```bash
> pytest selfdrive/car/tests/test_car_interfaces.py -k subaru  # replace with the brand you are working on

=====================================================================
FAILED selfdrive/car/tests/test_car_interfaces.py::TestCarInterfaces::test_car_interfaces_165_SUBARU_LEGACY_7TH_GEN - KeyError: 'CruiseControlOOPS'

```

### [selfdrive/debug/test_car_model.py](/selfdrive/debug/test_car_model.py)

Given a route, runs most of the car interface to check for common errors like missing signals, blocked panda messages, and safety mismatches.

#### Example: panda safety mismatch for gasPressed
```bash
> python selfdrive/debug/test_car_model.py '4822a427b188122a|2023-08-14--16-22-21'

=====================================================================
FAIL: test_panda_safety_carstate (__main__.CarModelTestCase.test_panda_safety_carstate)
Assert that panda safety matches openpilot's carState
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/batman/xx/openpilot/openpilot/selfdrive/car/tests/test_models.py", line 380, in test_panda_safety_carstate
    self.assertFalse(len(failed_checks), f"panda safety doesn't agree with openpilot: {failed_checks}")
AssertionError: 1 is not false : panda safety doesn't agree with openpilot: {'gasPressed': 116}
```


## Car port structure

### interface.py
Generic interface to send and receive messages from CAN (controlsd uses this to communicate with car)

### fingerprints.py
Fingerprints for matching to a specific car

### carcontroller.py
Builds CAN messages to send to car

##### carstate.py
Reads CAN from car and builds openpilot CarState message

##### values.py
Limits for actuation, general constants for cars, and supported car documentation

##### radar_interface.py
Interface for parsing radar points from the car
