# tools/car_porting

Check out [this blog post](https://blog.comma.ai/how-to-write-a-car-port-for-openpilot/) for a high-level overview of porting a car.

## Useful car porting utilities

Testing car ports in your car is very time-consuming. Check out these utilities to do basic checks on your work before running it in your car.

### [Cabana](/tools/cabana/README.md)

View your car's CAN signals through DBC files, which openpilot uses to parse and create messages that talk to the car.

Example:
```bash
> tools/cabana/cabana '1bbe6bf2d62f58a8|2022-07-14--17-11-43'
```

### [tools/car_porting/auto_fingerprint.py](/tools/car_porting/auto_fingerprint.py)

Given a route and platform, automatically inserts FW fingerprints from the platform into the correct place in fingerprints.py

Example:
```bash
> python3 tools/car_porting/auto_fingerprint.py '1bbe6bf2d62f58a8|2022-07-14--17-11-43' 'OUTBACK'
Attempting to add fw version for:  OUTBACK
```

### [selfdrive/car/tests/test_car_interfaces.py](/selfdrive/car/tests/test_car_interfaces.py)

Finds common bugs for car interfaces, without even requiring a route.


#### Example: Typo in signal name
```bash
> pytest selfdrive/car/tests/test_car_interfaces.py -k subaru  # replace with the brand you are working on

=====================================================================
FAILED selfdrive/car/tests/test_car_interfaces.py::TestCarInterfaces::test_car_interfaces_165_SUBARU_LEGACY_7TH_GEN - KeyError: 'CruiseControlOOPS'

```

### [tools/car_porting/test_car_model.py](/tools/car_porting/test_car_model.py)

Given a route, runs most of the car interface to check for common errors like missing signals, blocked panda messages, and safety mismatches.

#### Example: panda safety mismatch for gasPressed
```bash
> python3 tools/car_porting/test_car_model.py '4822a427b188122a|2023-08-14--16-22-21'

=====================================================================
FAIL: test_panda_safety_carstate (__main__.CarModelTestCase.test_panda_safety_carstate)
Assert that panda safety matches openpilot's carState
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/batman/xx/openpilot/openpilot/selfdrive/car/tests/test_models.py", line 380, in test_panda_safety_carstate
    self.assertFalse(len(failed_checks), f"panda safety doesn't agree with openpilot: {failed_checks}")
AssertionError: 1 is not false : panda safety doesn't agree with openpilot: {'gasPressed': 116}
```

## Jupyter notebooks

To use these notebooks, install Jupyter within your [openpilot virtual environment](/tools/README.md).

```bash
uv pip install jupyter ipykernel
```

Launching:

```bash
jupyter notebook
```

### [examples/subaru_steer_temp_fault.ipynb](/tools/car_porting/examples/subaru_steer_temp_fault.ipynb)

An example of searching through a database of segments for a specific condition, and plotting the results.

![steer warning example](https://github.com/commaai/openpilot/assets/9648890/d60ad120-4b44-4974-ac79-adc660fb8fe2)

*a plot of the steer_warning vs steering angle, where we can see it is clearly caused by a large steering angle change*

### [examples/subaru_long_accel.ipynb](/tools/car_porting/examples/subaru_long_accel.ipynb)

An example of plotting the response of an actuator when it is active.

![brake pressure example](https://github.com/commaai/openpilot/assets/9648890/8f32cf1d-8fc0-4407-b540-70625ebbf082)

*a plot of the brake_pressure vs acceleration, where we can see it is a fairly linear response.*

### [examples/ford_vin_fingerprint.ipynb](/tools/car_porting/examples/ford_vin_fingerprint.ipynb)

In this example, we use the public comma car segments database to check if vin fingerprinting is feasible for ford.

```
vin: 1FM5K8GC7LGXXXXXX real platform: FORD EXPLORER 6TH GEN              determined platform: mock                              correct: False
vin: 00000000000XXXXXX real platform: FORD ESCAPE 4TH GEN                determined platform: mock                              correct: False
vin: 3FTTW8F98NRXXXXXX real platform: FORD MAVERICK 1ST GEN              determined platform: mock                              correct: False
vin: 1FTVW1EL4NWXXXXXX real platform: FORD F-150 LIGHTNING 1ST GEN       determined platform: FORD F-150 LIGHTNING 1ST GEN      correct: True
vin: 1FM5K7LC0MGXXXXXX real platform: FORD EXPLORER 6TH GEN              determined platform: mock                              correct: False
vin: WF0NXXGCHNJXXXXXX real platform: FORD FOCUS 4TH GEN                 determined platform: mock                              correct: False
vin: 1FMCU9J94MUXXXXXX real platform: FORD ESCAPE 4TH GEN                determined platform: mock                              correct: False
vin: 5LM5J7XC9LGXXXXXX real platform: FORD EXPLORER 6TH GEN              determined platform: mock                              correct: False
vin: 3FMCR9B69NRXXXXXX real platform: FORD BRONCO SPORT 1ST GEN          determined platform: mock                              correct: False
vin: 3FMTK3SU0MMXXXXXX real platform: FORD MUSTANG MACH-E 1ST GEN        determined platform: FORD MUSTANG MACH-E 1ST GEN       correct: True
vin: 1FM5K8HC7MGXXXXXX real platform: FORD EXPLORER 6TH GEN              determined platform: mock                              correct: False
vin: 1FM5K8GC7NGXXXXXX real platform: FORD EXPLORER 6TH GEN              determined platform: mock                              correct: False
vin: 5LM5J7XC8MGXXXXXX real platform: FORD EXPLORER 6TH GEN              determined platform: mock                              correct: False
vin: 3FTTW8E31PRXXXXXX real platform: FORD MAVERICK 1ST GEN              determined platform: mock                              correct: False
vin: 3FTTW8E99NRXXXXXX real platform: FORD MAVERICK 1ST GEN              determined platform: mock                              correct: False
```

### [examples/find_segments_with_message.ipynb](/tools/car_porting/examples/find_segments_with_message.ipynb)

Searches for segments where a set of given CAN message IDs are present. In the example, we search for all messages
used for CAN-based ignition detection.

```
Match found: 46b21f1c5f7aa885/2024-01-23--15-19-34/20/s    JEEP GRAND CHEROKEE V6 2018            ['VW CAN Ign']
Match found: a63a23c3e628f288/2023-11-05--18-36-20/8/s     JEEP GRAND CHEROKEE V6 2018            ['VW CAN Ign']
Match found: ce31b7a998781ba8/2024-01-19--07-05-29/23/s    JEEP GRAND CHEROKEE 2019               ['VW CAN Ign']
Match found: e1dfba62a4e33f7b/2023-12-25--19-31-00/4/s     JEEP GRAND CHEROKEE 2019               ['VW CAN Ign']
Match found: e1dfba62a4e33f7b/2024-01-10--14-33-57/2/s     JEEP GRAND CHEROKEE 2019               ['VW CAN Ign']
Match found: ae679616266f4096/2023-12-05--15-43-46/4/s     RAM HD 5TH GEN                         ['Tesla 3/Y CAN Ign']
Match found: ae679616266f4096/2023-11-18--17-49-42/3/s     RAM HD 5TH GEN                         ['Tesla 3/Y CAN Ign']
Match found: ae679616266f4096/2024-01-03--21-57-09/25/s    RAM HD 5TH GEN                         ['Tesla 3/Y CAN Ign']
Match found: 6dae2984cc53cd7f/2023-12-10--11-53-15/17/s    FORD BRONCO SPORT 1ST GEN              ['Rivian CAN Ign']
Match found: 6dae2984cc53cd7f/2023-12-03--17-31-17/29/s    FORD BRONCO SPORT 1ST GEN              ['Rivian CAN Ign']
Match found: 6dae2984cc53cd7f/2023-11-27--23-29-07/1/s     FORD BRONCO SPORT 1ST GEN              ['Rivian CAN Ign']
```
