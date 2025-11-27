# CarState signals

## Required for basic lateral control

* `brakePressed`
* `cruiseState`
* `doorOpen`
* `espDisabled`
* `gasPressed`
* `gearShifter`
* `leftBlinker` / `rightBlinker`
* `seatbeltUnlatched`
* `standstill`
* `steeringAngleDeg`
* `steeringPressed`
* `steeringTorque`
* `steerFaultPermanent`
* `steerFaultTemporary`
* `vCruise`
* `wheelSpeeds.[fl|fr|rl|rr]`: Speed of each of the car's four wheels, in m/s. The car's CAN bus often broadcasts the
speed in kph, so the helper function `parse_wheel_speeds` performs this conversion by default.

## Recommended / Required for openpilot longitudinal control

* `accFaulted`
* `espActive`
* `parkingBrake`

## Application Dependent

* `blockPcmEnable`
* `buttonEnable`
* `brakeHoldActive`
* `carFaultedNonCritical`
* `invalidLkasSetting`
* `lowSpeedAlert`
* `regenBraking`
* `steeringAngleOffsetDeg`
* `steeringDisengage`
* `steeringTorqueEps`
* `stockLkas`
* `vCruiseCluster`
* `vEgoCluster`
* `vehicleSensorsInvalid`

## Automatically populated

* `buttonEvents`

These values are populated automatically by `parse_wheel_speeds`:

* `aEgo`: Acceleration of the ego vehicle, Kalman filtered derivative of `vEgo`.
* `vEgo`: Speed of the ego vehicle, Kalman filtered from `vEgoRaw`.
* `vEgoRaw`: Speed of the ego vehicle, based on the average of all four wheel speeds, unfiltered.

## Optional

* `brake`
* `charging`
* `fuelGauge`
* `leftBlindspot` / `rightBlindspot`
* `steeringRateDeg`
* `stockAeb`
* `stockFcw`
* `yawRate`
