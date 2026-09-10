# Merging a New Port

The openpilot driving experience is designed to be:

* safe (SAFETY.md)
* cooperative
* plug-and-play
* high quality

New brand or model ports are evaluated against these goals before they are merged into upstream
openpilot. This guide discusses the review process necessary to merge a new port.

In case of conflicts between this guide and comma safety standards, SAFETY.md always controls.

# Merge Process

New brand ports will initially be merged in dashcam mode. This allows for the comma.ai openpilot team
to further validate safety and quality before enabling official support with plug-and-play operation.
New model ports, which only add a new model to an already-supported ADAS platform, may be reviewable
with driving logs only, and may be directly merged as fully supported.

Cars that don't meet all guidelines aren't eligible for full support. However, on a case-by-case basis,
they may still be merged behind the dashcam flag.

## Unofficial Support (dashcam mode)

These ports are maintained on a best-effort basis and must pass a subset of CI testing. They will not
operate by default when plugged into a car. They can only be driven by advanced users who manually
remove the dashcam flag, or by installing a custom fork.

Some example reasons why a port might be in dashcam mode:

* A work-in-progress port that has advanced enough to pass tests and provide some value
* An otherwise finished port, pending comma safety and quality validation
* The car's control API does not allow a cooperative driving experience (see Tesla)
* Lateral/longitudinal control lockouts without a safe, effective mitigation
* Unresolved issues with lateral/longitudinal control quality and reliability
* No harness yet available for purchase, DIY harness fabrication required

## Official Support

Ports meeting these guidelines have been verified as safe and should deliver a high quality, plug and
play experience. These ports are eligible to appear in CARS.md and will appear in the car selector
at shop.comma.ai. They are all actively maintained, with full CI testing.

# Support Requirements

## Car Harness

* Full pinout documented, including any required pass-through pins
* The stock CAN termination resistances must be measured and documented
* Harness design places actuator controls on bus 0/2, bus 1 is optional
* Harness design allows stock ADAS operation with openpilot disconnected or disabled
* Connector shells and terminals available COTS or 3D model available

## Installation Experience

* Harness should plug in at LKA camera, other locations must be strongly justified
* Any special parts or tools needed for installation should be documented
* Fingerprinting must be as fully automatic as possible (fuzzy FP)
  * VIN fingerprinting acceptable with proper rigor/testing
  * Full functionality, including fingerprinting, must be available without comma power

## Integration Experience

* Dashcam mode works, all operation returns to stock
* No unpleasant/atypical car UI experiences for the driver, no warnings or beeps
* Should handle late startup seamlessly (cold boot, no comma power, thermal delay)
* Integration with the instrument cluster/HUD
  * Displayed speed and cruise setpoint match between the cluster and openpilot, both miles and km
    * Speed can be offset/adjusted in openpilot if necessary
    * Continues to scale properly if the driver changes the car's display units
  * Stock ADAS cluster display signals are replicated by openpilot
    * Displayed state for LKA active, standby, off (LKA on/off buttons and settings may be ignored)
    * Lane lines (if applicable)
    * Lead car presence (if applicable, openpilot does not make lead distance available)
    * Configured follow distance (if applicable, replicate using driving personality setting)

## Lateral Control

* Use only safe APIs designed for highway speed ADAS
* Probe the car's lateral control API limits
  * Identify speed envelope limits, any cutoff above zero handled correctly
  * Identify actuation limits, document even if openpilot cannot use the maximum
* Lateral API fault signals must be populated in CarState, must alert on loss of control
* If the car's API is torque based, use the lat accel torque controller unless justified not to
* Lateral control operates within safe limits
  * 2.5 m/s/s lateral accel for torque control
  * TODO: what are the steering angle control limits?
  * TODO: what are the curvature control limits?
  * Handles variability in actuator performance between cars (see HKG)
* Driver override is handled correctly and safely
  * openpilot should detect driver input and back off
    * Try to identify and match the stock LKA override threshold
    * Threshold will probably land in the 0.6-0.8 Nm range, if scaling is known
    * Threshold must avoid false wheel touches during normal driving
  * Must be a cooperative driving experience (see Tesla for a counterexample)

### Testing

* Maximum actuation limits don't exceed comma safety guidelines
* Good driving plan conformance, check with PlotJuggler
  * Test lane changes on both flat and road crown boundaries
  * TODO: Can we test std dev between desired and actual?
  * TODO: Can we test cost with the algorithm from the controls challenge?
* Reasonable steerActuatorDelay, check with PlotJuggler
* Reasonable wheel touch threshold (DM, lane change)
* Sane learned value for tire stiffness, CarParams startup value is set similarly
* Sane learned value for steer ratio, CarParams startup value is set similarly

## Longitudinal Control

### Control with Stock ACC

* Adaptive cruise is required: identify and reject engagement on non-adaptive cruise
* Speed envelope probed, engagement limits set if other than full Stop-and-Go
* Cancel spam works to maintain engagement state sync, reject engagements if needed
* (Optional) Resume spam works from a standstill
* (Optional) Identify faults like sensor-obstructed, populate in CarState

### Control with openpilot

* Use only safe APIs designed for highway speed ADAS
* Probe the car's longitudinal control API limits
  * Identify speed envelope limits, any cutoff above zero handled correctly
  * Identify actuation limits, document even if openpilot cannot use the maximum
* Don't exceed actuation limits in SAFETY.md
* Longitudinal API fault signals must be populated in CarState, must alert on loss of control
* Main switch on/off states explicitly identified (may require special state tracking)
* All CC button signals explicitly identified, driver control experience matches stock
* Test for good conformance to the openpilot longitudinal plan

### Longitudinal Testing

Applies to both stock and openpilot longitudinal control.

* Good quality operation when cruising at various speeds
* Good quality operation when tracking a lead car, including ACC braking
* Driver gas override (with and without disengage-on-gas)
* Short standstill, with automatic resume in traffic
* Long standstill (look for unintended roll-away)
* Disengage at standstill (look for unintended roll-away)
* Rejected engagements are handled properly, openpilot/car engagement state remain synced
* Brake signal is 100% reliable and 100% free of faults on disengage
  * Verify with very light braking (issues here with brake pressed switch vs pressure threshold)
  * Verify with ACC braking (make sure you're not looking at a brake light signal)
  * OR of multiple signals is acceptable

## CarState and Miscellaneous

* All CarState message frequencies match the car, or lowest freq if variable/triggered
* Turn signals, note signal behavior (oscillating vs fixed, one-touch vs latched)
* Doors, identify all
* Seatbelt, identify driver
* Parking brake (especially handbrake, EPB if it prevents ACC)
* Gearshift position
  * Detect all gears on automatic transmissions, including sport and manumatic variants
  * Manual trans acceptable if Reverse is detected

## Panda Safety

* Lateral and longitudinal actuation limits are safe, and exactly match openpilot
* Safety-relevant message sizes and frequencies checked, frequency checks match openpilot
* Safety-relevant message checksums checked, if present
* Safety-relevant message counters checked, if present
* Verify source messages/signals actually (not functionally) identical to openpilot
  * Driver input torque for steering override
  * Brake (OR of multiple signals acceptable, still matching openpilot)
  * ACC control state, as applicable to that car
  * Gas
  * Speed
  * Cruise control buttons
    * Cancel/Set logic for stock ACC cars
    * Full cruise button logic for openpilot longitudinal cars
* Match openpilot logic for detecting vehicle standstill
* All CI tests passing, including MISRA
* No instances of Controls Mismatch
* No CAN messages dropped during normal driving (a few discards are expected at startup/shutdown)

## Optional

* openpilot longitudinal control
* Radar points
* BSM
  * Verify both presence/not-blinking and warning/blinking states
* FCW/AEB
  * Detect activation of stock FCW/AEB (may be a good application for comma car dataset)
  * Signals for FCW/AEB actuation (this is still in development, extreme care required)
