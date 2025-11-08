# Stimulus-Response Tests

These are example test drives that can help identify the CAN bus messaging necessary for ADAS control. Each scripted
test should be done in a separate route (ignition cycle). These tests are a guide, not necessarily exhaustive.

While testing, constant power to the comma device is highly recommended, using [comma power](https://comma.ai/shop/comma-power) if
necessary to make sure all test activity is fully captured and for ease of uploading. If constant power isn't
available, keep the ignition on for at least one minute after your test to make sure power loss doesn't result
in loss of the last minute of testing data.

## Stationary ignition-only tests, part 1

1. Ignition on, but don't start engine, remain in Park
2. Open and close each door in a defined order: driver, passenger, rear left, rear right
3. Re-enter the vehicle, close the driver's door, and fasten the driver's seatbelt
4. Slowly press and release the accelerator pedal 3 times
5. Slowly press and release the brake pedal 3 times
6. Hold the brake and move the gearshift to reverse, then neutral, then drive, then sport/eco/etc if applicable
7. Return to Park, ignition off

Brake-pressed information may show up in several messages and signals, both as on/off states and as a percentage or
pressure. It may reflect a switch on the driver's brake pedal, or a pressure-threshold state, or signals to turn on
the rear brake lights. Start by identifying all the potential signals, and confirm while driving with ACC later.

Locate signals for all four door states if possible, but some cars only expose the driver's door state on the ADAS bus.
Driver/passenger door signals may or may not change positions for LHD vs RHD cars. For cars where only the driver's
door signal is available, the same signal may follow the driver.

## Stationary ignition-only tests, part 2

1. Ignition on, but don't start engine, remain in Park
2. Press each ACC button in a defined order: main switch on/off, set, resume, cancel, accel, decel, gap adjust
3. Set the left turn signal for about five seconds
4. Operate the left turn signal one time in its touch-to-pass mode
5. Set the right turn signal for about five seconds
6. Operate the right turn signal one time in its touch-to-pass mode
7. Set the hazard / emergency indicator switch for about five seconds
8. Ignition off

Your vehicle may have a momentary-press main ACC switch or a physical toggle that remains set. Actual ACC engagement
isn't necessary for purposes of detecting the ACC button presses.

## Steering angle and steering torque tests

Power steering should be available. On ICE cars, engine RPM may be present.

1. Ignition on, start engine if applicable, remain in Park
2. Rotate the steering wheel as follows, with a few seconds pause between each step
   * Start as close to exact center as possible
   * Turn to 45 degrees right and hold
   * Turn to 90 degrees right and hold
   * Turn to 180 degrees right and hold
   * Turn to full lock right and hold, with firm pressure against lock
   * Release the wheel and allow it to bounce back slightly from lock
   * Turn to 180 degrees left and hold
   * Return to center and release
3. Ignition off

Performing the full test to the right, followed by an abbreviated test to the left, helps give additional confirmation
of signal scale, and sign/direction for both the steering wheel angle and driver input torque signals.

## Low speed / parking lot driving tests

Before this test, drive to a place like an empty parking lot where you are free to drive in a series of curves.

1. Ignition on, start engine if applicable, prepare to drive
2. Slowly (10-20mph at most) drive a figure-8 if possible, or at least one sharp left and one sharp right.
3. Come to a complete stop
4. When and where safe, drive in reverse for a short distance (10-15 feet)
5. Park the car in a safe place, ignition off

## High speed / highway driving tests

Select a place and time where you can safely set cruise control at normal travel speeds with little interference from
traffic ahead, and safely test the response of your factory lane guidance system.

1. Ignition on, start engine if applicable, prepare to drive
2. When safely able, engage adaptive cruise control below 50 mph
3. When safely able, use the ACC buttons to accelerate to 50mph, then 55mph, then 60mph
4. Disengage adaptive cruise
5. When safely able, allow your factory lane guidance to prevent lane departures, 2-3 times on both the left and right

The series of setpoints can be adjusted to local traffic regulations, and of course metric units. The specific cruise
setpoints are useful for locating the ACC HUD signals later, and confirming their precise scaling. When the car reaches
and holds the setpoint, that can also provide additional confirmation of wheel speed scaling.
