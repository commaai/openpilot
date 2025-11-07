# Stimulus-Response Tests

These are example test drives that can help identify the CAN bus messaging necessary for ADAS control. Each scripted
test should be done in a separate route (ignition cycle).

Constant power to the comma device is highly recommended, using [comma power](https://comma.ai/shop/comma-power) if
necessary, to make sure all test activity is fully captured and for ease of uploading. If constant power isn't
available, keep the ignition on for at least one minute after your test to make sure the results are saved.

## Stationary ignition-only tests

Identify the signals for the accelerator pedal, brake pedal, cruise button, and door-open states.

1. Ignition on, but don't start engine, remain in Park
2. Slowly press and release the accelerator pedal 3 times
3. Slowly press and release the brake pedal 3 times
4. Open and close each door in a defined order driver, passenger, rear left, rear right
5. Press each ACC button in a defined order: main switch on/off, set, cancel, accel, coast, gap adjust
6. Ignition off

Brake-pressed information may show up in several messages and signals, both as on/off states and as a percentage or
pressure. It may reflect a switch on the driver's brake pedal, or a pressure-threshold state, or signals to turn on
the rear brake lights. Start by identifying all the potential signals, and confirm while driving with ACC later.

Locate signals for all four door states if possible, but some cars only expose the driver's door state on the ADAS bus.

## Steering angle and steering torque signals

Power steering should be available. On ICE cars, engine RPM may be present.

1. Ignition on, start engine if applicable, remain in Park
2. Rotate the steering wheel as follows, with a few seconds pause between each step
   1. Start as close to exact center as possible
   2. Turn to 45 degrees right and hold
   3. Turn to 90 degrees right and hold
   4. Turn to 180 degrees right and hold
   5. Turn to full lock right and hold, with firm pressure against lock
   6. Release the wheel and allow it to bounce back slightly from lock
   7. Turn to 180 degrees left and hold
   8. Return to center and release
3. Ignition off
