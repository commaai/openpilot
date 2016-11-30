openpilot Safety
======

openpilot is an Adaptive Cruise Control and Lane Keeping Assist System. Like
other ACC and LKAS systems, openpilot requires the driver to be alert and to pay
attention at all times. We repeat, **driver alertness is necessary, but not
sufficient, for openpilot to be used safely**.

Even with an attentive driver, we must make further efforts for the system to be
safe. We have designed openpilot with two other safety considerations.

1. The vehicle must always be controllable by the driver.
2. The vehicle must not alter its trajectory too quickly for the driver to safely
   react.

To address these, we came up with two safety principles.

1. Enforced disengagements. Step on either pedal or press the cancel button to
   retake manual control of the car immediately.
  - These are hard enforced by the board, and soft enforced by the software. The
    green led on the board signifies if the board is allowing control messages.
  - Honda CAN uses both a counter and a checksum to ensure integrity and prevent
    replay of the same message.

2. Actuation limits. While the system is engaged, the actuators are constrained
   to operate within reasonable limits; the same limits used by the stock system on
   the Honda.
  - Without an interceptor, the gas is controlled by the PCM. The PCM limits
    acceleration to what is reasonable for a cruise control system.  With an
    interceptor, the gas is clipped to 60% in longcontrol.py
  - The brake is controlled by the 0x1FA CAN message. This message allows full
    braking, although the board and the software clip it to 1/4th of the max.
    This is around .3g of braking.
  - Steering is controlled by the 0xE4 CAN message. The EPS controller in the
    car limits the torque to a very small amount, so regardless of the message,
    the controller cannot jerk the wheel.
