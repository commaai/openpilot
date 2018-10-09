openpilot Safety
======

openpilot is an Adaptive Cruise Control (ACC) and Lane Keeping Assist (LKA) system. 
Like other ACC and LKA systems, openpilot requires the driver to be alert and to 
pay attention at all times. We repeat, **driver alertness is necessary, but not 
sufficient, for openpilot to be used safely**.

In order to enforce driver alertness, openpilot includes a driver monitoring feature
that alerts the driver when distracted.

However, even with an attentive driver, we must make further efforts for the system to be
safe. We have designed openpilot with two other safety considerations.

1. The driver must always be capable to immediately retake manual control of the vehicle, 
   by stepping on either pedal or by pressing the cancel button.
2. The vehicle must not alter its trajectory too quickly for the driver to safely
   react. This means that while the system is engaged, the actuators are constrained
   to operate within reasonable limits.

Following are details of the car specific safety implementations:

Honda/Acura
------

  - While the system is engaged, gas, brake and steer commands are subject to the same limits used by
    the stock system.

  - Without an interceptor, the gas is controlled by the Powertrain Control Module (PCM). 
    The PCM limits acceleration to what is reasonable for a cruise control system.  With an
    interceptor, the gas is clipped to 60%.

  - The brake is controlled by the 0x1FA CAN message. This message allows full
    braking, although the panda firmware and openpilot clip it to 1/4th of the max.
    This is approximately 0.3g of braking.

  - Steering is controlled by the 0xE4 CAN message. The Electronic Power Steering (EPS) 
    controller in the car limits the torque to a very small amount, so regardless of the 
    message, the controller cannot jerk the wheel.

  - Brake and gas pedal pressed signals are contained in the 0x17C CAN message. A rising edge of
    either signals triggers a disengagement, which is enforced by the panda firmware and by openpilot. The
    white led on the panda signifies if the panda is allowing control messages.

  - Honda CAN uses both a counter and a checksum to ensure integrity and prevent
    replay of the same message.

Toyota/Lexus
------

  - While the system is engaged, gas, brake and steer commands are subject to the same limits used by
    the stock system.

  - With the stock Driving Support Unit (DSU) connected (or in DSU-less models like Camry and C-HR),
    the acceleration is controlled by the stock system and is subject to the stock adaptive cruise
    control limits. Without the stock DSU connected, the acceleration command is controlled by the
    0x343 CAN message and its value is limited between .3g of deceleration and .15g of acceleration
    by the panda firmware and by openpilot. The acceleration command is ignored by the Engine Control
    Module (ECM) while the cruise control system is disengaged.

  - Steering torque is controlled through the 0x2E4 CAN message and it's limited by the panda firmware and by
    openpilot to a value between -1500 and 1500. In addition, the vehicle EPS unit will not respond to
    commands outside these limits.  A steering torque rate limit is enforced by the panda firmware and by
    openpilot, so that the commanded steering torque must rise from 0 to max value no faster than
    1.5s. Commanded steering torque is limited by the panda firmware and by openpilot to be no more than 350
    units above the actual EPS generated motor torque to ensure limited differences between
    commanded and actual torques.

  - Brake and gas pedal pressed signals are contained in the 0x224 and 0x1D2 CAN messages,
    respectively. A rising edge of either signals triggers a disengagement, which is enforced by the
    panda firmware and by openpilot. Additionally, the cruise control system disengages on the rising edge of
    the brake pedal pressed signal.

  - The cruise control system state is contained in the 0x1D2 message. No control messages are
    allowed if the cruise control system is not active. This is enforced by openpilot and the
    panda firmware. The white led on the panda signifies if the panda is allowing control messages.

GM/Chevrolet
------

  - While the system is engaged, gas, brake and steer commands are subject to the same limits used by
    the stock system.

  - The gas and regen are controlled by the 0x2CB message and it's limited by the panda firmware and by
    openpilot to a value between 1404 and 3072. the minimum value correspond to a mild decel due to regen,
    while 3072 correspond to approximately 0.18g of acceleration from stop.

  - The friction brakes are controlled by the 0x315 message and its value is limited by the panda firmware
    and openpilot to 350. This is approximately 0.3g of braking.

  - Steering torque is controlled through the 0x180 CAN message and it's limited by the panda firmware and by
    openpilot to a value between -255 and 255. In addition, the vehicle EPS unit will not fault when
    commands outside these limits.  A steering torque rate limit is enforced by the panda firmware and by
    openpilot, so that the commanded steering torque must rise from 0 to max value no faster than
    0.75s. Commanded steering torque is gradually limited by the panda firmware and by openpilot if the driver's
    torque exceeds 12 units in the opposite dicrection to ensure limited applied torque against the
    driver's will.

  - Brake pedal and gas pedal potentiometer signals are contained in the 0xF1 and 0x1A1 CAN messages,
    respectively. A rising edge of either signals triggers a disengagement, which is enforced by the
    panda firmware and by openpilot. Additionally, the cruise control system disengages on the rising edge of
    the brake pedal pressed signal. The regen paddle pressed signal is in the 0xBD message. When the
    regen paddle is pressed, a disengagement is enforced by both the firmware and by openpilot.

  - GM CAN uses both a counter and a checksum to ensure integrity and prevent
    replay of the same message.

**Extra note"**: comma.ai strongly discourages the use of openpilot forks with safety code either missing or
  not fully meeting the above requirements.
