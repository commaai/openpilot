openpilot Safety
======

openpilot is an Adaptive Cruise Control (ACC) and Lane Keeping Assist (LKA) system. 
Like other ACC and LKA systems, openpilot requires the driver to be alert and to 
pay attention at all times. We repeat, **driver alertness is necessary, but not 
sufficient, for openpilot to be used safely**.

Even with an attentive driver, we must make further efforts for the system to be
safe. We have designed openpilot with two other safety considerations.

1. The driver must always be capable to immediately retake manual control of the vehicle, 
   by stepping on either pedal or by pressing the cancel button.
2. The vehicle must not alter its trajectory too quickly for the driver to safely
   react. This means that while the system is engaged, the actuators are constrained
   to operate within reasonable limits.

Following are details of the car specific safety implementations:

Honda/Acura
------

  - While the system is engaged, gas, brake and steer limits are subject to the same limits used by
    the stock system.

  - Without an interceptor, the gas is controlled by the Powertrain Control Module (PCM). 
    The PCM limits acceleration to what is reasonable for a cruise control system.  With an
    interceptor, the gas is clipped to 60%.

  - The brake is controlled by the 0x1FA CAN message. This message allows full
    braking, although the board and the software clip it to 1/4th of the max.
    This is around .3g of braking.

  - Steering is controlled by the 0xE4 CAN message. The Electronic Power Steering (EPS) 
    controller in the car limits the torque to a very small amount, so regardless of the 
    message, the controller cannot jerk the wheel.

  - Brake and gas pedal pressed signals are contained in the 0x17C CAN message. A rising edge of
    either signal triggers a disengagement, which is enforced by the board and in software. The
    green led on the board signifies if the board is allowing control messages.

  - Honda CAN uses both a counter and a checksum to ensure integrity and prevent
    replay of the same message.

Toyota/Lexus
------

  - While the system is engaged, gas, brake and steer limits are subject to the same limits used by
    the stock system.

  - With the stock Driving Support Unit (DSU) enabled, the acceleration is controlled 
    by the stock system and is subject to the stock adaptive cruise control limits. Without the
    stock DSU connected, the acceleration command is controlled by the 0x343 CAN message and its
    value is limited by the board and the software to between .3g of deceleration and .15g of
    acceleration. The acceleration command is ignored by the Engine Control Module (ECM) while the
    cruise control system is disengaged.

  - Steering torque is controlled through the 0x2E4 CAN message and it's limited by the board and in
    software to a value of -1500 and 1500. In addition, the vehicle EPS unit will not respond to
    commands outside these limits.  A steering torque rate limit is enforced by the board and in
    software so that the commanded steering torque must rise from 0 to max value no faster than
    1.5s. Commanded steering torque is limited by the board and in software to be no more than 350
    units above the actual EPS generated motor torque to ensure limited differences between
    commanded and actual torques.

  - Brake and gas pedal pressed signals are contained in the 0x224 and 0x1D2 CAN messages,
    respectively. A rising edge of either signal triggers a disengagement, which is enforced by the
    board and in software. Additionally, the cruise control system disengages on the rising edge of
    the brake pedal pressed signal.

  - The cruise control system state is contained in the 0x1D2 message. No control messages are
    allowed if the cruise control system is not active. This is enforced by the software and the
    board. The green led on the board signifies if the board is allowing control messages.
