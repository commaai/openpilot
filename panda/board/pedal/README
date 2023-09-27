# pedal

This is the firmware for the comma pedal.

The comma pedal is a gas pedal interceptor for Honda/Acura and Toyota/Lexus. It allows you to "virtually" press the pedal and borrows a lot from panda.

== Test Plan ==

* Startup
** Confirm STATE_FAULT_STARTUP
* Timeout
** Send value
** Confirm value is output
** Stop sending messages
** Confirm value is passthru after 100ms
** Confirm STATE_FAULT_TIMEOUT
* Random values
** Send random 6 byte messages
** Confirm random values cause passthru
** Confirm STATE_FAULT_BAD_CHECKSUM
* Same message lockout
** Send same message repeated
** Confirm timeout behavior
* Don't set enable
** Confirm no output
* Set enable and values
** Confirm output

