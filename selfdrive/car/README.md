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
