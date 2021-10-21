## Port structure
##### interace.py
Generic interface to send and receive messages from CAN (controlsd uses this to communicate with car)
##### carcontroller.py
Builds CAN messages to send to car
##### carstate.py
Reads CAN from car and builds openpilot CarState message
##### values.py
Fingerprints and absolute limits
##### radar_interface.py
Radar interface
