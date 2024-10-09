# OpenPilot Glossary

## comma.ai Terms

### comma.ai

**Abbreviation:**  
None  
**Definition:**  
The company behind openpilot.

### comma API

**Abbreviation:**  
None  
**Definition:**  
Link to the documentation.

### comma connect

**Abbreviation:**  
None  
**Definition:**  
An open-source progressive web app used to view drives and interact with the device remotely.

### car harness

**Abbreviation:**  
None  
**Definition:**  
A universal interface to your car. A unique harness exists for each supported make and model. This replaces the older giraffe connector.

### comma pedal

**Abbreviation:**  
None  
**Definition:**  
A device that provides stop-and-go capability on cars that don't currently support it. This device is not officially supported by comma.ai but is supported in openpilot.

### comma points

**Abbreviation:**  
None  
**Definition:**  
Awarded for various activities on the platform, primarily for bragging rights.

### comma power

**Abbreviation:**  
CPv1, CPv2  
**Definition:**  
Uses your car's OBD-II port to power your Toyota, Bosch, or FCA giraffe. CPv1 was the initial version for use with giraffes. CPv2 uses the comma harness and has an RJ45 jack.

### comma prime

**Abbreviation:**  
None  
**Definition:**  
A subscription service from comma.ai offering specific benefits.

### comma three devkit

**Abbreviation:**  
C3, comma three  
**Definition:**  
The latest generation devkit from comma, running Ubuntu and supporting the latest openpilot releases. Has an integrated panda (dos) and supports CAN-FD vehicles with an external red panda.

### comma two devkit

**Abbreviation:**  
C2, comma two  
**Definition:**  
A smartphone running a customized version of Android and a custom case with additional cooling. Runs openpilot software and has an integrated panda (uno).

### EON devkit

**Abbreviation:**  
EON, EON Gold, EON SE  
**Definition:**  
The previous generation of the comma two devkit, without an integrated panda.

### fingerprint

**Abbreviation:**  
FPv1, FPv2  
**Definition:**  
A list of CAN bus signals unique to a vehicle, allowing openpilot to recognize the car. CAN-based FPv1 is deprecated; FPv2 uses firmware-based recognition.

### FrEON

**Abbreviation:**  
None  
**Definition:**  
"Free EON" - an open-source variant of the EON case. Files for 3D printing the case were developed by @Chase#7213.

### giraffe connector

**Abbreviation:**  
None  
**Definition:**  
An adapter board that reads buses not exposed on the main OBD-II connector, with variants for different vehicle models.

### Lane Change Assist

**Abbreviation:**  
LCA  
**Definition:**  
Activates the turn signal and gently nudges the wheel for a lane change, always requiring driver attention.

### LeEco Le Pro 3

**Abbreviation:**  
LeEco, Lepro  
**Definition:**  
The phone used in comma two and EON Gold devkits.

### LiveParameters

**Abbreviation:**  
None  
**Definition:**  
A continually updated file that stores learned calibration data for the vehicle.

### OnePlus 3T

**Abbreviation:**  
op3t  
**Definition:**  
A phone used in older EON devkits, discontinued due to supply issues. Model numbers: A3000 (US version), A3010 (Asian version).

### openpilot

**Abbreviation:**  
op  
**Definition:**  
An open-source driver assistance system developed by comma.ai.

### panda

**Abbreviation:**  
None  
**Definition:**  
A CAN bus interface available in white, grey, and black variants. Integrated pandas are found inside the comma two (uno) and comma three (dos).

### panda paw

**Abbreviation:**  
None  
**Definition:**  
A device that helps unbrick a panda.

## openpilot Terms

### big model

**Abbreviation:**  
None  
**Definition:**  
A new paradigm in model development using a larger input frame. Useful for detecting signs and lights more effectively.

### Driving Model

**Abbreviation:**  
model  
**Definition:**  
The neural network trained by comma.ai that processes inputs to provide driving-related outputs.

### End to End

**Abbreviation:**  
e2e  
**Definition:**  
An approach where the model reacts like a human driver by analyzing the entire scene without manual labeling.

### longitudinal

**Abbreviation:**  
long  
**Definition:**  
Refers to gas and brake control.

### lateral

**Abbreviation:**  
lat  
**Definition:**  
Refers to steering control.

### Model Predictive Control

**Abbreviation:**  
mpc  
**Definition:**  
An advanced method for controlling processes while satisfying constraints, used for longitudinal and lateral control.

### lead

**Abbreviation:**  
None  
**Definition:**  
A radar point selected by openpilot's driving model, used for longitudinal control with attributes like distance, speed, and acceleration.

## Driver Assistance Terms

### Adaptive Cruise Control

**Abbreviation:**  
ACC  
**Definition:**  
A cruise control system that automatically adjusts speed to maintain a safe distance from the car ahead.

### Advanced Driver-Assistance Systems

**Abbreviation:**  
ADAS  
**Definition:**  
Electronic systems that aid the driver in various driving tasks.

### Lane Keep Assist System

**Abbreviation:**  
LKAS  
**Definition:**  
A system that assists the driver in keeping the car centered within the lane.

### Lane Departure Warning System

**Abbreviation:**  
LDWS  
**Definition:**  
Warns the driver when the car unintentionally drifts out of its lane.

### Pedestrian Crash Avoidance Mitigation

**Abbreviation:**  
PCAM  
**Definition:**  
A system that uses AI to recognize pedestrians and bikes, taking safety actions when needed.

## Automotive Terms

### Controller Area Network

**Abbreviation:**  
CAN, CAN bus  
**Definition:**  
A message-based protocol for communication between ECUs in a vehicle.

### CAN-FD

**Abbreviation:**  
None  
**Definition:**  
A newer version of CAN that supports higher data rates and longer messages.

### Electronic Control Unit

**Abbreviation:**  
ECU  
**Definition:**  
An embedded system that controls one or more of a vehicle's electrical systems.

### Electric Power Steering

**Abbreviation:**  
EPS  
**Definition:**  
An electric motor assists the driver in steering the vehicle.

### On-Board Diagnostics Connector

**Abbreviation:**  
OBD-II, OBD-II port  
**Definition:**  
Gives access to the status of various vehicle subsystems. The comma power v2 uses this port to power devices and access diagnostic buses.

## Discord Terms

### Direct Message

**Abbreviation:**  
DM (PM preferred)  
**Definition:**  
A private message to an individual on Discord.

### Private Message

**Abbreviation:**  
PM  
**Definition:**  
A private message to an individual on Discord.

### Want To Buy

**Abbreviation:**  
WTB  
**Definition:**  
Used to indicate you want to buy an item (in the #for-sale channel).

### For Sale

**Abbreviation:**  
FS  
**Definition:**  
Used to indicate you want to sell an item (in the #for-sale channel).

