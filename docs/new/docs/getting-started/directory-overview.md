# Directory Overview
This is a quick reference of all the different folders openpilot has.

```
.
├── cereal              # The messaging spec and libs used for all logs
├── common              # Library like functionality we've developed here
├── docs                # Documentation
├── opendbc             # Files showing how to interpret data from cars
├── panda               # Code used to communicate on CAN
├── third_party         # External libraries
└── system              # Generic services
    ├── camerad         # Driver to capture images from the camera sensors
    ├── clocksd         # Broadcasts current time
    ├── hardware        # Hardware abstraction classes
    ├── logcatd         # systemd journal as a service
    ├── loggerd         # Logger and uploader of car data
    ├── proclogd        # Logs information from /proc
    ├── sensord         # IMU interface code
    └── ubloxd          # u-blox GNSS module interface code
└── selfdrive           # Code needed to drive the car
    ├── assets          # Fonts, images, and sounds for UI
    ├── athena          # Allows communication with the app
    ├── boardd          # Daemon to talk to the board
    ├── car             # Car specific code to read states and control actuators
    ├── controls        # Planning and controls
    ├── debug           # Tools to help you debug and do car ports
    ├── locationd       # Precise localization and vehicle parameter estimation
    ├── manager         # Daemon that starts/stops all other daemons as needed
    ├── modeld          # Driving and monitoring model runners
    ├── monitoring      # Daemon to determine driver attention
    ├── navd            # Turn-by-turn navigation
    ├── test            # Unit tests, system tests, and a car simulator
    └── ui              # The UI
```

*Found this in Sunny's repo https://github.com/sunnyhaibin/openpilot-1