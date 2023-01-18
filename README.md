# zenpilot

zenpilot is my passion project, a custom fork of the open-source driver assistance system, [Openpilot](https://github.com/commaai/openpilot). My goal is to focus on sane branch management strategies and UI organization to keep the software as close to stock as possible while enhancing it when necessary.

I'm currently building a community of like-minded developers who share my vision of creating a safe and user-friendly self-driving software. If you're passionate about making a difference in the world of autonomous vehicles and want to be a part of this project, I welcome you to join me.

I actively welcome pull requests, issues, and feature requests on GitHub. I believe that together, we can make this software better, safer and more user-friendly. With our easy-to-use documentation, you can quickly start making a difference.

Join me in this journey, and let's elevate the self-driving experience to new heights. Upgrade your ride with zenpilot today!

## Table of Contents

---

- [zenpilot](#zenpilot)
  - [Table of Contents](#table-of-contents)
  - [Community and Contributing](#community-and-contributing)
  - [What is openpilot?](#what-is-openpilot)
  - [Safety and Testing](#safety-and-testing)
  - [Directory Structure](#directory-structure)
  - [Licensing and Disclaimer](#licensing-and-disclaimer)
  - [Other languages](#other-languages)

## Community and Contributing

---

zenpilot is developed by [chadgauth](https://github.com/chadgauth) and by drivers like you. We welcome both pull requests and issues on [GitHub](http://github.com/chadgauth/zenpilot). Feature requests are encouraged, and integration will be faster if you share commit hashes from other forks. Check out [the contributing docs](docs/CONTRIBUTING.md) for more information.

## What is [openpilot](https://github.com/commaai/openpilot)?

---

openpilot is an open source driver assistance system that includes features such as Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW), and Lane Departure Warning (LDW) for a growing number of supported car makes, models, and model years. Also includes a camera-based Driver Monitoring (DM) feature that alerts drivers who may be distracted or asleep.

## Safety and Testing

---

zenpilot will follow all safety/tests guidelines defined in [openpilot safety documentation](https://github.com/commaai/openpilot/docs/SAFETY.md).

## Directory Structure

---

    .
    ├── cereal              # The messaging spec and libraries used for logging
    ├── common              # A collection of developed libraries and functionalities
    ├── docs                # All documentation related to the project
    ├── opendbc             # Files for interpreting data from cars
    ├── panda               # Code for communicating on Controller Area Network (CAN)
    ├── third_party         # External libraries used in the project
    ├── pyextra             # Additional Python packages
    └── system              # A collection of generic services
        ├── camerad         # Daemon that captures images from camera sensors
        ├── clocksd         # Service that broadcasts current time
        ├── hardware        # Abstraction classes for hardware
        ├── logcatd         # Systemd journal as a service
        └── proclogd        # Service that logs information from /proc
    └── selfdrive           # Code necessary for driving the car
        ├── assets          # Assets such as fonts, images, and sounds for the UI
        ├── athena          # Allows communication with the app
        ├── boardd          # Daemon that communicates with the board
        ├── car             # Car-specific code for reading states and controlling actuators
        ├── controls        # Planning and control modules
        ├── debug           # Tools for debugging and car porting
        ├── locationd       # Service for precise localization and vehicle parameter estimation
        ├── loggerd         # Service for logging and uploading car data
        ├── manager         # Daemon that starts and stops all other daemons as needed
        ├── modeld          # Driving and monitoring model runners
        ├── monitoring      # Daemon for determining driver attention
        ├── navd            # Service for turn-by-turn navigation
        ├── sensord         # Interface code for Inertial Measurement Unit (IMU)
        ├── test            # Unit tests, system tests, and a car simulator
        └── ui              # The User Interface

## Licensing and Disclaimer

---

zenpilot is released under the MIT license. Some parts of the software may be released under other licenses as specified.

By using this software, you acknowledge and agree that:

1. The software is provided "as is" and "with all faults." The developers and maintainers of this software make no representations or warranties of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, accuracy, or non-infringement.

2. The developers and maintainers of this software shall not be liable for any damages, including but not limited to, direct, indirect, special, incidental, or consequential damages, arising out of or in connection with the use or inability to use the software.

3. You are solely responsible for complying with all applicable laws, regulations, and guidelines related to the use of this software, including but not limited to, those related to self-driving cars and vehicle safety.

4. This software is intended for research and development purposes only and is not intended for use in a production environment.

5. The developers and maintainers of this software are not responsible for any injuries or damage resulting from the use of this software.

6. You shall indemnify and hold harmless the developers and maintainers of this software from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by you.

## Other languages

---

- [Japanese](README.ja.md)
