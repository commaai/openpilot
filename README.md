[![](https://i.imgur.com/UetIFyH.jpg)](#)

Table of Contents
=======================

* [What is openpilot?](#what-is-openpilot)
* [Integration with Stock Features](#integration-with-stock-features)
* [Supported Hardware](#supported-hardware)
* [Supported Cars](#supported-cars)
* [Community Maintained Cars and Features](#community-maintained-cars-and-features)
* [Installation Instructions](#installation-instructions)
* [Limitations of openpilot ALC and LDW](#limitations-of-openpilot-alc-and-ldw)
* [Limitations of openpilot ACC and FCW](#limitations-of-openpilot-acc-and-fcw)
* [Limitations of openpilot DM](#limitations-of-openpilot-dm)
* [User Data and comma Account](#user-data-and-comma-account)
* [Safety and Testing](#safety-and-testing)
* [Testing on PC](#testing-on-pc)
* [Community and Contributing](#community-and-contributing)
* [Directory Structure](#directory-structure)
* [Licensing](#licensing)

---

What is openpilot?
------

[openpilot](http://github.com/commaai/openpilot) is an open source driver assistance system. Currently, openpilot performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW) and Lane Departure Warning (LDW) for a growing variety of supported [car makes, models and model years](#supported-cars). In addition, while openpilot is engaged, a camera based Driver Monitoring (DM) feature alerts distracted and asleep drivers.

<table>
  <tr>
    <td><a href="https://www.youtube.com/watch?v=mgAbfr42oI8" title="YouTube" rel="noopener"><img src="https://i.imgur.com/kAtT6Ei.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=394rJKeh76k" title="YouTube" rel="noopener"><img src="https://i.imgur.com/lTt8cS2.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=1iNOc3cq8cs" title="YouTube" rel="noopener"><img src="https://i.imgur.com/ANnuSpe.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=Vr6NgrB-zHw" title="YouTube" rel="noopener"><img src="https://i.imgur.com/Qypanuq.png"></a></td>
  </tr>
  <tr>
    <td><a href="https://www.youtube.com/watch?v=Ug41KIKF0oo" title="YouTube" rel="noopener"><img src="https://i.imgur.com/3caZ7xM.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=NVR_CdG1FRg" title="YouTube" rel="noopener"><img src="https://i.imgur.com/bAZOwql.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=tkEvIdzdfUE" title="YouTube" rel="noopener"><img src="https://i.imgur.com/EFINEzG.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=_P-N1ewNne4" title="YouTube" rel="noopener"><img src="https://i.imgur.com/gAyAq22.png"></a></td>
  </tr>
</table>

Integration with Stock Features
------

In all supported cars:
* Stock Lane Keep Assist (LKA) and stock ALC are replaced by openpilot ALC, which only functions when openpilot is engaged by the user.
* Stock LDW is replaced by openpilot LDW.

Additionally, on specific supported cars (see ACC column in [supported cars](#supported-cars)):
* Stock ACC is replaced by openpilot ACC.
* openpilot FCW operates in addition to stock FCW.

openpilot should preserve all other vehicle's stock features, including, but are not limited to: FCW, Automatic Emergency Braking (AEB), auto high-beam, blind spot warning, and side collision warning.

Supported Hardware
------

At the moment, openpilot supports the [EON DevKit](https://comma.ai/shop/products/eon-dashcam-devkit). A [car harness](https://comma.ai/shop/products/car-harness) is recommended to connect the EON to the car. In the future, we'd like to support other platforms as well.

Supported Cars
------

| Make      | Model (US Market Reference)        | Supported Package | ACC              | No ACC accel below | No ALC below |
| ----------| -----------------------------------| ------------------| -----------------| -------------------| -------------|
| Acura     | ILX 2016-18                        | AcuraWatch Plus   | openpilot        | 25mph<sup>6</sup>  | 25mph        |
| Acura     | RDX 2016-18                        | AcuraWatch Plus   | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Chrysler  | Pacifica 2017-18<sup>1</sup>       | Adaptive Cruise   | Stock            | 0mph               | 9mph         |
| Chrysler  | Pacifica Hybrid 2017-18<sup>1</sup>| Adaptive Cruise   | Stock            | 0mph               | 9mph         |
| Chrysler  | Pacifica Hybrid 2019<sup>1</sup>   | Adaptive Cruise   | Stock            | 0mph               | 39mph        |
| Honda     | Accord 2018-19                     | All               | Stock            | 0mph               | 3mph         |
| Honda     | Accord Hybrid 2018-19              | All               | Stock            | 0mph               | 3mph         |
| Honda     | Civic Sedan/Coupe 2016-18          | Honda Sensing     | openpilot        | 0mph               | 12mph        |
| Honda     | Civic Sedan/Coupe 2019             | Honda Sensing     | Stock            | 0mph               | 2mph         |
| Honda     | Civic Hatchback 2017-19            | Honda Sensing     | Stock            | 0mph               | 12mph        |
| Honda     | CR-V 2015-16                       | Touring           | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Honda     | CR-V 2017-19                       | Honda Sensing     | Stock            | 0mph               | 12mph        |
| Honda     | CR-V Hybrid 2017-2019              | Honda Sensing     | Stock            | 0mph               | 12mph        |
| Honda     | Fit 2018-19                        | Honda Sensing     | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Honda     | Odyssey 2018-20                    | Honda Sensing     | openpilot        | 25mph<sup>6</sup>  | 0mph         |
| Honda     | Passport 2019                      | All               | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Honda     | Pilot 2016-18                      | Honda Sensing     | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Honda     | Pilot 2019                         | All               | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Honda     | Ridgeline 2017-19                  | Honda Sensing     | openpilot        | 25mph<sup>6</sup>  | 12mph        |
| Hyundai   | Santa Fe 2019<sup>2</sup>          | All               | Stock            | 0mph               | 0mph         |
| Hyundai   | Elantra 2017-19<sup>2</sup>        | SCC + LKAS        | Stock            | 19mph              | 34mph        |
| Hyundai   | Genesis 2018<sup>2</sup>           | All               | Stock            | 19mph              | 34mph        |
| Jeep      | Grand Cherokee 2016-18<sup>1</sup> | Adaptive Cruise   | Stock            | 0mph               | 9mph         |
| Jeep      | Grand Cherokee 2019<sup>1</sup>    | Adaptive Cruise   | Stock            | 0mph               | 39mph        |
| Kia       | Optima 2019<sup>2</sup>            | SCC + LKAS        | Stock            | 0mph               | 0mph         |
| Kia       | Sorento 2018<sup>2</sup>           | All               | Stock            | 0mph               | 0mph         |
| Kia       | Stinger 2018<sup>2</sup>           | SCC + LKAS        | Stock            | 0mph               | 0mph         |
| Lexus     | CT Hybrid 2017-18                  | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Lexus     | ES Hybrid 2019                     | All               | openpilot        | 0mph               | 0mph         |
| Lexus     | RX Hybrid 2016-19                  | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Lexus     | IS 2017-2019                       | All               | Stock            | 22mph              | 0mph         |
| Lexus     | IS Hybrid 2017                     | All               | Stock            | 0mph               | 0mph         |
| Subaru    | Crosstrek 2018-19                  | EyeSight          | Stock            | 0mph               | 0mph         |
| Subaru    | Impreza 2019-20                    | EyeSight          | Stock            | 0mph               | 0mph         |
| Toyota    | Avalon 2016                        | TSS-P             | Stock<sup>5</sup>| 20mph<sup>6</sup>  | 0mph         |
| Toyota    | Avalon 2017-18                     | All               | Stock<sup>5</sup>| 20mph<sup>6</sup>  | 0mph         |
| Toyota    | Camry 2018-19                      | All               | Stock            | 0mph<sup>3</sup>   | 0mph         |
| Toyota    | Camry Hybrid 2018-19               | All               | Stock            | 0mph<sup>3</sup>   | 0mph         |
| Toyota    | C-HR 2017-19                       | All               | Stock            | 0mph               | 0mph         |
| Toyota    | C-HR Hybrid 2017-19                | All               | Stock            | 0mph               | 0mph         |
| Toyota    | Corolla 2017-19                    | All               | Stock<sup>5</sup>| 20mph<sup>6</sup>  | 0mph         |
| Toyota    | Corolla 2020                       | All               | openpilot        | 0mph               | 0mph         |
| Toyota    | Corolla Hatchback 2019             | All               | openpilot        | 0mph               | 0mph         |
| Toyota    | Corolla Hybrid 2020                | All               | openpilot        | 0mph               | 0mph         |
| Toyota    | Highlander 2017-19                 | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Highlander Hybrid 2017-19          | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Prius 2016                         | TSS-P             | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Prius 2017-19                      | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Prius Prime 2017-20                | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Rav4 2016                          | TSS-P             | Stock<sup>5</sup>| 20mph<sup>6</sup>  | 0mph         |
| Toyota    | Rav4 2017-18                       | All               | Stock<sup>5</sup>| 20mph<sup>6</sup>  | 0mph         |
| Toyota    | Rav4 2019                          | All               | openpilot        | 0mph               | 0mph         |
| Toyota    | Rav4 Hybrid 2016                   | TSS-P             | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Rav4 Hybrid 2017-18                | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Toyota    | Sienna 2018                        | All               | Stock<sup>5</sup>| 0mph               | 0mph         |
| Volkswagen| Golf 2016-19<sup>4</sup>           | Driver Assistance | Stock            | 0mph               | 0mph         |

<sup>1</sup>Requires a [panda](https://comma.ai/shop/products/panda-obd-ii-dongle) and [FCA giraffe](https://comma.ai/shop/products/giraffe) <br />
<sup>2</sup>Requires a [panda](https://comma.ai/shop/products/panda-obd-ii-dongle) and open sourced [Hyundai giraffe](https://github.com/commaai/neo/tree/master/giraffe/hyundai), designed for the 2019 Sante Fe; pinout may differ for other Hyundai and Kia models. <br />
<sup>3</sup>28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control. <br />
<sup>4</sup>Requires a [custom connector](https://community.comma.ai/wiki/index.php/Volkswagen#Integration_at_R242_Camera) for the [car harness](https://comma.ai/shop/products/car-harness) <br />

Community Maintained Cars and Features
------

| Make      | Model (US Market Reference)        | Supported Package | ACC              | No ACC accel below | No ALC below |
| ----------| -----------------------------------| ------------------| -----------------| -------------------| -------------|
| Buick     | Regal 2018<sup>7</sup>             | Adaptive Cruise   | openpilot        | 0mph               | 7mph         |
| Chevrolet | Malibu 2017<sup>7</sup>            | Adaptive Cruise   | openpilot        | 0mph               | 7mph         |
| Chevrolet | Volt 2017-18<sup>7</sup>           | Adaptive Cruise   | openpilot        | 0mph               | 7mph         |
| Cadillac  | ATS 2018<sup>7</sup>               | Adaptive Cruise   | openpilot        | 0mph               | 7mph         |
| GMC       | Acadia Denali 2018<sup>7</sup>     | Adaptive Cruise   | openpilot        | 0mph               | 7mph         |
| Holden    | Astra 2017<sup>7</sup>             | Adaptive Cruise   | openpilot        | 0mph               | 7mph         |

<sup>5</sup>When disconnecting the Driver Support Unit (DSU), openpilot ACC will replace stock ACC. For DSU locations, see [Toyota Wiki page](https://community.comma.ai/wiki/index.php/Toyota). ***NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).*** <br />
<sup>6</sup>[Comma Pedal](https://community.comma.ai/wiki/index.php/Comma_Pedal) is used to provide stop-and-go capability to some of the openpilot-supported cars that don't currently support stop-and-go. Here is how to [build a Comma Pedal](https://medium.com/@jfrux/comma-pedal-building-with-macrofab-6328bea791e8). ***NOTE: The Comma Pedal is not officially supported by [comma](https://comma.ai).*** <br />
<sup>7</sup>Requires a [panda](https://comma.ai/shop/products/panda-obd-ii-dongle) and [community built giraffe](https://zoneos.com/volt/). ***NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).*** <br />

Community Maintained Cars and Features are not verified by comma to meet our [safety model](SAFETY.md). Be extra cautious using them. They are only available after enabling the toggle in `Settings->Developer->Enable Community Features`.

Installation Instructions
------

Install openpilot on a EON by entering ``https://openpilot.comma.ai`` during the installer setup.

Follow this [video instructions](https://youtu.be/3nlkomHathI) to properly mount the EON on the windshield. Note: openpilot features an automatic pose calibration routine and openpilot performance should not be affected by small pitch and yaw misalignments caused by imprecise EON mounting.

Before placing the device on your windshield, check the state and local laws and ordinances where you drive. Some state laws prohibit or restrict the placement of objects on the windshield of a motor vehicle.

You will be able to engage openpilot after reviewing the onboarding screens and finishing the calibration procedure.

Limitations of openpilot ALC and LDW
------

openpilot ALC and openpilot LDW do not automatically drive the vehicle or reduce the amount of attention that must be paid to operate your vehicle. The driver must always keep control of the steering wheel and be ready to correct the openpilot ALC action at all times.

Many factors can impact the performance of openpilot ALC and openpilot LDW, causing them to be unable to function as intended. These include, but are not limited to:

* Poor visibility (heavy rain, snow, fog, etc.) or weather conditions that may interfere with sensor operation.
* The road facing camera is obstructed, covered or damaged by mud, ice, snow, etc.
* Obstruction caused by applying excessive paint or adhesive products (such as wraps, stickers, rubber coating, etc.) onto the vehicle.
* The EON is mounted incorrectly.
* When in sharp curves, like on-off ramps, intersections etc...; openpilot is designed to be limited in the amount of steering torque it can produce.
* In the presence of restricted lanes or construction zones.
* When driving on highly banked roads or in presence of strong cross-wind.
* Extremely hot or cold temperatures.
* Bright light (due to oncoming headlights, direct sunlight, etc.).
* Driving on hills, narrow, or winding roads.

The list above does not represent an exhaustive list of situations that may interfere with proper operation of openpilot components. It is the driver's responsibility to be in control of the vehicle at all times.

Limitations of openpilot ACC and FCW
------

openpilot ACC and openpilot FCW are not systems that allow careless or inattentive driving. It is still necessary for the driver to pay close attention to the vehicle’s surroundings and to be ready to re-take control of the gas and the brake at all times.

Many factors can impact the performance of openpilot ACC and openpilot FCW, causing them to be unable to function as intended. These include, but are not limited to:

* Poor visibility (heavy rain, snow, fog, etc.) or weather conditions that may interfere with sensor operation.
* The road facing camera or radar are obstructed, covered, or damaged by mud, ice, snow, etc.
* Obstruction caused by applying excessive paint or adhesive products (such as wraps, stickers, rubber coating, etc.) onto the vehicle.
* The EON is mounted incorrectly.
* Approaching a toll booth, a bridge or a large metal plate.
* When driving on roads with pedestrians, cyclists, etc...
* In presence of traffic signs or stop lights, which are not detected by openpilot at this time.
* When the posted speed limit is below the user selected set speed. openpilot does not detect speed limits at this time.
* In presence of vehicles in the same lane that are not moving.
* When abrupt braking maneuvers are required. openpilot is designed to be limited in the amount of deceleration and acceleration that it can produce.
* When surrounding vehicles perform close cut-ins from neighbor lanes.
* Driving on hills, narrow, or winding roads.
* Extremely hot or cold temperatures.
* Bright light (due to oncoming headlights, direct sunlight, etc.).
* Interference from other equipment that generates radar waves.

The list above does not represent an exhaustive list of situations that may interfere with proper operation of openpilot components. It is the driver's responsibility to be in control of the vehicle at all times.

Limitations of openpilot DM
------

openpilot DM should not be considered an exact measurements of the status of alertness of the driver.

Many factors can impact the performance of openpilot DM, causing it to be unable to function as intended. These include, but are not limited to:

* Low light conditions, such as driving at night or in dark tunnels.
* Bright light (due to oncoming headlights, direct sunlight, etc.).
* The driver face is partially or completely outside field of view of the driver facing camera.
* Right hand driving vehicles.
* The driver facing camera is obstructed, covered, or damaged.

The list above does not represent an exhaustive list of situations that may interfere with proper operation of openpilot components. A driver should not rely on openpilot DM to assess their level of attention.

User Data and comma Account
------

By default, openpilot uploads the driving data to our servers. You can also access your data by pairing with the comma connect app ([iOS](https://apps.apple.com/us/app/comma-connect/id1456551889), [Android](https://play.google.com/store/apps/details?id=ai.comma.connect&hl=en_US)). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver facing camera is only logged if you explicitly opt-in in settings. The microphone is not recorded.

By using openpilot, you agree to [our Privacy Policy](https://my.comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.

Safety and Testing
----

* openpilot observes ISO26262 guidelines, see [SAFETY.md](SAFETY.md) for more detail.
* openpilot has software in the loop [tests](run_docker_tests.sh) that run on every commit.
* The safety model code lives in panda and is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
* panda has software in the loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
* Internally, we have a hardware in the loop Jenkins test suite that builds and unit tests the various processes.
* panda has additional hardware in the loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
* We run the latest openpilot in a testing closet containing 10 EONs continuously replaying routes.

Testing on PC
------

Check out [openpilot-tools](https://github.com/commaai/openpilot-tools): lots of tools you can use to replay driving data, test and develop openpilot from your pc.

Community and Contributing
------

openpilot is developed by [comma](https://comma.ai/) and by users like you. We welcome both pull requests and issues on [GitHub](http://github.com/commaai/openpilot). Bug fixes and new car ports are encouraged.

You can add support for your car by following guides we have written for [Brand](https://medium.com/@comma_ai/how-to-write-a-car-port-for-openpilot-7ce0785eda84) and [Model](https://medium.com/@comma_ai/openpilot-port-guide-for-toyota-models-e5467f4b5fe6) ports. Generally, a car with adaptive cruise control and lane keep assist is a good candidate. [Join our Discord](https://discord.comma.ai) to discuss car ports: most car makes have a dedicated channel.

Want to get paid to work on openpilot? [comma is hiring](https://comma.ai/jobs/). We also have a [bounty program](https://comma.ai/bounties.html).

And [follow us on Twitter](https://twitter.com/comma_ai).

Directory Structure
------
    .
    ├── apk                 # The apk files used for the UI
    ├── cereal              # The messaging spec and libs used for all logs on EON
    ├── common              # Library like functionality we've developed here
    ├── installer/updater   # Manages auto-updates of openpilot
    ├── opendbc             # Files showing how to interpret data from cars
    ├── panda               # Code used to communicate on CAN
    ├── phonelibs           # Libraries used on EON
    ├── pyextra             # Libraries used on EON
    └── selfdrive           # Code needed to drive the car
        ├── assets          # Fonts and images for UI
        ├── athena          # Allows communication with the app
        ├── boardd          # Daemon to talk to the board
        ├── camerad         # Driver to capture images from the camera sensors
        ├── car             # Car specific code to read states and control actuators
        ├── common          # Shared C/C++ code for the daemons
        ├── controls        # Perception, planning and controls
        ├── debug           # Tools to help you debug and do car ports
        ├── locationd       # Soon to be home of precise location
        ├── logcatd         # Android logcat as a service
        ├── loggerd         # Logger and uploader of car data
        ├── modeld          # Driving model runner
        ├── monitoringd     # Driver monitoring model runner
        ├── proclogd        # Logs information from proc
        ├── sensord         # IMU / GPS interface code
        ├── tests           # Unit tests, system tests and a car simulator
        └── ui              # The UI

To understand how the services interact, see `cereal/service_list.yaml`.

Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
