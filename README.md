[![](https://i.imgur.com/UetIFyH.jpg)](#)

Welcome to openpilot
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent. Currently, it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for selected Honda, Toyota, Acura, Lexus, Chevrolet, Hyundai, Kia. It's about on par with Tesla Autopilot and GM Super Cruise, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

The openpilot codebase has been written to be concise and to enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

Table of Contents
=======================

* [Community](#community)
* [Hardware](#hardware)
* [Supported Cars](#supported-cars)
* [Community Maintained Cars](#community-maintained-cars)
* [In Progress Cars](#in-progress-cars)
* [How can I add support for my car?](#how-can-i-add-support-for-my-car)
* [Directory structure](#directory-structure)
* [User Data / chffr Account / Crash Reporting](#user-data--chffr-account--crash-reporting)
* [Testing on PC](#testing-on-pc)
* [Contributing](#contributing)
* [Licensing](#licensing)

---

Community
------

openpilot is developed by [comma.ai](https://comma.ai/) and users like you.

We have a [Twitter you should follow](https://twitter.com/comma_ai).

Also, we have a several thousand people community on [Discord](https://discord.comma.ai).

<table>
  <tr>
    <td><a href="https://www.youtube.com/watch?v=ICOIin4p70w" title="YouTube" rel="noopener"><img src="https://i.imgur.com/gBTo7yB.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=1zCtj3ckGFo" title="YouTube" rel="noopener"><img src="https://i.imgur.com/gNhhcep.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=Qd2mjkBIRx0" title="YouTube" rel="noopener"><img src="https://i.imgur.com/tFnSexp.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=ju12vlBm59E" title="YouTube" rel="noopener"><img src="https://i.imgur.com/3BKiJVy.png"></a></td>
  </tr>
  <tr>
    <td><a href="https://www.youtube.com/watch?v=Z5VY5FzgNt4" title="YouTube" rel="noopener"><img src="https://i.imgur.com/3I9XOK2.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=blnhZC7OmMg" title="YouTube" rel="noopener"><img src="https://i.imgur.com/f9IgX6s.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=iRkz7FuJsA8" title="YouTube" rel="noopener"><img src="https://i.imgur.com/Vo5Zvmn.png"></a></td>
    <td><a href="https://www.youtube.com/watch?v=IHjEqAKDqjM" title="YouTube" rel="noopener"><img src="https://i.imgur.com/V9Zd81n.png"></a></td>
  </tr>
</table>

Hardware
------

At the moment openpilot supports the [EON Dashcam DevKit](https://comma.ai/shop/products/eon-dashcam-devkit). A [panda](https://shop.comma.ai/products/panda-obd-ii-dongle) and a [giraffe](https://comma.ai/shop/products/giraffe/) are recommended tools to interface the EON with the car. We'd like to support other platforms as well.

Install openpilot on a neo device by entering ``https://openpilot.comma.ai`` during NEOS setup.

Supported Cars
------

| Make                 | Model                    | Supported Package    | Lateral | Longitudinal   | No Accel Below   | No Steer Below | Giraffe           |
| ---------------------| -------------------------| ---------------------| --------| ---------------| -----------------| ---------------|-------------------|
| Acura                | ILX 2016-17              | AcuraWatch Plus      | Yes     | Yes            | 25mph<sup>1</sup>| 25mph          | Nidec             |
| Acura                | RDX 2018                 | AcuraWatch Plus      | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Nidec             |
| Buick<sup>3</sup>    | Regal 2018               | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           | Custom<sup>7</sup>|
| Chevrolet<sup>3</sup>| Malibu 2017              | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           | Custom<sup>7</sup>|
| Chevrolet<sup>3</sup>| Volt 2017-18             | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           | Custom<sup>7</sup>|
| Cadillac<sup>3</sup> | ATS 2018                 | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           | Custom<sup>7</sup>|
| Chrysler             | Pacifica 2018            | Adaptive Cruise      | Yes     | Stock          | 0mph             | 9mph           | FCA               |
| Chrysler             | Pacifica Hybrid 2017-18  | Adaptive Cruise      | Yes     | Stock          | 0mph             | 9mph           | FCA               |
| Chrysler             | Pacifica Hybrid 2019     | Adaptive Cruise      | Yes     | Stock          | 0mph             | 39mph          | FCA               |
| GMC<sup>3</sup>      | Acadia Denali 2018       | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           | Custom<sup>7</sup>|
| Holden<sup>3</sup>   | Astra 2017               | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           | Custom<sup>7</sup>|
| Honda                | Accord 2018              | All                  | Yes     | Stock          | 0mph             | 3mph           | Bosch             |
| Honda                | Civic Sedan/Coupe 2016-18| Honda Sensing        | Yes     | Yes            | 0mph             | 12mph          | Nidec             |
| Honda                | Civic Sedan/Coupe 2019   | Honda Sensing        | Yes     | Stock          | 0mph             | 2mph           | Bosch             |
| Honda                | Civic Hatchback 2017-19  | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          | Bosch             |
| Honda                | CR-V 2015-16             | Touring              | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Nidec             |
| Honda                | CR-V 2017-19             | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          | Bosch             |
| Honda                | CR-V Hybrid 2017-2019    | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          | Bosch             |
| Honda                | Odyssey 2018-19          | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 0mph           | Inverted Nidec    |
| Honda                | Passport 2019            | All                  | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Inverted Nidec    |
| Honda                | Pilot 2016-18            | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Nidec             |
| Honda                | Pilot 2019               | All                  | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Inverted Nidec    |
| Honda                | Ridgeline 2017-19        | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Nidec             |
| Hyundai              | Santa Fe 2019            | All                  | Yes     | Stock          | 0mph             | 0mph           | Custom<sup>6</sup>|
| Hyundai              | Elantra 2017             | SCC + LKAS           | Yes     | Stock          | 19mph            | 34mph          | Custom<sup>6</sup>|
| Hyundai              | Genesis 2018             | All                  | Yes     | Stock          | 19mph            | 34mph          | Custom<sup>6</sup>|
| Jeep                 | Grand Cherokee 2017-18   | Adaptive Cruise      | Yes     | Stock          | 0mph             | 9mph           | FCA               |
| Jeep                 | Grand Cherokee 2019      | Adaptive Cruise      | Yes     | Stock          | 0mph             | 39mph          | FCA               |
| Kia                  | Optima 2019              | SCC + LKAS           | Yes     | Stock          | 0mph             | 0mph           | Custom<sup>6</sup>|
| Kia                  | Sorento 2018             | All                  | Yes     | Stock          | 0mph             | 0mph           | Custom<sup>6</sup>|
| Kia                  | Stinger 2018             | SCC + LKAS           | Yes     | Stock          | 0mph             | 0mph           | Custom<sup>6</sup>|
| Lexus                | ES Hybrid 2019           | All                  | Yes     | Yes            | 0mph             | 0mph           | Toyota            |
| Lexus                | RX Hybrid 2016-19        | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |
| Subaru               | Crosstrek 2018           | EyeSight             | Yes     | Stock          | 0mph             | 0mph           | Custom<sup>4</sup>|
| Subaru               | Impreza 2019             | EyeSight             | Yes     | Stock          | 0mph             | 0mph           | Custom<sup>4</sup>|
| Toyota               | Avalon 2016              | TSS-P                | Yes     | Yes<sup>2</sup>| 20mph<sup>1</sup>| 0mph           | Toyota            |
| Toyota               | Camry 2018               | All                  | Yes     | Stock          | 0mph<sup>5</sup> | 0mph           | Toyota            |
| Toyota               | C-HR 2017-18             | All                  | Yes     | Stock          | 0mph             | 0mph           | Toyota            |
| Toyota               | Corolla 2017-19          | All                  | Yes     | Yes<sup>2</sup>| 20mph<sup>1</sup>| 0mph           | Toyota            |
| Toyota               | Corolla 2020             | All                  | Yes     | Yes            | 0mph             | 0mph           | Toyota            |
| Toyota               | Corolla Hatchback 2019   | All                  | Yes     | Yes            | 0mph             | 0mph           | Toyota            |
| Toyota               | Highlander 2017-18       | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |
| Toyota               | Highlander Hybrid 2018   | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |
| Toyota               | Prius 2016               | TSS-P                | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |
| Toyota               | Prius 2017-19            | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |
| Toyota               | Prius Prime 2017-19      | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |
| Toyota               | Rav4 2016                | TSS-P                | Yes     | Yes<sup>2</sup>| 20mph<sup>1</sup>| 0mph           | Toyota            |
| Toyota               | Rav4 2017-18             | All                  | Yes     | Yes<sup>2</sup>| 20mph<sup>1</sup>| 0mph           | Toyota            |
| Toyota               | Rav4 2019                | All                  | Yes     | Yes            | 0mph             | 0mph           | Toyota            |
| Toyota               | Rav4 Hybrid 2017-18      | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           | Toyota            |

<sup>1</sup>[Comma Pedal](https://community.comma.ai/wiki/index.php/Comma_Pedal) is used to provide stop-and-go capability to some of the openpilot-supported cars that don't currently support stop-and-go. Here is how to [build a Comma Pedal](https://medium.com/@jfrux/comma-pedal-building-with-macrofab-6328bea791e8). ***NOTE: The Comma Pedal is not officially supported by [comma.ai](https://comma.ai).*** <br />
<sup>2</sup>When disconnecting the Driver Support Unit (DSU), otherwise longitudinal control is stock ACC. For DSU locations, see [Toyota Wiki page](https://community.comma.ai/wiki/index.php/Toyota). <br />
<sup>3</sup>[GM installation guide](https://zoneos.com/volt/). <br />
<sup>4</sup>Subaru Giraffe is DIY. <br />
<sup>5</sup>28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control. <br />
<sup>6</sup>Open sourced [Hyundai Giraffe](https://github.com/commaai/neo/tree/master/giraffe/hyundai) is designed for the 2019 Sante Fe; pinout may differ for other Hyundais. <br />
<sup>7</sup>Community built Giraffe, find more information [here](https://zoneos.com/shop/). <br />

Community Maintained Cars
------

| Make                 | Model                    | Supported Package    | Lateral | Longitudinal   | No Accel Below   | No Steer Below | Giraffe           |
| ---------------------| -------------------------| ---------------------| --------| ---------------| -----------------| ---------------|-------------------|
| Honda                | Fit 2018                 | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          | Inverted Nidec    |
| Tesla                | Model S 2012-13          | All                  | Yes     | Not yet        | Not applicable   | 0mph           | Custom<sup>8</sup>|

[[Honda Fit Pull Request]](https://github.com/commaai/openpilot/pull/266). <br />
[[Tesla Model S Pull Request]](https://github.com/commaai/openpilot/pull/246) <br />
<sup>8</sup>Community built Giraffe, find more information here [Community Tesla Giraffe](https://github.com/jeankalud/neo/tree/tesla_giraffe/giraffe/tesla) <br />

Community Maintained Cars are not confirmed by comma.ai to meet our [safety model](https://github.com/commaai/openpilot/blob/devel/SAFETY.md). Be extra cautious using them.

In Progress Cars
------
- All TSS-P Toyota with Steering Assist and LSS-P Lexus with Steering Assist or Lane Keep Assist.
  - Only remaining Toyota cars with no port yet are the Avalon and the Sienna.
- All Hyundai with SmartSense.
- All Kia with SCC and LKAS.
- All Chrysler, Jeep, Fiat with Adaptive Cruise Control and LaneSense.

How can I add support for my car?
------

If your car has adaptive cruise control and lane keep assist, you are in luck. Using a [panda](https://comma.ai/shop/products/panda-obd-ii-dongle/) and [cabana](https://community.comma.ai/cabana/), you can understand how to make your car drive by wire.

We've written guides for [Brand](https://medium.com/@comma_ai/how-to-write-a-car-port-for-openpilot-7ce0785eda84) and [Model](https://medium.com/@comma_ai/openpilot-port-guide-for-toyota-models-e5467f4b5fe6) ports. These guides might help you after you have the basics figured out.

- BMW, Audi, Volvo, and Mercedes all use [FlexRay](https://en.wikipedia.org/wiki/FlexRay) and can be supported after [FlexRay support](https://github.com/commaai/openpilot/pull/463) is merged.
- We put time into a Ford port, but the steering has a 10 second cutout limitation that makes it unusable.
- The 2016-2017 Honda Accord uses a custom signaling protocol for steering that's unlikely to ever be upstreamed.

Directory structure
------
    .
    ├── apk                 # The apk files used for the UI
    ├── cereal              # The messaging spec used for all logs on EON
    ├── common              # Library like functionality we've developed here
    ├── installer/updater   # Manages auto-updates of openpilot
    ├── opendbc             # Files showing how to interpret data from cars
    ├── panda               # Code used to communicate on CAN and LIN
    ├── phonelibs           # Libraries used on EON
    ├── pyextra             # Libraries used on EON
    └── selfdrive           # Code needed to drive the car
        ├── assets          # Fonts and images for UI
        ├── athena          # Allows communication with the app
        ├── boardd          # Daemon to talk to the board
        ├── can             # Helpers for parsing CAN messages
        ├── car             # Car specific code to read states and control actuators
        ├── common          # Shared C/C++ code for the daemons
        ├── controls        # Perception, planning and controls
        ├── debug           # Tools to help you debug and do car ports
        ├── locationd       # Soon to be home of precise location
        ├── logcatd         # Android logcat as a service
        ├── loggerd         # Logger and uploader of car data
        ├── proclogd        # Logs information from proc
        ├── sensord         # IMU / GPS interface code
        ├── test            # Car simulator running code through virtual maneuvers
        ├── ui              # The UI
        └── visiond         # Vision pipeline

To understand how the services interact, see `selfdrive/service_list.yaml`

User Data / chffr Account / Crash Reporting
------

By default, openpilot creates an account and includes a client for chffr, our dashcam app. We use your data to train better models and improve openpilot for everyone.

It's open source software, so you are free to disable it if you wish.

It logs the road facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The user facing camera is only logged if you explicitly opt-in in settings.
It does not log the microphone.

By using it, you agree to [our privacy policy](https://community.comma.ai/privacy.html). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma.ai. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma.ai for the use of this data.

Testing on PC
------

Check out [openpilot-tools](https://github.com/commaai/openpilot-tools): lots of tools you can use to replay driving data, test and develop openpilot from your pc.

Also, within openpilot there is a rudimentary infrastructure to run a basic simulation and generate a report of openpilot's behavior in different longitudinal control scenarios.

```bash
# Requires working docker
./run_docker_tests.sh
```

Contributing
------

We welcome both pull requests and issues on [github](http://github.com/commaai/openpilot). Bug fixes and new car ports encouraged.

We also have a [bounty program](https://comma.ai/bounties.html).

Want to get paid to work on openpilot? [comma.ai is hiring](https://comma.ai/jobs/)

Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
