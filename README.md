[![](https://i.imgur.com/UetIFyH.jpg)](#)

Welcome to openpilot
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent. Currently it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for selected Honda, Toyota, Acura, Lexus, Chevrolet, Hyundai, Kia. It's about on par with Tesla Autopilot and GM Super Cruise, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

The openpilot codebase has been written to be concise and enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

Community
------

openpilot is developed by [comma.ai](https://comma.ai/) and users like you.

We have a [Twitter you should follow](https://twitter.com/comma_ai).

Also, we have a several thousand people community on [slack](https://slack.comma.ai).

<table>
  <tr>
    <td><a href="https://www.youtube.com/watch?v=9TDi0BHgXyo" title="YouTube" rel="noopener"><img src="https://i.imgur.com/gBTo7yB.png"></a></td>
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

Right now openpilot supports the [EON Dashcam DevKit](https://comma.ai/shop/products/eon-dashcam-devkit). We'd like to support other platforms as well.

Install openpilot on a neo device by entering ``https://openpilot.comma.ai`` during NEOS setup.

Supported Cars
------

| Make                 | Model                 | Supported Package    | Lateral | Longitudinal   | No Accel Below   | No Steer Below |
| ---------------------| ----------------------| ---------------------| --------| ---------------| -----------------| ---------------|
| Acura                | ILX 2016              | AcuraWatch Plus      | Yes     | Yes            | 25mph<sup>1</sup>| 25mph          |
| Acura                | ILX 2017              | AcuraWatch Plus      | Yes     | Yes            | 25mph<sup>1</sup>| 25mph          |
| Acura                | RDX 2018              | AcuraWatch Plus      | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Chevrolet<sup>3</sup>| Volt 2017             | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           |
| Chevrolet<sup>3</sup>| Volt 2018             | Adaptive Cruise      | Yes     | Yes            | 0mph             | 7mph           |
| Honda                | Accord 2018           | All                  | Yes     | Stock          | 0mph             | 3mph           |
| Honda                | Civic 2016            | Honda Sensing        | Yes     | Yes            | 0mph             | 12mph          |
| Honda                | Civic 2017            | Honda Sensing        | Yes     | Yes            | 0mph             | 12mph          |
| Honda                | Civic 2017 *(Hatch)*  | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          |
| Honda                | Civic 2018            | Honda Sensing        | Yes     | Yes            | 0mph             | 12mph          |
| Honda                | Civic 2018 *(Hatch)*  | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          |
| Honda                | CR-V 2015             | Touring              | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Honda                | CR-V 2016             | Touring              | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Honda                | CR-V 2017             | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          |
| Honda                | CR-V 2018             | Honda Sensing        | Yes     | Stock          | 0mph             | 12mph          |
| Honda                | Odyssey 2017          | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 0mph           |
| Honda                | Odyssey 2018          | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 0mph           |
| Honda                | Odyssey 2019          | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 0mph           |
| Honda                | Pilot 2017            | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Honda                | Pilot 2018            | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Honda                | Pilot 2019            | All                  | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Honda                | Ridgeline 2017        | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Honda                | Ridgeline 2018        | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Hyundai<sup>6</sup>  | Santa Fe 2019         | All                  | Yes     | Stock          | 0mph             | 0mph           |
| Hyundai<sup>6</sup>  | Elantra 2017          | SCC + LKAS           | Yes     | Stock          | 19mph            | 34mph          |
| Hyundai<sup>6</sup>  | Genesis 2018          | All                  | Yes     | Stock          | 19mph            | 34mph          |
| Kia<sup>6</sup>      | Sorento 2018          | All                  | Yes     | Stock          | 0mph             | 0mph           |
| Kia<sup>6</sup>      | Stinger 2018          | SCC + LKAS           | Yes     | Stock          | 0mph             | 0mph           |
| Lexus                | RX Hybrid 2017        | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Lexus                | RX Hybrid 2018        | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Camry 2018<sup>4</sup>| All                  | Yes     | Stock          | 0mph<sup>5</sup> | 0mph           |
| Toyota               | C-HR 2018<sup>4</sup> | All                  | Yes     | Stock          | 0mph             | 0mph           |
| Toyota               | Corolla 2017          | All                  | Yes     | Yes<sup>2</sup>| 20mph            | 0mph           |
| Toyota               | Corolla 2018          | All                  | Yes     | Yes<sup>2</sup>| 20mph            | 0mph           |
| Toyota               | Highlander 2017       | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Highlander Hybrid 2018| All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Prius 2016            | TSS-P                | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Prius 2017            | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Prius 2018            | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Prius Prime 2017      | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Prius Prime 2018      | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Rav4 2016             | TSS-P                | Yes     | Yes<sup>2</sup>| 20mph            | 0mph           |
| Toyota               | Rav4 2017             | All                  | Yes     | Yes<sup>2</sup>| 20mph            | 0mph           |
| Toyota               | Rav4 2018             | All                  | Yes     | Yes<sup>2</sup>| 20mph            | 0mph           |
| Toyota               | Rav4 Hybrid 2017      | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |
| Toyota               | Rav4 Hybrid 2018      | All                  | Yes     | Yes<sup>2</sup>| 0mph             | 0mph           |

<sup>1</sup>[Comma Pedal](https://community.comma.ai/wiki/index.php/Comma_Pedal) is used to provide stop-and-go capability to some of the openpilot-supported cars that don't currently support stop-and-go. Here is how to [build a Comma Pedal](https://medium.com/@jfrux/comma-pedal-building-with-macrofab-6328bea791e8). ***NOTE: The Comma Pedal is not officially supported by [comma.ai](https://comma.ai)***  
<sup>2</sup>When disconnecting the Driver Support Unit (DSU), otherwise longitudinal control is stock ACC. For DSU locations, see [Toyota Wiki page](https://community.comma.ai/wiki/index.php/Toyota)  
<sup>3</sup>[GM installation guide](https://www.zoneos.com/volt.htm)  
<sup>4</sup>It needs an extra 120Ohm resistor ([pic1](https://i.imgur.com/CmdKtTP.jpg), [pic2](https://i.imgur.com/s2etUo6.jpg)) on bus 3 and giraffe switches set to 01X1 (11X1 for stock LKAS), where X depends on if you have the [comma power](https://comma.ai/shop/products/power/).  
<sup>5</sup>28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.  
<sup>6</sup>Giraffe is under development: architecture similar to Toyota giraffe, with an extra 120Ohm resistor on bus 3.

Community Maintained Cars
------

| Make          | Model                     | Supported Package    | Lateral | Longitudinal   | No Accel Below   | No Steer Below |
| -------       | ----------------------    | -------------------- | ------- | ------------   | --------------   | -------------- |
| Honda         | Fit 2018                  | Honda Sensing        | Yes     | Yes            | 25mph<sup>1</sup>| 12mph          |
| Tesla         | Model S 2012              | All                  | Yes     | Not yet        | Not applicable   | 0mph           |
| Tesla         | Model S 2013              | All                  | Yes     | Not yet        | Not applicable   | 0mph           |

[[Honda Fit Pull Request]](https://github.com/commaai/openpilot/pull/266).
[[Tesla Model S Pull Request]](https://github.com/commaai/openpilot/pull/246)

Community Maintained Cars are not confirmed by comma.ai to meet our [safety model](https://github.com/commaai/openpilot/blob/devel/SAFETY.md). Be extra cautious using them.

In Progress Cars
------
- All TSS-P Toyota with Steering Assist.
  - 'Full Speed Range Dynamic Radar Cruise Control' is required to enable stop-and-go. Only the Prius, Camry and C-HR have this option.
  - Even though the Tundra, Sequoia and the Land Cruiser have TSS-P, they don't have Steering Assist and are not supported.
- All LSS-P Lexus with Steering Assist or Lane Keep Assist.
  - 'All-Speed Range Dynamic Radar Cruise Control' is required to enable stop-and-go. Only the GS, GSH, F, RX, RXH, LX, NX, NXH, LC, LCH, LS, LSH have this option.
  - Even though the LX have TSS-P, it does not have Steering Assist and is not supported.
- All Hyundai with SmartSense.
- All Kia with SCC and LKAS.

How can I add support for my car?
------

If your car has adaptive cruise control and lane keep assist, you are in luck. Using a [panda](https://comma.ai/shop/products/panda-obd-ii-dongle/) and [cabana](https://community.comma.ai/cabana/), you can understand how to make your car drive by wire.

We've written guides for [Brand](https://medium.com/@comma_ai/how-to-write-a-car-port-for-openpilot-7ce0785eda84) and [Model](https://medium.com/@comma_ai/openpilot-port-guide-for-toyota-models-e5467f4b5fe6) ports. These guides might help you after you have the basics figured out.

- BMW, Audi, Volvo, and Mercedes all use [FlexRay](https://en.wikipedia.org/wiki/FlexRay) and are unlikely to be supported any time soon.
- We put time into a Ford port, but the steering has a 10 second cutout limitation that makes it unusable.
- The 2016-2017 Honda Accord use a custom signaling protocol for steering that's unlikely to ever be upstreamed.

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
        └── visiond         # Embedded vision pipeline

To understand how the services interact, see `selfdrive/service_list.yaml`

User Data / chffr Account / Crash Reporting
------

By default openpilot creates an account and includes a client for chffr, our dashcam app. We use your data to train better models and improve openpilot for everyone.

It's open source software, so you are free to disable it if you wish.

It logs the road facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The user facing camera is only logged if you explicitly opt-in in settings.
It does not log the microphone.

By using it, you agree to [our privacy policy](https://community.comma.ai/privacy.html). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma.ai. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma.ai for the use of this data.

Testing on PC
------

There is rudimentary infrastructure to run a basic simulation and generate a report of openpilot's behavior in different scenarios.

```bash
# Requires working docker
./run_docker_tests.sh
```

The resulting plots are displayed in `selfdrive/test/tests/plant/out/longitudinal/index.html`

More extensive testing infrastructure and simulation environments are coming soon.

Contributing
------

We welcome both pull requests and issues on
[github](http://github.com/commaai/openpilot). Bug fixes and new car ports encouraged.

Want to get paid to work on openpilot? [comma.ai is hiring](https://comma.ai/jobs/)

Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
