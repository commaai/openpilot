Welcome to openpilot
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent.

Currently it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for Hondas, Acuras and Toyotas. It's about on par with Tesla Autopilot at launch, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

The openpilot codebase has been written to be concise and enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

Here are [some](https://www.youtube.com/watch?v=9OwTJFuDI7g) [videos](https://www.youtube.com/watch?v=64Wvt5pYQmE) [of](https://www.youtube.com/watch?v=6IW7Nejsr3A) [it](https://www.youtube.com/watch?v=-VN1YcC83nA) [running](https://www.youtube.com/watch?v=EQJZvVeihZk). And a really cool [tutorial](https://www.youtube.com/watch?v=PwOnsT2UW5o).

Hardware
------

Right now openpilot supports the [NEO research platform](http://github.com/commaai/neo) and the [EON Dashcam DevKit](https://shop.comma.ai/products/eon-dashcam-devkit). We'd like to support other platforms as well.

Install openpilot on a neo device by entering ``https://openpilot.comma.ai`` during NEOS setup.

Supported Cars
------

- Acura ILX 2016 with AcuraWatch Plus
  - Due to use of the cruise control for gas, it can only be enabled above 25 mph

- Honda Civic 2016-2017 with Honda Sensing
  - Due to limitations in steering firmware, steering is disabled below 12 mph
  - Note that the hatchback model is not supported

- Honda CR-V Touring 2015-2016
  - Can only be enabled above 25 mph

- Honda Odyssey 2018 with Honda Sensing (alpha!)
  - Can only be enabled above 25 mph

- Toyota RAV-4 2016+ non-hybrid with TSS-P
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Rav4_.28for_openpilot.29) and can be enabled above 20 mph

- Toyota Prius 2017 (alpha!)
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Prius_.28for_openpilot.29)
  - Lateral control needs improvements

- Toyota RAV-4 2017 hybrid (alpha!)
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Rav4_.28for_openpilot.29) and can do stop and go

- Toyota Corolla 2017 (alpha!)
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Corolla_.28for_openpilot.29) and can be enabled above 20 mph

In Progress Cars
------
- Probably all TSS-P Toyota with Steering Assist.
  - 'Full Speed Range Dynamic Radar Cruise Control' is required to enable stop-and-go. Only the Prius, Camry and C-HR have this option.
  - Even though the Tundra, Sequoia and the Land Cruiser have TSS-P, they don't have Steering Assist and are not supported.

Community WIP Cars
------

- [Chevy Volt 2016-2018 Premier with Driver Confidence II](https://github.com/commaai/openpilot/pull/104)

- [Classic Tesla Model S (pre-AP)](https://github.com/commaai/openpilot/pull/145)

- [Honda Pilot 2017 with Honda Sensing](https://github.com/commaai/openpilot/pull/161)

- [Acura RDX 2018 with AcuraWatch Plus](https://github.com/commaai/openpilot/pull/162)

Directory structure
------

- cereal        -- The messaging spec used for all logs on the phone
- common        -- Library like functionality we've developed here
- opendbc       -- Files showing how to interpret data from cars
- panda         -- Code used to communicate on CAN and LIN
- phonelibs     -- Libraries used on the phone
- selfdrive     -- Code needed to drive the car
  - assets        -- Fonts for ui
  - boardd        -- Daemon to talk to the board
  - car           -- Code that talks to the car and implements CarInterface
  - common        -- Shared C/C++ code for the daemons
  - controls      -- Python controls (PID loops etc) for the car
  - debug         -- Tools to help you debug and do car ports
  - logcatd       -- Android logcat as a service
  - loggerd       -- Logger and uploader of car data
  - proclogd      -- Logs information from proc
  - sensord       -- IMU / GPS interface code
  - test/plant    -- Car simulator running code through virtual maneuvers
  - ui            -- The UI
  - visiond       -- embedded vision pipeline

To understand how the services interact, see `selfdrive/service_list.yaml`

Testing on PC
------

There is rudimentary infrastructure to run a basic simulation and generate a report of openpilot's behavior in different scenarios.

```bash
# Requires working docker
./run_docker_tests.sh
```

The results are written to `selfdrive/test/plant/out/index.html`

More extensive testing infrastructure and simulation environments are coming soon.

Adding Car Support
------

comma.ai offers [bounties](http://comma.ai/bounties.html) for adding additional car support.

CR-V Touring support came in through this program. Chevy Volt is close. Accord is close as well.

User Data / chffr Account / Crash Reporting
------

By default openpilot creates an account and includes a client for chffr, our dashcam app. We use your data to train better models and improve openpilot for everyone.

It's open source software, so you are free to disable it if you wish.

It logs the road facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
It does not log the user facing camera or the microphone.

By using it, you agree to [our privacy policy](https://beta.comma.ai/privacy.html). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma.ai. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma.ai for the use of this data.

Contributing
------

We welcome both pull requests and issues on
[github](http://github.com/commaai/openpilot). See the TODO file for a list of
good places to start.

Want to get paid to work on openpilot? [comma.ai is hiring](http://comma.ai/positions.html)

Licensing
------

openpilot is released under the MIT license.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
