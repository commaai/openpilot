Welcome to openpilot
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent.

Currently it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for Hondas, Acuras and Toyotas. It's about on par with Tesla Autopilot at launch, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

The openpilot codebase has been written to be concise and enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

Here are [some](https://www.youtube.com/watch?v=9OwTJFuDI7g) [videos](https://www.youtube.com/watch?v=64Wvt5pYQmE) [of](https://www.youtube.com/watch?v=6IW7Nejsr3A) [it](https://www.youtube.com/watch?v=-VN1YcC83nA) [running](https://www.youtube.com/watch?v=EQJZvVeihZk). And a really cool [tutorial](https://www.youtube.com/watch?v=PwOnsT2UW5o).

Community
------

openpilot is supported by [comma.ai](https://comma.ai/)

We have a [Twitter you should follow](https://twitter.com/comma_ai).

Also, we have a 3500+ person [community on slack](https://slack.comma.ai).

Hardware
------

Right now openpilot supports the [EON Dashcam DevKit](https://shop.comma.ai/products/eon-dashcam-devkit). We'd like to support other platforms as well.

Install openpilot on a neo device by entering ``https://openpilot.comma.ai`` during NEOS setup.

Supported Cars
------

### Honda + Acura ###

- Honda Accord 2018 with Honda Sensing (alpha!)
  - Uses stock Honda Sensing for longitudinal control

- Honda Civic 2016+ with Honda Sensing
  - Due to limitations in steering firmware, steering is disabled below 12 mph
  - Note that the hatchback model is not supported

- Honda Civic Hatchback 2017+ with Honda Sensing (alpha!)
  - Due to limitations in steering firmware, steering is disabled below 12 mph
  - Uses stock Honda Sensing for longitudinal control

- Honda CR-V 2017-2018 with Honda Sensing (alpha!)
  - Due to limitations in steering firmware, steering is disabled below 12 mph
  - Uses stock Honda Sensing for longitudinal control

- Honda CR-V Touring 2015-2016
  - Can only be enabled above 25 mph

- Honda Odyssey 2018 with Honda Sensing (alpha!)
  - Can only be enabled above 25 mph

- Honda Pilot 2017 with Honda Sensing (alpha!)
  - Can only be enabled above 27 mph

- Honda Ridgeline 2017 with Honda Sensing (alpha!)
  - Can only be enabled above 27 mph

- Acura ILX 2016 with AcuraWatch Plus
  - Due to use of the cruise control for gas, it can only be enabled above 25 mph

- Acura RDX 2018 with AcuraWatch Plus (alpha!)
  - Can only be enabled above 25 mph

### Toyota + Lexus ###

- Toyota RAV-4 2016+ non-hybrid with TSS-P
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Rav4_.28for_openpilot.29) and can be enabled above 20 mph

- Toyota Prius 2017+
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Prius_.28for_openpilot.29)
  - Lateral control needs improvements

- Toyota RAV-4 2017+ hybrid
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Rav4_.28for_openpilot.29) and can do stop and go

- Toyota Corolla 2017+
  - By default it uses stock Toyota ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Corolla_.28for_openpilot.29) and can be enabled above 20 mph

- Lexus RX 2017+ hybrid (alpha!)
  - By default it uses stock Lexus ACC for longitudinal control
  - openpilot longitudinal control available after unplugging the [Driving Support ECU](https://community.comma.ai/wiki/index.php/Toyota#Lexus_RX_hybrid)

### GM (Chevrolet + Cadillac) ###

- Chevrolet Volt Premier 2017+
  - Driver Confidence II package (adaptive cruise control) required
  - Can only be enabled above 18 mph
  - Read the [installation guide](https://www.zoneos.com/volt.htm)

- Cadillac CT6
  - Uses stock ACC for longitudinal control
  - Requires multiple panda for proxying the ASCMs

In Progress Cars
------
- All TSS-P Toyota with Steering Assist.
  - 'Full Speed Range Dynamic Radar Cruise Control' is required to enable stop-and-go. Only the Prius, Camry and C-HR have this option.
  - Even though the Tundra, Sequoia and the Land Cruiser have TSS-P, they don't have Steering Assist and are not supported.
- All LSS-P Lexus with Steering Assist or Lane Keep Assist.
  - 'All-Speed Range Dynamic Radar Cruise Control' is required to enable stop-and-go. Only the GS, GSH, GS, F, RX, RXH, LX, NX, NXH, LC, LCH, LS, LSH have this option.
  - Even though the LX have TSS-P, it does not have Steering Assist and is not supported.

Community Maintained Cars
------

- [Classic Tesla Model S (pre-AP)](https://github.com/commaai/openpilot/pull/246)

How can I add support for my car?
------

If your car has adaptive cruise control and lane keep assist, you are in luck. Using a [panda](https://panda.comma.ai) and [cabana](https://community.comma.ai/cabana/), you can understand how to make your car drive by wire.

We've written a [porting guide](https://medium.com/@comma_ai/openpilot-port-guide-for-toyota-models-e5467f4b5fe6) for Toyota that might help you after you have the basics figured out.

Sadly, BMW, Audi, Volvo, and Mercedes all use [FlexRay](https://en.wikipedia.org/wiki/FlexRay) and are unlikely to be supported any time soon. We also put time into a Ford port, but the steering has a 10 second cutout limitation that makes it unusable.

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
  - orbd          -- Service generating ORB features from road camera
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

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
