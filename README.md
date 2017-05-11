Welcome to openpilot
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent.

Currently it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for Hondas and Acuras. It's about on par with Tesla Autopilot at launch, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

The openpilot codebase has been written to be concise and enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

Hardware
------

Right now openpilot supports the [neo research platform](http://github.com/commaai/neo) for vehicle control. We'd like to support other platforms as well.

Install openpilot on a neo device by entering ``https://openpilot.comma.ai`` during NEOS setup.

Supported Cars
------

- Acura ILX 2016 with AcuraWatch Plus
  - Limitations: Due to use of the cruise control for gas, it can only be enabled above 25 mph

- Honda Civic 2016 with Honda Sensing
  - Limitations: Due to limitations in steering firmware, steering is disabled below 18 mph

Directory structure
------

- cereal        -- The messaging spec used for all logs on the phone
- common        -- Library like functionality we've developed here
- dbcs          -- Files showing how to interpret data from cars
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
  - radar         -- Code that talks to the radar and implements RadarInterface
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

It should be relatively easy to add support for the Honda CR-V Touring. The brake message is the same. Steering has a slightly different message with a different message id. Sniff CAN while using LKAS to find it.

The Honda Accord uses different signaling for the steering and probably requires new hardware.

Adding other manufacturers besides Honda/Acura is doable but will be more of an undertaking.

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

Want to get paid to work on openpilot? [comma.ai is hiring](http://comma.ai/hiring.html)

Licensing
------

openpilot is released under the MIT license.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
