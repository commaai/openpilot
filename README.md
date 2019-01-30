Honda Accord Hybrid Fork

This repository contains a Honda Accord Hybrid fork of OpenPilot, with the following enhancements:
- Allow turning OP steering on/off using LKAS button
- Allow use of gas pedal (like stock)
- Set ACC speed with virtual buttons
- Brake light visualization
- Blinker display
- Mock radar display
- Upload toggle works on WIFI (very helpful for debugging)
- Show device IP address and current network bandwidth on home screen
- Engineering display (thanks to community)
- On-screen log display (toggled on/off by volume up)
- Enable tethering via volume down button
- Shutdown device automatically after all uploads complete (saves battery while car is parked)


Code changes: [Code diff to comma.ai devel branch](https://github.com/commaai/openpilot/compare/devel...perpetuoviator:devel) - click on the "Files changed" tab.


There are a bunch of changes for debugging/compiling on Mac, etc. The relevant changes are in the following 11 files:

| File                                             | Added features                                                |
|--------------------------------------------------|---------------------------------------------------------------|
| opendbc/honda_accord_s2t_2018_can_generated.dbc  | Brake lights                                                  |
| panda/board/safety/safety_honda.h                | Gas pedal use, virtual buttons                                |
| selfdrive/car/honda/carcontroller.py             | Virtual buttons                                               |
| selfdrive/car/honda/carstate.py                  | Brake lights                                                  |
| selfdrive/car/honda/interface.py                 | Virtual buttons, brake lights, gas pedal use, LKAS button     |
| selfdrive/controls/controlsd.py                  | LKAS button, No beep on user disable                          |
| selfdrive/controls/lib/alertmanager.py           | No beep on user disable                                       |
| selfdrive/controls/radard.py                     | Mock radar support                                            |
| selfdrive/loggerd/uploader.py                    | Upload toggle works on WIFI                                   |
| selfdrive/ui/ui.c                                | Buttons, brake lights, blinkers, engineering display          |
| selfdrive/ui/devicestate.c                       | Support functions for various UI changes                      |

Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
