
[openpilot](http://github.com/commaai/openpilot) is an open source driving agent. Currently it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for Hondas, Acuras, Toyotas, and a Chevy. It's about on par with Tesla Autopilot and GM Super Cruise, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

The openpilot codebase has been written to be concise and enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

Community
------

openpilot is developed by [comma.ai](https://comma.ai/) and users like you.

We have a [Twitter you should follow](https://twitter.com/comma_ai).

Also, we have a 3500+ person [community on slack](https://slack.comma.ai).



Honda Accord Hybrid Fork
------

This repository contains a Honda Accord Hybrid fork of OpenPilot, with the following enhancements:

- Allow turning OP steering on/off using LKAS button
- Allow use of gas pedal (like stock)
- Set ACC speed with virtual buttons
- Brake light visualization
- Blinker display
- Mock radar display
- Upload toggle works on WIFI (very helpful for debugging)
- Engineering display (thanks to community)
- Latest, non-wiggle vision model (thanks to @energee)

Code changes: [Code diff to comma.ai devel branch](https://github.com/commaai/openpilot/compare/devel...perpetuoviator:devel) - click on the "Files changed" tab.


There are a bunch of changes for debugging/compiling on Mac, etc. The relevant changes are in the following 11 files:

! File                                             ! Added features                                                !
!--------------------------------------------------!---------------------------------------------------------------!
! opendbc/honda_accord_s2t_2018_can_generated.dbc  ! Brake lights                                                  !
! panda/board/safety/safety_honda.h                ! Gas pedal use, virtual buttons                                !
! selfdrive/car/honda/carcontroller.py             ! Virtual buttons                                               !
! selfdrive/car/honda/carstate.py                  ! Brake lights                                                  !
! selfdrive/car/honda/interface.py                 ! Virtual buttons, brake lights, gas pedal use, LKAS button     !
! selfdrive/controls/controlsd.py                  ! LKAS button, No beep on user disable                          !
! selfdrive/controls/lib/alertmanager.py           ! No beep on user disable                                       !
! selfdrive/controls/radard.py                     ! Mock radar support                                            !
! selfdrive/loggerd/uploader.py                    ! Upload toggle works on WIFI                                   !
! selfdrive/ui/ui.c                                ! Virtual buttons, brake lights, blinkers, engineering display  !
! selfdrive/visiond/visiond                        ! Latest, non-wiggle model                                      !




Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.


**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
