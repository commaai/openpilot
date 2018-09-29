Honda Accord Hybrid fork of OpenPilot, with the following enhancements:

- Allow turning OP steering on/off using LKAS button
- Allow use of gas pedal (like stock)
- Set ACC speed with virtual buttons
- Brake light visualization
- Blinker display
- Mock radar display
- Upload toggle works on WIFI (very helpful for debugging)
- Engineering display (thanks to community)
- Latest, non-wiggle vision model (thanks to @energee)


You can review the changes when you go to https://github.com/commaai/openpilot/compare/devel...perpetuoviator:devel and click on the "Files changed" tab. There are a bunch of changes for debugging/compiling on Mac, etc. The relevant changes are in the following 11 files:

opendbc/honda_accord_s2t_2018_can_generated.dbc: Brake lights
panda/board/safety/safety_honda.h: Gas pedal use, virtual buttons
selfdrive/car/honda/carcontroller.py: Virtual buttons
selfdrive/car/honda/carstate.py: Brake lights
selfdrive/car/honda/interface.py: Virtual buttons, brake lights, gas pedal use, LKAS button 
selfdrive/controls/controlsd.py: LKAS button, No beep on user disable
selfdrive/controls/lib/alertmanager.py: No beep on user disable
selfdrive/controls/radard.py: Mock radar support
selfdrive/loggerd/uploader.py: Upload toggle works on WIFI
selfdrive/ui/ui.c: Virtual buttons, brake lights, blinkers, engineering display
selfdrive/visiond/visiond: Latest, non-wiggle model