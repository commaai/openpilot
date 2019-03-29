[![](https://i.imgur.com/UetIFyH.jpg)](#)

Welcome to openpilot by Arne Schwarck
https://youtu.be/WKwSq8TPdpo
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent. Currently, it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS) for selected Honda, Toyota, Acura, Lexus, Chevrolet, Hyundai, Kia. It's about on par with Tesla Autopilot and GM Super Cruise, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

Highlight Features
=======================

* **Automatic Lane Change Assist (ALC)**: Check your surroundings, signal in the direction you would like to change lanes, and let openpilot do the rest. You can choose between three ALC profiles, [Normal, Wifey, and Mad Max](https://github.com/ShaneSmiskol/openpilot/blob/release2/selfdrive/car/toyota/carstate.py#L145). Each increasing in steering torque.
* **Stock Lane Keeping Assist (LKA)**: Arne has worked on recreating the lake keeping assist system present in your car for openpilot. It works with cruise control not engaged, attempting to steer to keep you inside your lane when it detects you are departing it.
* **Dynamic Following Distance Profile**: Three following distance (TR) profiles are available to select; 0.9 seconds, 2.7 seconds, and a [custom tuned dynamic follow profile](https://github.com/ShaneSmiskol/openpilot/blob/7252ab74458328028b3296d17fde3c9a39db8d7b/selfdrive/controls/lib/planner.py#L207). The first two behave as your stock cruise control system does. Dynamic follow aims to provide a more natural feeling drive, adjusting your distance from the lead car based on your speed, your relative velocity with the lead car, and your acceleration (or deceleration). If the system detects the lead car decelerating, your car should start to brake sooner than a hard-coded TR value.
* **Slow Mode (SLO)**: For cars with longitudinal control down to 0 mph, you have the option to activate SLO mode which enables you to set your car's cruise control under your car's limit. For example, you could coast along at 15, 10, or even 5 mph.
* **Acceleration Profiles (GAS)**: You can select from [two acceleration profiles](https://github.com/ShaneSmiskol/openpilot/blob/dynamic-follow/selfdrive/controls/controlsd.py#L271) with the GAS button. If your car accelerates too slowly for your liking, this will solve that.
* **Select Vision Model (on 0.5.8, `dynamic-follow` branch only)**: You can select whether you would like to use the wiggly model or the normal vision model for path planning. Wiggly has more torque and can better guess the road curvature without lane lines, but it occasionally crashes or mispredicts the path.
* **EON and openpilot Stats**: With the on-screen UI, you can view stats about your EON such as its temperature, your grey panda's GPS accuracy, the lead car's relative velocity, its distance, and more.
