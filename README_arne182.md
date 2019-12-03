# Welcome to arne fork of openpilot

This README describes the custom features build by me (Arne Schwarck) on top of [openpilot](http://github.com/commaai/openpilot) of [comma.ai](http://comma.ai). This fork is optimized for the Toyota RAV4 Hybrid 2016 and for driving in Germany but also works with other cars and in other countries 
- [ ] TODO describe which other cars and countries are known
[![](https://i.imgur.com/xY2gdHv.png)](#)

For a demo of this version of openpilot check the video below:
[![demo of openpilot with this branch](https://img.youtube.com/vi/WKwSq8TPdpo/0.jpg)](https://www.youtube.com/watch?v=WKwSq8TPdpo)

# Installation

- [ ] TODO describe when Panda flashing is needed, what it does and a link to how this can be done
- [ ] TODO add a link how to install a custom fork

# Configuration

- [ ] TODO describe how to change/add custom setting in json file

# Branches

- [ ] TODO described relevant branches 

# Features

- [ ] TODO add other features

- [ ] TODO check if these features are still relevant

- [ ] TODO if applicable describe what config options are available
 
## Automatic Lane Change Assist (ALC)
Check your surroundings, signal in the direction you would like to change lanes, and let openpilot do the rest. You can choose between three ALC profiles, Wifey, Normal, and Mad Max. Each increasing in steering torque.


## Stock Lane Keeping Assist (LKA)
Arne has worked on recreating the lane keeping assist system present in your car for openpilot. It works with cruise control not engaged, attempting to steer to keep you inside your lane when it detects you are departing it.


## [Dynamic Following Distance Profile](https://github.com/ShaneSmiskol/openpilot/blob/dynamic-follow/README.md)
(outdated: on 0.5.8, `dynamic-follow` branch only): Three following distance (TR) profiles are available to select; 0.9 seconds, 2.7 seconds, and a custom tuned dynamic follow profile. The first two behave as your stock cruise control system does. Dynamic follow aims to provide a more natural feeling drive, adjusting your distance from the lead car based on your speed, your relative velocity with the lead car, and your acceleration (or deceleration). If the system detects the lead car decelerating, your car should start to brake sooner than a hard-coded TR value. Same with accelerating.


## Slow Mode (SLO)
For cars with longitudinal control down to 0 mph, you have the option to activate SLO mode which enables you to set your car's cruise control under your car's limit. For example, you could coast along at 15, 10, or even 5 mph.

## Acceleration Profiles (GAS)
You can select from three acceleration profiles with the GAS button. If your car accelerates too slowly for your liking, this will solve that. **Recently added**: dynamic acceleration profile for users with comma pedals. This should provide a smoother acceleration experience in stop and go traffic.

## Select Vision Model (on 0.5.8, `dynamic-follow` branch only)
You can select whether you would like to use the wiggly model or the normal vision model for path planning. Wiggly has more torque and can better guess the road curvature without lane lines, but it occasionally crashes or mispredicts the path.

## EON and openpilot Stats
With the on-screen UI, you can view stats about your EON such as its temperature, your grey panda's GPS accuracy, the lead car's relative velocity, its distance, and more.

Warning from kegman: `WARNING: Do NOT depend on OP to stop the car in time if you are approaching an object which is not in motion in the same direction as your car. The radar will NOT detect the stationary object in time to slow your car enough to stop. If you are approaching a stopped vehicle you must disengage and brake as radars ignore objects that are not in motion.`


# Licensing

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
