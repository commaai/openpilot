This README describes the custom features build by Arne Schwarck and Kumar on top of [ArnePilot](http://github.com/commaai/ArnePilot) of [comma.ai](http://comma.ai). This fork is optimized for the Toyota RAV4 Hybrid 2016, Prius TSS2 and Corolla TSS2 when driving on Germany and American roads. If you would like to support the developement on this project feel free to https://www.patreon.com/arneschwarck


[![demo of ArnePilot with this branch](https://img.youtube.com/vi/WKwSq8TPdpo/0.jpg)](https://www.youtube.com/playlist?list=PL3CGUyxys8DuTE1JTkdZwY93ejSfAGxyV)

For a demo of this version 0.5.7(old) of ArnePilot

Find us on Discord https://discord.gg/F95ukSU

# Installation

`cd /data; rm -rf openpilot; git clone --depth 1 https://github.com/arne182/openpilot -b DP085-clean; reboot`

#### Troubleshooting
Arnepilot does't use comma logger instead uses [retropilots api](https://api.retropilot.org/useradmin) no more fear of being banned. This gives a 35% more cpu but at cost of giving [connection error](https://cdn.discordapp.com/attachments/538741329799413760/743231854764884067/image0.jpg)

If you get the [no vehicle](https://cdn.discordapp.com/attachments/538741329799413760/743231854764884067/image0.jpg) after installing Arnepilot completely power cycle your device. If this still doesn't fix the problem look below at panda flashing and run the command. This is a known issue with comma2 users.

## Panda flashing
This is done automatically otherwise run `cd /data/openpilot/panda/board; ./flash.sh; scons -u; reboot` Make sure this done while your device is connnect to your car or if you are feelign lazy their is a button under networking to flash panda:
- allowing no disengage on brake and gas for Toyota
- changing acceleration limits for Toyota and
- adapting lane departure warning where it gives you a slight push back into the middle of the lane without needing to be engaged (not yet complete)
- The Panda version is also changed and checked.

# Branches
`release5`: this is the default branch that is most up to date with the ArnePilot 0.8.4 based off of [dragonpilot](https://github.com/dragonpilot-community/dragonpilot) release branch.

`DP085-clean`: Current development branch.

`release4`: this is my old branch, that is compatible with ArnePilot 0.7.

`release3`: this is my old branch, that is compatible with ArnePilot 0.6.

`release2`: this is my old branch, that is compatible with ArnePilot 0.5.


## Supported Cars
Fork is known to work in both US and Europe
Most of the Toyota works well but it is optimized for:
- RAV4 Hybrid 2016-19
- RAV4 2017-19
- Corolla 2019-21
- Prius 2017-2021
- Avalon TSS-P

# Features

## Retropliot API
This alternative cloud server housing made with the API by [daywalker](https://github.com/florianbrede-ayet/retropilot-server) and hosted by [debugghosting](https://debugged-hosting.com/).

Register an account at [RetroPilot](https://api.retropilot.org/useradmin). Scan the QR and link your account than enjoy the free ban free cloud hosting.

## [move-fast modules](https://github.com/move-fast)
- MapD - control speed of your car and turning by fetching map data from OpenStreetMap.
- Speed Limit Control - module to control cars gas and break so it can set speed limit from map or rsa.
- Turn Controller - vision based breaking by looking at the lane lines or use map data from OpenStreetMap.
- Hands On Wheel - no idea how to explain it.


## Dragonpilot
Since openpilot v0.8.4 Arne has decide to base his fork on [DragonPilot](https://github.com/dragonpilot-community/dragonpilot). So expect all your favorite features to work
- Feature for Dragonpilot can be found ``/data/openpilot/common/dp_conf.py``

## Arnepilot
- Braking:
    - by model(commaai),
    - acceleration measured by steering angle,
    - by curvature (mapd and lane lines),
    - by mapped sign(speeds, roundabouts, bump, hump, road attribute) by [alfhern](https://github.com/move-fast)
- Smooth longitudinal controller also at low speeds
- Forward collision warning actually brakes for you.
- Smart speed (smart speed is essentially speedlimit which eon will not go over unless you have set custom offset) can be overridden by pressing cruise up or down button.
- Hands on wheel sensing to comply with European driving regulations by [alfhern](https://github.com/move-fast)
- Reacting Toyota tssp higher acceleration and braking limits.
- Speed sign reading
- Stock Toyota ldw steering assist
- Cruise set speed available down to 7 kph
- Always on Dashcam recording ( it will save video's to the `/sdcard/media/dashcam`)

### OpEdit features
all OpEdit features can be manged by running the command `python /data/openpilot/op_edit.py`
- [Dynamic Gas](https://github.com/ShaneSmiskol/ArnePilot/tree/stock_additions-devel#dynamic-gas)
This aims to provide a smoother driving experience in stop and go traffic (under 20 mph) by modifying the maximum gas that can be applied based on your current velocity and the relative velocity of the lead car. It'll also of course increase the maximum gas when the lead is accelerating to help you get up to speed quicker than stock. And smoother; this eliminates the jerking you get from stock ArnePilot with comma pedal. It tries to coast if the lead is only moving slowly, it doesn't use maximum gas as soon as the lead inches forward :). When you are above 20 mph, relative velocity and the following distance is taken into consideration.

### Control Modifications
- No disengage for gas, only longitudinal disengage for brake, tire slip or cancel
- Only disengage on main off and on brake at low speed
- Engine profiles button
- Follow distance button
- Lane lines toggle Steer Assist

### UI Modifications
- Dev UI toggle in APK setting.
- GPS Accurecy on the SideBar
- Battery has percentage instead of the battery icon.
- [Dynamic distance profiles] fixed TR: (`0.9`, `1.2`, `1.5`, `1.8`).
0.9 FOLLOW DISTANCE IS TO BE USED IN TRAFFIC UNDER 25MPH ONLY. OTHERWISE YOU NOT STOP AND WILL CRASH INTO THE CAR IN FRONT OF YOU.
- Control 3 gas profiles with sport eco and normal buttons on car (only on limited car)

## Data collection
- Loggin has been Disabled by default on this fork. Just create a account and pair it to community server and turn on the toggles in setting.
- Offline crash logging. sentry does not catches all the error. now if their is no internet it will still log error in `/data/community/crashes`
- OSM tracers logging and uploading anonymously to help improve MapD as well as OSM accuracy. [Arne is currently ranked 5th for overal tracers uploaded](https://www.openstreetmap.org/stats/data_stats.html).
- Added stats that track meter driven as well as overrides/disengagement. These go to a leaderboard. Please added your name to `python /data/opepilot/op_edit.py` to participate.

# Licensing
© OpenStreetMap contributors

ArnePilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://github.com/arne182/ArnePilot/blob/DP08-clean/selfdrive/assets/img_chffr_wheel.png" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
