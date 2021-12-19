# Stock Additions - [2021-12-19](/SA_RELEASES.md) (0.8.13)

Stock Additions is a fork of openpilot designed to be minimal in design while boasting various feature additions and behavior improvements over stock. I have a 2017 Toyota Corolla with comma pedal, so most of my changes are designed to improve the longitudinal performance.

Want to request a feature or create a bug report? [Open an issue here!](https://github.com/sshane/openpilot/issues/new/choose) Want to reach me to debug an issue or have a question? `Shane#6175` on Discord!

[View Stock Additions Changelog](/SA_RELEASES.md)

---
## Highlight Features

### Behavior Changes
* [**Dynamic follow (now with profiles!)**](#dynamic-follow-3-profiles) - 3 + auto profiles to control distance
  * [**`auto-df` model for automatic distance profile switching**](#Automatic-DF-profile-switching)
* **Lane Speed**
  * [**Lane Speed Alerts**](#Lane-Speed-alerts) - alerts for when an adjacent lane is faster
  * [**~~Dynamic camera offsetting~~ (removed temporarily)**](#Dynamic-camera-offset-based-on-oncoming-traffic) - moves you over if adjacent lane has oncoming traffic
* [**~~Dynamic gas~~**](#dynamic-gas) - smoother gas control
* [**~~Adding derivative to PI for better control~~**](#pi---pid-controller-for-long-and-lat) - lat: smoother control in turns; long: fix for comma pedal overshoot

### General Features
* [**~~Smoother long control using delay~~ (stock added this!)**](#compensate-for-longitudinal-delay-for-earlier-braking) - using an accel delay, just like for lateral
* [**Customize this fork**](#Customize-this-fork-opEdit) - easily edit fork parameters with support for live tuning
* [**Automatic updates**](#Automatic-updates)
* [**ZSS Support**](#ZSS-support) - takes advantage of your high-precision Zorrobyte Steering Sensor
* [**~~Offline crash logging~~ (removed temporarily)**](#Offline-crash-logging) - check out `/data/community/crashes`

### Visual Changes (LINKS WIP)
* [**Colored lane lines**]() - based on distance from car
* [**Colored model path**]() - based on curvature

## Documentation
* [**Quick Installation**](#Quick-installation)
* [**Branches**](#Branches)
* [**Videos**](#Videos)

---
## Behavior changes

### Dynamic follow (3 profiles)
Dynamic follow aims to provide the stock (Toyota) experience of having three different distance settings. Dynamic follow works by dynamically changing the distance in seconds which is sent to the long MPC to predict a speed to travel at. Basically, if the lead is decelerating or might soon, increase distance to prepare. And if the lead is accelerating, reduce distance to get up to speed quicker.

Dynamic follow works if openpilot can control your vehicle's gas and brakes (longitudinal). [Check if openpilot can control your vehicle's longitudinal from this list.](https://github.com/commaai/openpilot#supported-cars)

Just use the button on the button right of the screen while driving to change between these profiles:
  * [`traffic`](#Videos) - Meant to keep you a bit closer in traffic, hopefully reducing cut-ins. Always be alert, as you are with any driving assistance software.
  * `relaxed` - This is the default dynamic follow profile for casual driving.
  * `stock` - This is the stock 1.8 second profile default in stock openpilot, with no dynamic follow mods. The previous roadtrip profile was closer than a *true road trip* profile, this is more in line with that intention.
  * [`auto`](#Automatic-DF-profile-switching) - The auto dynamic follow model was trained on about an hour of me manually cycling through the different profiles based on driving conditions, this profile tries to replicate those decisions entirely on its own.

<p align="center">
  <img src=".media/df_profiles.jpg?raw=true">
</p>

---
### Automatic DF profile switching
I've trained a custom model with Keras that takes in the past 35 seconds of your speed, the lead's speed and the lead's distance. With these inputs, it tries to correctly predict which profile is the best for your current situation.

It's only been trained on about an hour of data, so it's not perfect yet, but it's great for users who just want to set it and forget it. **To enable the `auto` profile, simply tap the profile changing button for dynamic follow until it reaches the `auto` profile!**

If you're annoyed by the silent alerts that show when the model has changed the profile automatically, just use [opEdit](#Customize-this-fork-opEdit) and set `hide_auto_df_alerts` to `True`. Auto profile and model will remain functional but will not show alerts.

Resources:
- [The auto-df repo.](https://github.com/sshane/auto-df)
- [The model file.](https://github.com/sshane/openpilot/blob/SA-master/selfdrive/controls/lib/dynamic_follow/auto_df.py)
- I converted the Keras model to be able to run with pure NumPy using [Konverter](https://github.com/sshane/Konverter).

---
### Lane Speed alerts
This feature alerts you of faster-travelling adjacent lanes and can be configured using the on-screen *LS* button on the bottom right to either be disabled, audible, or silent.

The idea behind this feature is since we often become very relaxed behind the wheel when being driven by openpilot, we don't always notice when we've become stuck behind a slower-moving vehicle. When either the left or right adjacent lane is moving faster than your current lane, LaneSpeed alerts the user that a faster lane is available so that they can make a lane change, overtaking the slower current lane. Thus saving time in the long run on long road trips or in general highway driving!

The original idea is thanks to [Greengree#5537](https://github.com/greengree) on Discord. This feature is available at 35 mph and up.

---
### Dynamic camera offset (based on oncoming traffic)
This feature automatically adjusts your position in the lane if an adjacent lane has oncoming traffic. For example, if you're on a two-lane highway and the left adjacent lane has oncoming cars, LaneSpeed recognizes those cars and applies an offset to your `CAMERA_OFFSET` to move you over in the lane, keeping you farther from oncoming cars.

**This feature is available from 35 to ~60 mph due to a limitation with the Toyota radar**. It may not recognize oncoming traffic above 60 mph or so. To enable or disable this feature, use `opEdit` and change this parameter: `dynamic_camera_offset`.

---
### Dynamic gas
Dynamic gas aims to provide a smoother driving experience in stop and go traffic (under 20 mph) by reducing the maximum gas that can be applied based on your current velocity, the relative velocity of the lead, the acceleration of the lead, and the distance of the lead. This usually results in quicker and smoother acceleration from a standstill without the jerking you get in stock openpilot with comma pedal (ex. taking off from a traffic light). It tries to coast if the lead is just inching up, it doesn’t use maximum gas as soon as the lead inches forward. When you are above 20 mph, relative velocity and the current following distance in seconds is taken into consideration.

All cars that have a comma pedal are supported! However to get the smoothest acceleration, I've custom tuned gas curve profiles for the following cars:

pedal cars:
  * 2017 Toyota Corolla (non-TSS2)
  * Toyota RAV4 (non-TSS2)
  * 2017 Honda Civic
  * 2019 Honda Pilot

non-pedal cars:
  * None yet

If you have a car without a pedal, or you do have one but I haven't created a profile for you yet, please let me know and we can develop one for your car to test.

---
### PI -> PID Controller for Long and Lat
(long: longitudinal, speed control. lat: latitudinal, steering control)

**Changes for lat control: (NEW❗)**
- Adding the derivative componenet to lat control greatly improves the turning performance of openpilot, I've found it loses control much less frequently in both slight and sharp curves and smooths out steering in all situations. Basically it ramps down torque as your wheel approaches the desired angle, and ramps up torque quicky when your wheel is moving away from desired.

  ***Currently Supported Cars: (when param `use_lqr` is False)***
  - 2017 Toyota Corolla
  - TSS2 Toyota Corolla (when param `corollaTSS2_use_indi` is False) - tune from birdman!
  - All Prius years (when param `prius_use_pid` is True) - tune from [Trae](https://github.com/d412k5t412)!

**Changes for long control:**
- I've added a custom implementation of derivative to the PI loop controlling the gas and brake output sent to your car. Derivative (change in error) is calculated based on the current and last error and added to the class's integral variable. It's essentially winding down integral according to derivative. It helps fix overshoot on some cars with the comma pedal and increases responsiveness (like when going up and down hills) on all other cars! Still need to figure out the tuning, right now it's using the same derivative gain for all cars. Test it out and let me know what you think!

  Long derivative is disabled by default due to only one tune for all cars, but can be enabled by using [opEdit](#Customize-this-fork-opEdit) and setting the `enable_long_derivative` parameter to `True`. It works well on my '17 Corolla with pedal.

---
## General Features

### Compensate for longitudinal delay for earlier braking
This feature mod was so good it was added to stock openpilot! The only change now on here is that it's adjusted a bit based on speed, and a uses different delay for desired acceleration vs. speed.

---
### Customize this fork (opEdit)
This is a handy tool to change your `opParams` parameters without diving into any json files or code. You can specify parameters to be used in any fork's operation that supports `opParams`. First, ssh in to your EON and make sure you're in `/data/openpilot`, then start `opEdit`:
```python
cd /data/openpilot
python op_edit.py  # or ./op_edit.py
```

[**To see what features opEdit has, click me!**](/OPEDIT_FEATURES.md)

🆕 All params now update live while driving, and params that are marked with `static` need a reboot of the device, or ignition.

Here are the main parameters you can change with this fork:
- **Tuning params**:
  - `camera_offset` **`(live!)`**: Your camera offset to use in lane_planner.py. Helps fix lane hugging
  - `steer_ratio` **`(live!)`**: The steering ratio you want to use with openpilot. If you enter None, it will use the learned steer ratio from openpilot instead
  - [`enable_long_derivative`](#pi---pid-controller-for-long-and-lat): This enables derivative-based integral wind-down to help overshooting within the PID loop. Useful for Toyotas with pedals or cars with bad long tuning
  - [`use_lqr`](#pi---pid-controller-for-long-and-lat): Enable this to use LQR for lateral control with any car. It uses the RAV4 tuning, but has proven to work well for many cars
- **General fork params**:
  - `alca_no_nudge_speed`: Above this speed (mph), lane changes initiate IMMEDIATELY after turning on the blinker. Behavior is stock under this speed (waits for torque)
  - `upload_on_hotspot`: Controls whether your EON will upload driving data on your phone's hotspot
  - [`update_behavior`](#Automatic-updates): `off` will never update, `alert` shows an alert on-screen. `auto` will reboot the device when an update is seen
  - `disengage_on_gas`: Whether you want openpilot to disengage on gas input or not
  - `hide_model_long`: Enable this to hide the Model Long button on the screen
- **Dynamic params**:
  - `dynamic_gas`: Whether to use [dynamic gas](#dynamic-gas) if your car is supported
  - `global_df_mod` **`(live!)`**: The multiplier for the current distance used by dynamic follow. The range is limited from 0.85 to 2.5. Smaller values will get you closer, larger will get you farther. This is applied to ALL profiles!
  - `min_TR` **`(live!)`**: The minimum allowed following distance in seconds. Default is 0.9 seconds, the range of this mod is limited from 0.85 to 1.3 seconds. This is applied to ALL profiles!
  - `hide_auto_df_alerts`: Hides the alert that shows what profile the model has chosen
  - `df_button_alerts`: How you want to be alerted when you change your dynamic following profile, can be: 'off', 'silent', or 'audible' (default)
  - `toyota_distance_btn`: Set to True to use the steering wheel distance button on Toyota vehicles to control the dynamic follow profile.
  Works on TSS2 vehicles and on TSS1 vehicles with an sDSU with a [Sep. 2020](https://github.com/wocsor/panda/commit/b5120f6427551345c543b490fe47da189c1e48e1) firmware or newer.'
  - [`dynamic_camera_offset`](#Dynamic-camera-offset-based-on-oncoming-traffic): Whether to automatically keep away from oncoming traffic. Works from 35 to ~60 mph
    - [`dynamic_camera_offset_time`](#Dynamic-camera-offset-based-on-oncoming-traffic): How long to keep the offset after losing the oncoming lane/radar track in seconds
  - `dynamic_follow`: *Deprecated, use the on-screen button to change profiles*
- **Experimental params**:
  - `support_white_panda`: This allows users with the original white panda to use openpilot above 0.7.7. The high precision localizer's performance may be reduced due to a lack of GPS
  - [`prius_use_pid`](#pi---pid-controller-for-long-and-lat): This enables the PID lateral controller with new a experimental derivative tune
  - `corollaTSS2_use_indi`: Enable this to use INDI for lat with your TSS2 Corolla *(can be enabled for all years by request)*
  - `standstill_hack`: Some cars support stop and go, you just need to enable this

A full list of parameters that you can modify are [located here](common/op_params.py#L101).

An archive of opParams [lives here.](https://github.com/sshane/op_params)

Parameters are stored at `/data/op_params.json`

---
### opEdit Demo
<img src=".media/op_edit.gif?raw=true" width="1000">

---
### Automatic updates
When a new update is available on GitHub for Stock Additions, your EON/C2 will pull and reset your local branch to the remote. It then queues a reboot to occur when the following is true:
- your EON has been inactive or offroad for more than 5 minutes
- `update_behavior` param is set to `auto`

Therefore, if your device sees an update while you're driving it will reboot approximately 5 to 10 minutes after you finish your drive, it resets the timer if you start driving again before the time is up.

---
### ZSS Support
If you have a Prius with a ZSS ([Zorrobyte](https://github.com/zorrobyte) Steer Sensor), you can use this fork to take full advantage of your high-precision angle sensor! Added support for ZSS with [PR #198](https://github.com/sshane/openpilot/pull/198), there's nothing you need to do. Special thanks to [Trae](https://github.com/d412k5t412) for helping testing the addition!

If you have a ZSS but not a Prius, let me know and I can add support for your car.

---
### Offline crash logging
If you experience a crash or exception while driving with this fork, and you're not on internet for the error to be uploaded to Sentry, you should be able to check out the directory `/data/community/crashes` to see any and all logs of exceptions caught in openpilot. Simply view the logs with `ls -lah` and then `cat` the file you wish to view by date. This does not catch all errors, for example scons compilation errors or some Python syntax errors will not be caught, `tmux a` is usually best to view these (if openpilot didn't start).

❗***Quickly view the latest crash:*** `cat /data/community/crashes/latest.log`

Feel free to reach out to me on [Discord](#stock-additions-v066-082) if you're having any issues with the fork!

---
## Documentation

### Quick Installation
To install Stock Additions, just enter the following URL on the setup screen for "Custom Software" after you factory reset:

```
https://smiskol.com/fork/sshane
```

- *Or use the [emu CLI](https://github.com/emu-sh/.oh-my-comma) to easily switch to this fork's default branch: `emu fork switch sshane`. The initial setup may take longer than the above method, but you gain the ability to quickly switch to any fork you want.*
- *Or run the following commands in an ssh terminal on your device:*

  ```
  cd /data/
  mv openpilot openpilot.old  # or equivalent
  git clone -b SA-master --depth 1 --recurse-submodules https://github.com/sshane/openpilot
  sudo reboot
  ```

---
### Branches
Most of the branches on this fork are development branches I use as various openpilot tests. The few that more permanent are the following:
  * [`SA-master`](https://github.com/sshane/openpilot/tree/SA-master): My development branch of Stock Additions I use to test new features or changes; similar to the master branch. Not recommended as a daily driver.
  * [`SA-release`](https://github.com/sshane/openpilot/tree/SA-release): This is similar to stock openpilot's release branch. Will receive occasional and tested updates to Stock Additions.

---
### Archive Stock Additions branches
* [Stock Additions 0.7](https://github.com/sshane/openpilot-archive/tree/stock_additions-07)
* [Stock Additions 0.7.1](https://github.com/sshane/openpilot-archive/tree/stock_additions-071)
* [Stock Additions 0.7.4](https://github.com/sshane/openpilot-archive/tree/stock_additions-074)
* [Stock Additions 0.7.5](https://github.com/sshane/openpilot-archive/tree/stock_additions-075)
* [Stock Additions 0.7.7](https://github.com/sshane/openpilot-archive/tree/stock_additions-077)
* [Stock Additions 0.7.10](https://github.com/sshane/openpilot-archive/tree/stock_additions-0710)
* [Stock Additions 0.8](https://github.com/sshane/openpilot-archive/tree/stock_additions-08)
* [Stock Additions 0.8.2](https://github.com/sshane/openpilot-archive/tree/stock_additions-082)

---
### Videos
Here's a short video showing how close the traffic profile was in `0.7.4`. In `0.7.5`, the traffic profile is an average of 7.371 feet closer from 18 mph to 90 mph. Video thanks to [@rolo01](https://github.com/rolo01)!

[![](https://img.youtube.com/vi/sGsODeP_G_c/0.jpg)](https://www.youtube.com/watch?v=sGsODeP_G_c)

---
If you'd like to support my development of Stock Additions with a [dollar for a RaceTrac ICEE](https://paypal.me/ssmiskol) (my PayPal link). Thanks! 🥰
