![](https://user-images.githubusercontent.com/47793918/233812617-beab2e71-57b9-479e-8bff-c3931347ca40.png)

Table of Contents
=======================

* [Join our Discord](#-join-our-discord)
* [What is sunnypilot?](#-what-is-sunnypilot)
* [Running in a car](#-running-on-a-dedicated-device-in-a-car)
* [Read Before Installing](#-read-before-installing)
* [Prohibited Safety Modifications](#-prohibited-safety-modifications)
* [Installation](#-installation)
* [Highlight Features](#-highlight-features)
* [Driving Enhancements](#-driving-enhancements)
* [Branch Definitions](#-branch-definitions)
* [Recommended Branches](#-recommended-branches)
* [How-To's](#-How-Tos)
* [Pull Requests](#-Pull-Requests)
* [Special Thanks](#-special-thanks)
* [User Data](#-user-data)
* [Licensing](#licensing)
* [Donate](#-support-sunnypilot)

---

<details><summary><h3>💭 Join our Discord</h3></summary>

---

Join the official sunnypilot Discord server to stay up to date with all the latest features and be a part of shaping the future of sunnypilot!
* https://discord.gg/sunnypilot

  ![](https://dcbadge.vercel.app/api/server/wRW3meAgtx?style=flat) ![Discord Shield](https://discordapp.com/api/guilds/880416502577266699/widget.png?style=shield)

</details>

<details><summary><h3>🌞 What is sunnypilot?</h3></summary>

---

[sunnypilot](https://github.com/sunnyhaibin/sunnypilot) is a fork of comma.ai's openpilot, an open source driver assistance system. sunnypilot offers the user a unique driving experience for over 250+ supported car makes and models with modified behaviors of driving assist engagements. sunnypilot complies with comma.ai's safety rules as accurately as possible.

</details>

<details><summary><h3>🚘 Running on a dedicated device in a car</h3></summary>

---

To use sunnypilot in a car, you need the following:
* A supported device to run this software
    * a [comma three](https://comma.ai/shop/products/three), or
    * a comma two (only with older versions below 0.8.13)
* This software
* One of [the 250+ supported cars](https://github.com/commaai/openpilot/blob/master/docs/CARS.md). We support Honda, Toyota, Hyundai, Nissan, Kia, Chrysler, Lexus, Acura, Audi, VW, Ford and more. If your car is not supported but has adaptive cruise control and lane-keeping assist, it's likely able to run sunnypilot.
* A [car harness](https://comma.ai/shop/products/car-harness) to connect to your car

Detailed instructions for [how to mount the device in a car](https://comma.ai/setup).

</details>

<details><summary><h3>🚨 Read Before Installing</h3></summary>

---

It is recommended to read this **entire page** before proceeding. This will ensure that you fully understand each added feature on sunnypilot, and you are selecting the right branch for your car to have the best driving experience.

This is a fork of [comma.ai's openpilot](https://github.com/commaai/openpilot). By installing this software, you accept all responsibility for anything that might occur while you use it. All contributors to sunnypilot are not liable. ❗<ins>**Use at your own risk.**</ins>❗

</details>

<details><summary><h3>⛔ Prohibited Safety Modifications</h3></summary>

---

All [official sunnypilot branches](https://github.com/sunnyhaibin/sunnypilot/branches) strictly adhere to [comma.ai's safety policy](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Any changes that go against this policy will result in your fork and your device being banned from both comma.ai and sunnypilot channels.

The following changes are a **VIOLATION** of this policy and **ARE NOT** included in any sunnypilot branches:
* Driver Monitoring:
    * ❌ "Nerfing" or reducing monitoring parameters.
* Panda safety:
    * ❌ No preventing disengaging of <ins>**LONGITUDINAL CONTROL**</ins> (acceleration/brake) on brake pedal press.
    * ❌ No auto re-engaging of <ins>**LONGITUDINAL CONTROL**</ins> (acceleration/brake) on brake pedal release.
    * ❌ No disengaging on ACC MAIN in OFF state.

</details>


<details><summary><h3>⚒ Installation</h3></summary>

---

  <details><summary>URL (Easy)</summary>

comma three
------

Please refer to [Recommended Branches](#-recommended-branches) to find your preferred/supported branch. This guide will assume you want to install the latest `release-c3` branch.

* sunnypilot not installed or you installed a version before 0.8.17?
  1. [Factory reset/uninstall](https://github.com/commaai/openpilot/wiki/FAQ#how-can-i-reset-the-device) the previous software if you have another software/fork installed.
  2. After factory reset/uninstall and upon reboot, select `Custom Software` when given the option.
  3. Input the installation URL per [Recommended Branches](#-recommended-branches). Example: ```bit.ly/sp-release-c3``` [^4] (note: `https://` is not requirement on the comma three)
  4. Complete the rest of the installation following the onscreen instructions.

* sunnypilot already installed and you installed a version after 0.8.17?
  1. On the comma three, go to `Settings` ▶️ `Software`.
  2. At the `Download` option, press `CHECK`. This will fetch the list of latest branches from sunnypilot.
  3. At the `Target Branch` option, press `SELECT` to open the Target Branch selector.
  4. Scroll to select the desired branch per [Recommended Branches](#-recommended-branches). Example: `release-c3`

Requires further assistance with software installation? Join the [sunnypilot Discord server](https://discord.sunnypilot.com) and message us in the `#installation-help` channel.

comma two
------

1. [Factory reset/uninstall](https://github.com/commaai/openpilot/wiki/FAQ#how-can-i-reset-the-device) the previous software if you have another software/fork installed.
2. After factory reset/uninstall and upon reboot, select `Custom Software` when given the option.
3. Input the installation URL per [Recommended Branches](#-recommended-branches). Example: ```https://smiskol.com/fork/sunnyhaibin/0.8.12-4-prod```
4. Complete the rest of the installation following the onscreen instructions.

Requires further assistance with software installation? Join the [sunnypilot Discord server](https://discord.sunnypilot.com) and message us in the `#installation-help` channel.

  </details>

  <details>
  <summary>SSH (More Versatile)</summary>
  <br>

Prerequisites: [How to SSH](https://github.com/commaai/openpilot/wiki/SSH)

If you are looking to install sunnypilot via SSH, run the following command in an SSH terminal after connecting to your device:

comma three:
------
* [`release-c3`](https://github.com/sunnyhaibin/openpilot/tree/release-c3):

  ```
  cd /data; rm -rf ./openpilot; git clone -b release-c3 --recurse-submodules https://github.com/sunnyhaibin/sunnypilot.git openpilot; cd openpilot; sudo reboot
  ```

comma two:
------
* [`0.8.12-prod-personal-hkg`](https://github.com/sunnyhaibin/openpilot/tree/0.8.12-prod-personal-hkg):

  ```
  cd /data; rm -rf ./openpilot; git clone -b 0.8.12-prod-personal-hkg --recurse-submodules https://github.com/sunnyhaibin/sunnypilot.git openpilot; cd openpilot; sudo reboot
  ```

After running the command to install the desired branch, your comma device should reboot.
  </details>

</details>


<details><summary><h3>🚗 Highlight Features</h3></summary>

---

### Quality of Life Enhancements
- [**Modified Assistive Driving Safety (MADS)**](#modified-assistive-driving-safety-mads) - Automatic Lane Centering (ALC) / Lane Keep Assist System (LKAS) and Adaptive Cruise Control (ACC) / Smart Cruise Control (SCC) can be engaged independently of each other
- [**Dynamic Lane Profile (DLP)**](#dynamic-lane-profile-dlp) - Dynamically switch lane profile (between Laneful and Laneless) based on lane recognition confidence
- [**Enhanced Speed Control**](#enhanced-speed-control) - Automatically adjust cruise control speed using vision model, OpenStreetMap (OSM) data, and/or Speed Limit control (SLC) without user interaction
    * Vision-based Turn Speed Control (V-TSC) - lower speed when going around corners using vision model
    * Map-Data-based Turn Speed Control (M-TSC) - lower speed when going around corners using OSM data[^1]
    * Speed Limit Control (SLC) - Set speed limit based on map data or car interface (if applicable)
    * HKG only: Highway Driving Assist (HDA) status integration - Use cars native speed sign detection to set desired speed (on applicable HKG cars only)
- [**Gap Adjust Cruise (GAC)**](#gap-adjust-cruise) - Allow `GAP`/`INTERVAL`/`DISTANCE` button on the steering wheel or on-screen button to adjust the follow distance from the lead car. See table below for options
    - [**Quiet Drive 🤫**](#-quiet-drive) - Toggle to mute all notification sounds (excluding driver safety warnings)
    - [**Auto Lane Change Timer**](#Auto-Lane-Change-Timer) - Set a timer to delay the auto lane change operation when the blinker is used. No nudge on the steering wheel is required to auto lane change if a timer is set
    - [**Force Car Recognition (FCR)**](#Force-Car-Recognition-) - Use a selector to force your car to be recognized by sunnypilot
    - [**Fix sunnypilot No Offroad**](#Fix-sunnypilot-No-Offroad) - Enforce sunnypilot to go offroad and turns off after shutting down the car. This feature fixes non-official devices running sunnypilot without comma power
    - [**Enable ACC+MADS with RES+/SET-**](#Enable-ACC+MADS-with-RES+/SET-) - Engage both ACC and MADS with a single press of RES+ or SET- button
    - [**Offline OSM Maps**](#Offline-OSM-Maps) - OSM database can now be downloaded locally for offline use[^2]. This enables offline SLC, V-TSC and M-TSC. Currently available for US South, US West, US Northeast, Florida, Taiwan, South Africa and New Zealand
    - [**Various Live Tuning**](#Various-Live-Tuning) - Ability to tailor your driving experience on the fly:
        * Enforce Torque Lateral Control - Use the newest [torque controller](https://blog.comma.ai/0815release/#torque-controller) for all vehicles.
        * Torque Lateral Control Live Tune - Ability to adjust the torque controller’s `FRICTION` and `LAT_ACCEL_FACTOR` values to suit your vehicle.
        * Torque Lateral Controller Self-Tune - Enable automatic turning for the Torque controller.

### Visual Enhancements
* **M.A.D.S Status Icon** - Dedicated icon to display M.A.D.S. engagement status
    * Green🟢: M.A.D.S. engaged
    * White⚪: M.A.D.S. suspended or disengaged
* **Lane Path Color** - Various lane path colors to display real-time Lane Model and M.A.D.S. engagement status
    * 0.8.14 and later:
        * Blue🔵: Laneful mode & M.A.D.S. engaged
        * Green🟢: Laneless mode & M.A.D.S. engaged
        * Yellow🟡: Experimental e2e & M.A.D.S. engaged
    * Pre 0.8.14:
        * Green🟢: Laneful mode & M.A.D.S. engaged
        * Red🔴: Laneless mode & M.A.D.S. engaged
    * White⚪: M.A.D.S. suspended or disengaged
    * Black⚫: M.A.D.S. engaged, steering is being manually overridden by user
* **Developer (Dev) UI** - Display various real-time metrics on screen while driving
* **Stand Still Timer** - Display time spent at a stop with M.A.D.S engaged (i.e., at traffic lights, stop signs, traffic congestions)
* **Braking Status** - Current car speed text turns red when the car is braking by the driver or ACC/SCC

### Operational Enhancements
* **Fast Boot** - sunnypilot will fast boot by creating a Prebuilt file
* **Disable Onroad Uploads** - Disable uploads completely when onroad. Necessary to avoid high data usage when connected to Wi-Fi hotspot
* **Brightness Control (Global)** - Manually adjusts the global brightness of the screen
* **Driving Screen Off Timer** - Turns off the device screen or reduces brightness to protect the screen after car starts
* **Driving Screen Off Brightness (%)** - When using the Driving Screen Off feature, the brightness is reduced according to the automatic brightness ratio
* **Max Time Offroad** - Device is automatically turned off after a set time when the engine is turned off (off-road) after driving (on-road)

</details>

<details><summary><h3>🚗 Driving Enhancements</h3></summary>

---

### Modified Assistive Driving Safety (MADS)
The goal of Modified Assistive Driving Safety (MADS) is to enhance the user driving experience with modified behaviors of driving assist engagements. This feature complies with comma.ai's safety rules as accurately as possible with the following changes:
* sunnypilot Automatic Lane Centering (ALC) and ACC/SCC can be engaged independently of each other
* Dedicated button to toggle sunnypilot ALC:
    * `CRUISE (MAIN)` button: All supported cars on sunnypilot
        * `LFA` button: Newer HKG cars with `LFA` button
        * `LKAS` button: Honda, Toyota, Global Subaru
* `SET-` button enables ACC/SCC
* `CANCEL` button only disables ACC/SCC
* `CRUISE (MAIN)` must be `ON` to use ACC/SCC
* `CRUISE (MAIN)` button disables sunnypilot completely when `OFF` **(strictly enforced in panda safety code)**

### Disengage Lateral ALC on Brake Press Mode toggle
Dedicated toggle to handle Lateral state on brake pedal press and release:
1. `ON`: `BRAKE pedal` press will pause Automatic Lane Centering; `BRAKE pedal` release will resume Automatic Lane Centering. Note: `BRAKE pedal` release will NOT resume ACC/SCC/Long control without explicit user engagement **(strictly enforced in panda safety code)**
2. `OFF`: `BRAKE pedal` press will NOT pause Automatic Lane Centering; `BRAKE pedal` release will NOT resume ACC/SCC/Long control without explicit user engagement **(strictly enforced in panda safety code)**

### Miscellaneous
* `TURN SIGNALS` (`Left` or `Right`) will pause Automatic Lane Centering if the vehicle speed is below the [threshold](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/desire_helper.py#L8) for Automatic Lane Change
* Event audible alerts are more relaxed to match manufacturer's stock behavior
* Critical events trigger disengagement of Automatic Lane Centering completely. The disengagement is enforced in sunnypilot and panda safety

### Dynamic Lane Profile (DLP)

Dynamic Lane Profile (DLP) aims to provide the best driving experience at staying within a lane confidently. Dynamic Lane Profile allows sunnypilot to dynamically switch between lane profiles based on lane recognition confidence level on road.

There are 3 modes to select on the onroad camera screen:
* **Auto Lane**: sunnypilot dynamically chooses between `Laneline` or `Laneless` model
* **Laneline**: sunnypilot uses Laneline model only.
* **Laneless**: sunnypilot uses Laneless model only.

To use Dynamic Lane Profile, do the following:
```
1. sunnypilot Settings -> `SP - Controls` -> Enable Dynamic Lane Profile -> ON toggle
2. Reboot.
3. Before driving, on the onroad camera screen, toggle between the 3 modes by pressing on the button.
4. Drive.
```

### Enhanced Speed Control
This fork now allows supported cars to dynamically adjust the longitudinal plan based on the fetched map data. Big thanks to the Move Fast team for the amazing implementation!

**Supported cars:**
* sunnypilot Longitudinal Control capable
* Stock Longitudinal Control
    * Hyundai/Kia/Genesis (non CAN-FD)
    * Honda Bosch
    * Volkswagen MQB

Certain features are only available with an active data connection, via:
* [comma Prime](https://comma.ai/prime) - Intuitive service provided directly by comma, or
* Personal Hotspot - From your mobile device, or a dedicated hotspot from a cellular carrier.

**Features:**
* Vision-based Turn Speed Control (VTSC) - Use vision path predictions to estimate the appropriate speed to drive through turns ahead - i.e. slowing down for curves
* Map-Data-based Turn Speed Control (MTSC) - Use curvature information from map data to define speed limits to take turns ahead - i.e. slowing down for curves[^1]
* Speed Limit Control (SLC) - Use speed limit signs information from map data and car interface to automatically adapt cruise speed to road limits
    * HKG only: Highway Driving Assist (HDA) status integration - on applicable HKG cars only[^1]
    * Speed Limit Offset - When Speed Limit Control is enabled, set speed limit slightly higher than the actual speed limit for a more natural drive[^1]
* Toggle Hands on Wheel Monitoring - Monitors and alerts the driver when their hands have not been on the steering wheel for an extended time

### Custom Stock Longitudinal Control
While using stock Adaptive/Smart Cruise Control, Custom Stock Longitudinal Control in sunnypilot allows sunnypilot to manipulate and take over the set speed on the car's dashboard.

**Supported Cars:**
* Hyundai/Kia/Genesis
    * CAN platform
    * CAN-FD platform with 0x1CF broadcasted in CAN traffic
* Honda Bosch
* Volkswagen MQB

**Instruction**

**📗 How to use Custom Longitudinal Control on sunnypilot **

When using Speed Limit, Vision, or Map based Turn control, you will be setting the "MAX" ACC speed on the sunnypilot display instead of the one in the dashboard. The car will then set the ACC setting in the dashboard to the targeted speed, but will never exceed the max speed set on the sunnypilot display. A quick press of the RES+ or SET- buttons will change this speed by 5 MPH or KM/H on the sunnypilot display, while a long deliberate press (about a 1/2 second press) changes it by 1 MPH or KM/H. DO NOT hold the RES+ or SET- buttons for longer that a 1 second. Either make quick or long deliberate presses only.

**‼ Where to look when setting ACC speed ‼**

Do not look at the dashboard when setting your ACC max speed. Instead, only look at the one on the sunnypilot display, "MAX". The reason you need to look at the sunnypilot display is because sunnypilot will be changing the one in the dashboard. It will be adjusting it as needed, never raising it above the one set on the sunnypilot display. ONLY look at the MAX speed on the sunnypilot display when setting the ACC speed instead of the dashboard!

(Courtesy instructions from John, author of jvePilot)

### Gap Adjust Cruise
This fork now allows supported openpilot longitudinal cars to adjust the cruise gap between the car and the lead car.

**Supported cars:**
* sunnypilot Longitudinal Control capable

🚨**PROCEED WITH EXTREME CAUTION AND BE READY TO MANUALLY TAKE OVER AT ALL TIMES**

There are 4 modes to select on the steering wheel and/or the onroad camera screen:
* **Stock Gap**: Stock sunnypilot distance - 1.45 second profile
* **Mild Gap**: Semi-aggressive distance - 1.25 second profile
* 🚨**Aggro Gap**🚨: Aggressive distance - 1.0 second profile

**Availability**

|      Car Make       | Stock Gap | Mild Gap | Aggro Gap |
|:-------------------:|:---------:|:--------:|:---------:|
|     Honda/Acura     |     ✅     |    ✅     |     ✅     |
| Hyundai/Kia/Genesis |     ✅     |    ✅     |     ✅     |
|    Toyota/Lexus     |     ✅     |    ✅     |     ✅     |
|  Volkswagen MQB/PQ  |     ✅     |    ✅     |     ✅     |

</details>


<details><summary><h3>⚒ Branch Definitions</h3></summary>

---

|    Tag    | Definition           | Description                                                                                                                                                                                 |
|:---------:|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `release` | Release branches     | Include features that are **verified** by trusted testers and the community. Ready to use. ✅                                                                                                |
| `staging` | Staging branches     | Include new features that are **tested** by trusted testers and the community. Stability may vary. ⚠                                                                                        |
|   `dev`   | Development branches | All features are gathered in respective versions. Reviewed and merged features will be committed to `dev`. Stability may vary. ⚠                                                            |
| `master`  | Main branch          | Syncs with [commaai's openpilot `master`](https://github.com/commaai/openpilot) upstream branch. Accepts all pull requests. Does not include all sunnypilot features. Stability may vary. ⚠ |

Example:
* [`release-c3`](https://github.com/sunnyhaibin/sunnypilot/tree/release-c3): Latest release branch for comma three that are verified by trusted testers and the community. Ready to use.
* [`staging-c3`](https://github.com/sunnyhaibin/sunnypilot/tree/staging-c3): Latest staging branch for comma three that are tested by trusted testers and the community. Verification required.
* [`dev-c3`](https://github.com/sunnyhaibin/sunnypilot/tree/dev-c3): Latest development branch for comma three that include all sunnypilot features. Testing required.

</details>

<details><summary><h3>✅ Recommended Branches</h3></summary>

---

| Branch                                                                              | Definition                                              | Compatible Device | Changelogs                                                                                 |
|:------------------------------------------------------------------------------------|---------------------------------------------------------|-------------------|--------------------------------------------------------------------------------------------|
| [`release-c3`](https://github.com/sunnyhaibin/sunnypilot/tree/release-c3)           | • Latest release/stable branch                          | comma three       | [`CHANGELOGS.md`](https://github.com/sunnyhaibin/sunnypilot/blob/release-c3/CHANGELOGS.md) |
| [`staging-c3`](https://github.com/sunnyhaibin/sunnypilot/tree/staging-c3)           | • Latest staging branch                                 | comma three       | [`CHANGELOGS.md`](https://github.com/sunnyhaibin/sunnypilot/blob/staging-c3/CHANGELOGS.md) |
| [`dev-c3`](https://github.com/sunnyhaibin/sunnypilot/tree/dev-c3)                   | • Latest development branch with experimental features  | comma three       | [`CHANGELOGS.md`](https://github.com/sunnyhaibin/sunnypilot/blob/dev-c3/CHANGELOGS.md)     |

</details>

<details><summary><h3>📗 How To's</h3></summary>

---

How-To instructions can be found in [HOW-TOS.md](https://github.com/sunnyhaibin/openpilot/blob/(!)README/HOW-TOS.md).

</details>


<details><summary><h3>🎆 Pull Requests</h3></summary>

---

We welcome both pull requests and issues on GitHub. Bug fixes are encouraged.

Pull requests should be against the most current `master` branch.

</details>

<details><summary><h3>🏆 Special Thanks</h3></summary>

---

* [spektor56](https://github.com/spektor56/openpilot)
* [rav4kumar](https://github.com/rav4kumar/openpilot)
* [mob9221](https://github.com/mob9221/opendbc)
* [briantran33](https://github.com/briantran33/openpilot)
* [Aragon7777](https://github.com/aragon7777/openpilot)
* [sshane](https://github.com/sshane/openpilot-installer-generator)
* [jung](https://github.com/chanhojung/openpilot)
* [dri94](https://github.com/dri94/openpilot)
* [FrogAi](https://github.com/frogAi/FrogPilot/)
* [twilsonco](https://github.com/twilsonco/openpilot)
* [martinl](https://github.com/martinl/openpilot)
* [multikyd](https://github.com/openpilotkr)
* [Move Fast GmbH](https://github.com/move-fast/openpilot)
* [dragonpilot](https://github.com/dragonpilot-community/dragonpilot)
* [neokii](https://github.com/neokii/openpilot)
* [AlexandreSato](https://github.com/AlexandreSato/openpilot)
* [Moodkiller](https://github.com/moodkiller)

</details>

<details><summary><h3>📊 User Data</h3></summary>

---

By default, sunnypilot uploads the driving data to comma servers. You can also access your data through [comma connect](https://connect.comma.ai/).

sunnypilot is open source software. The user is free to disable data collection if they wish to do so.

sunnypilot logs the road-facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera is only logged if you explicitly opt-in in settings. The microphone is not recorded.

By using this software, you understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.

</details>

<details><summary><h3>Licensing</h3></summary>

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

</details>

<h3>💰 Support sunnypilot</h3>

---

If you find any of the features useful, consider becoming a [patron on Patreon](https://www.patreon.com/sunnyhaibin) or a [sponsor on GitHub](https://github.com/sponsors/sunnyhaibin) to support future feature development and improvements.


By becoming a patron/sponsor, you will gain access to exclusive content, early access to new features, and the opportunity to directly influence the project's development.

<h3>Patreon</h3>

<a href="https://www.patreon.com/sunnyhaibin">
  <img src="https://user-images.githubusercontent.com/47793918/244128051-bc7e913e-a196-4455-926e-23aec9a4bd3b.png" alt="Become a Patron" width="300" style="max-width: 100%; height: auto;">
</a>
<br>

<h3>GitHub Sponsor</h3>

<a href="https://github.com/sponsors/sunnyhaibin">
  <img src="https://user-images.githubusercontent.com/47793918/244135584-9800acbd-69fd-4b2b-bec9-e5fa2d85c817.png" alt="Become a Sponsor" width="300" style="max-width: 100%; height: auto;">
</a>
<br>

<h3>PayPal</h3>

<a href="https://paypal.me/sunnyhaibin0850" target="_blank">
<img src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" alt="PayPal this" title="PayPal - The safer, easier way to pay online!" border="0" />
</a>
<br></br>

Your continuous love and support are greatly appreciated! Enjoy 🥰

<span>-</span> Jason, Founder of sunnypilot

[^1]:Requires data connection if not using Offline Maps data
[^2]:At least 50 GB of storage space is required. If you have the 32 GB version of comma three, upgrading with a compatible 250 GB or 1 TB SSD is strongly recommended
[^4]:Shortened URL for convenience. Full URL is ```smiskol.com/fork/sunnyhaibin/release-c3```
