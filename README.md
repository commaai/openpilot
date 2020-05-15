HKG Community information
======

This is the "Community fork" for Kia, Hyundai and Genesis.
it is a fork of comma's openpilot: https://github.com/commaai/openpilot. It is open source and inherits MIT license.  By installing this software you accept all responsibility for anything that might occur while you use it.  All contributors to this fork are not liable.  <b>Use at your own risk.</b>

<b>The port was started by Andrew Frahn of Emmertex, ku7 tech on youtube
https://www.youtube.com/c/ku7tech
I am going to try to mintain this fork for the commuinty, if you like it you can support me from here:  [Donate](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=JX92RNKXRPJKN&currency_code=USD&source=url)</b>

Changes:
------

- <b>  Auto LCA:</b> credit to @SiGmAX666: Auto Lane change assist, no need for steering nudge. LCA will abort if driver override steering wheel. To enable Auto LCA(disabled by default),  change line 189 in selfdrive/car/hyundai/interface.py to:
```python
    ret.autoLcaEnabled = 1
```
- <b>  Enable by Cruise button:</b> Only for Car without long control, Openpilot will engage when turn cruise control on. To revert to SET button for enable, change line 54 in selfdrive/car/hyundai/carcontroller.py to:
```python
    self.longcontrol = 1
```
- <b>  Turning disable:</b> thank to Ku7: Openpilot will disable steering while turning signal on and speed below 60 kph, Enable again after 1 second. 
- <b>  Increase driver monitoring timer</b>  
- <b>  Disabling by LKAS button:</b> Openpilot will disable and enable steering by toggling LKAS button.
- <b>  Setup Auto Detection:</b> Openpilot and Panda will detect MDPS, SCC and SAS buses and behaive acordingly.
- <b>  Panda Universal Forwarding(PUF):</b> Panda will auto forwading for all CAN messages if Eon disconnected.




Known issues
------
(temporarily fixed)LKAS fauls when driver override steering in opposite direction of Openpilot, it cause by Panda safety bolcking LKAS messages.

HKG Supported Cars
------

To add new car or fingerprint, please make Pull Requset or send me the fingerprint along with below information:

| Make      | Model (US Market Reference)        | Supported Package | ACC              | No ACC accel below | No ALC below |
| ----------| -----------------------------------| ------------------| -----------------| -------------------| -------------|
| Genesis   | G80 2018                           | All               | Stock            | 0mph               | 0mph         |
| Genesis   | G90 2018                           | All               | Stock            | 0mph               | 0mph         |
| Hyundai   | Elantra 2017-19<sup>5</sup>        | SCC + LKAS        | Stock            | 19mph              | 34mph        |
| Hyundai   | Elantra GT/i30 2017-19             | All               | Stock            | 0mph               | 30mph        |
| Hyundai   | Genesis 2018                       | All               | Stock            | 19mph              | 34mph        |
| Hyundai   | Ioniq 2017<sup>5</sup>             | All               | Stock            | 0mph               | 34mph        |
| Hyundai   | Kona 2017-19<sup>5</sup>           | LDWS              | Stock            | 22mph              | 0mph         |
| Hyundai   | Santa Fe 2019<sup>5</sup>          | All               | Stock            | 0mph               | 0mph         |
| Kia       | Forte 2018<sup>5</sup>             | LKAS              | Stock            | 0mph               | 0mph         |
| Kia       | Forte 2019<sup>5</sup>             | LKAS              | Stock            | 0mph               | 0mph         |
| Kia       | Ceed 2019                          | SCC + LKAS + LFA  | Stock            | 0mph               | 0mph         |
| Kia       | Optima 2017<sup>5</sup>            | SCC + LKAS/LDWS   | Stock            | 0mph               | 34mph        |
| Kia       | Optima 2019<sup>5</sup>            | SCC + LKAS        | Stock            | 0mph               | 0mph         |
| Kia       | Sorento 2018<sup>5</sup>           | All               | Stock            | 0mph               | 0mph         |
| Kia       | Stinger 2018<sup>5</sup>           | SCC + LKAS        | Stock            | 0mph               | 0mph         |

Known issues
------




Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
