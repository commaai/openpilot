<b>NOTE:  If you have upgraded at any time to v0.5.10 and you want to go back to a branch with v0.5.9 or v0.5.8, then you have to SSH into the Eon and edit the file /data/params/d/ControlsParams and rename "angle_model_bias" to "angle_offset" or your car will have Dash Errors and you'll be scratching your head for hours! 

Pedal Users: Also note that you need to flash your Pedal to go to v0.5.10.  If you want to go back to 0.5.9 or 0.5.8 you need to flash your pedal back to 0.5.9.  Instructions are here:  https://medium.com/@jfrux/comma-pedal-updating-the-firmware-over-can-fa438a3cf910.  Also. After you flash your Pedal..  All hell will break loose on your dash.  Traction control error, Power Steering Error, Trailer Error, OMFG the sky is falling error etc.  DON'T PANIC.  Just drive around a bit and it will disappear after about 2-3 restarts of the car.  Don't rush it I believe it's time dependent as well.  Just drive as normal.  They'll go away.
</b>

<b>IMPORTANT:</b> I have added the stopping of services for Nidec vehicles, which always have power from the panda.  See feature list below for details.  
  

This is a fork of comma's openpilot, and contains tweaks for Hondas and GM vehicles 

<b>WARNING:</b>  Do NOT depend on OP to stop the car in time if you are approaching an object which is not in motion in the same direction as your car.  The radar will NOT detect the stationary object in time to slow your car enough to stop.  If you are approaching a stopped vehicle you must disengage and brake as radars ignore objects that are not in motion.

<b>NOTICE:</b>  Due to feedback I have turned on OTA updates.  You will receive updates automatically (after rebooting 2X) on your Eon so you don't have to reclone or git pull any longer to receive new features *MADE BETWEEN COMMA RELEASES*.  The reason why I separate the branches by release is because some releases can sometimes cause issues.  Features that I or others add, will continue to be updated when you are on the most current release.  If you DO NOT want OTA updates then create a file called "/data/no_ota_updates" and it will not perform OTA updates as long as that file exists.  


I will attempt to detail the changes in each of the branches here:


<b>kegman</b> - this is the default branch which does not include Gernby's resonant feed forward steering (i.e. it's comma's default steering)

<b>kegman-plusGernbySteering</b> - this branch has everything in the kegman branch PLUS Gernby's latest resonant mpc interp steering.  NEW! Now includes a primitive tuning script for your cell phone (or laptop) for live tuning (see feature section below for details)

<b>kegman-plusPilotAwesomeness-0.5.8</b> - <u>Older version of Gernbys steering branch.  Will not be updated.
  
<b>kegman-plusClarity</b> - branch specifically for the Honda Clarity (does not contain Gernby steering)

<b>kegman-plusAccordHybrid</b> - branch broken out for Accord Hybrid due to Insight fingerprint conflict

<b>kegman-plusGernbySteering-plusAccordHybrid</b> - same as plusAccordHybrid branch but with Gernby Steering added


List of changes and tweaks (latest changes at the top):
- <b>Toggle Comma's live tuning</b>:  Comma live tunes things like alignment, steerRatio etc.  But sometimes it doesn't converge to the right value and throws lane centering off during turns.  This allows you to use /data/openpilot/.tune.sh to toggle the auto-tune to off when the car feels right so that it doesn't tune the car any further than necessary.
  
- <b>Live tuner for Kp and Ki</b>:  Tune your Kp and Ki values live using your cell phone by SSHing into the Eon and executing cd /data/openpilot && ./tune.sh

- <b>Kill services if plugged in and Eon batt < kegman.json --> battPercOff</b>  Shutting down of the Eon never worked on Nidec vehicles because the Panda always supplies power.  When Eon senses power it just starts up again.  So I have mitigated the power drain by about 40% when it is discharging in the car.  Reminder that the Eon continues to charge cycling between battChargeMax and battChargeMin in the /data/kegman.json file.  If the car battery falls below carVoltageMinEonShutdown in the /data/kegman.json file WHILE CHARGING THE EON then charging is disabled.  As charging is disabled, the Eon battery will continue to drain until it reaches battPercOff (again in /data/kegman.json) at which point it will shut down services to conserve power.  This will not prevent the Eon from discharging completely but will cut the drainage by about 40%, buying you some more time before it goes dead.  If the Eon is in this "Power Saving" mode you will need to reboot the Eon by pressing on the power button and touching somewhere near the center of the screen (note that the screen will note work).  Also note that if you unplug the Eon and the battery is below battPercOff it will shutdown.  If you reboot the Eon while unplugged, it will give you 3 minutes until it shuts down again unless it is plugged in during this time.  If you unplugged the Eon and forgot to turn it off, it will shutdown when the battery falls below battPercOff.  Hopefully this mitigates the "dead Eon" syndrome that occurs when people have trouble powering their device back up after the battery is completely drained.
  
For Bosch vehicles, the Eon will just simply shutdown as usual when battery falls below battPercOff.  Killing of services only occurs for Nidecs.



- <b>Add @pjlao307's Dashcam Recording</b>:  Sometimes you just want to record a wicked OP run on a twisty highway to show your friends.  Sometimes you want to record a big flashing red error and complain about it and show your friends.  This does a screen video capture of the Eon screen and stores the files in /sdcard/videos on your Eon when the REC button is pressed.  Thanks to @pjlao307 and @theantihero for submitting the PR.

- <b>Stop logging when space hits 18% free space</b>:  Thanks to @emmertex for this easy fix to stop the Eon from filling up while driving when free space is low.

- <b>Added primitive tuning script</b>: To invoke live tuning:  (a) turn on tethering on your Eon,  (b) install JuiceSSH or similar and connect your cellphone to the wifi of the Eon using 192.168.43.1 and import the Comma private key,  (c) in JuiceSSH in the SSH session on the Eon issue cd /data/openpilot command, then ./tune.sh.  The text UI will be shown.  (d) turn "tuneGernby" to a "1"  (e) start driving and change the values to tune your steering.  It is best to have a cell phone mount in your car.  Note:  It takes 3 seconds for any changes to take effect.  

- <b>Replaced dev UI</b> with @perpetuoviator dev UI with brake light icon by @berno22 - Thank you both!  NOTE:  There are lots of conveniences in this UI.  When the car is on, you have to press the top left corner to get to the Settings screen.  If you tap the lower right corner you can see the tmux session.  The brake light icon doesn't work properly with some cars (needs a fingerprint tweak I believe.  The wifi IP address and upload speed is printed on the screen.  The brake icon is so that you can see if OP is causing the brake lights to turn on and off and pissing the guy or gal off behind you. NOTE:  For GM vehicles, the brake icon indicates use of the friction brakes on the vehicle instead of the brake lights themselves.
<b>UI touch controls:</b>
[Acess "EON Settings" by touching the top left hand corner (left side of the Speed Limit sign)]
[Acess "TmuxLog" by touching the bottom right hand corner]

- <b>Added moar JSON parameters</b>:  

"battPercOff": "25",  Turn off the Eon if the Eon battery percentage dips below this value - NOTE this only works when the Eon is NOT powered by the USB cable!

"brakeStoppingTarget": "0.25",  How much OP should mash the brakes when the car is stopped.  Increase if you live in hilly areas and need more standstill braking pressure.

"carVoltageMinEonShutdown": "11800", in mV.  Eon stops charging if car battery goes below this level - NOTE: this is the DISCHARGING voltage.  When the Eon is drawing current the voltage on the battery DROPS.  This is NOT the standing no-load voltage.  I would recommended that you unplug your Eon if you are away from your vehicle for more than a few hours and put a battery charger on your car's battery weekly to avoid wrecking your battery if your Eon stays powered when you shut off the car.

- <b>Tone down PID tuning for Pilot and Ridgline for 0.5.9</b>:  Comma changed latcontrol for 0.5.9, so I had to tone down the PID tuning, reducing steerKpV and steerKiV (to 0.45 and 0.135) because of a slow ping-pong on my 2018 Pilot.  Wheel shaking on 2017 Pilots with 0.5.9 have been reported and this change should help, but may not be sufficient for the 2017 model (and possibly 2016).  2016/7 owners may need to adjust steerKpV and steerKiV manually back to 0.38 and 0.11 in /data/openpilot/selfdrive/car/honda/interface.py to reduce the shake.

- <b>Persist some configuration data in JSON file (/data/kegman.json)</b>:  Sometimes you just want to make a tweak and persist some data that doesn't get wiped out the next time OP is updated.  Stuff like:


"battChargeMax": "70",  (Max limit % to stop charging Eon battery)

"battChargeMin": "60",  (Min limit % to start charging Eon battery)

"cameraOffset": "0.06", (CAMERA_OFFSET - distance from the center of car to Eon camera)

"lastTrMode": "2",      (last distance interval bars you used - (auto generated - do not touch this)

"wheelTouchSeconds": "180"  (time interval between wheel touches when driver facial monitoring is not on - MAX LIMIT 600 seconds)


^^^ This file is auto generated here:  <b>/data/kegman.json</b> so it will remain even when you do a fresh clone.  If you mess something up, just delete the file and it will auto generate to default values.  Use vim or nano to edit this file to your heart's content.

- <b>Interpolated (smoothed) the discontinuity of longitudinal braking profiles</b>:  Prior to this enhancement the braking profiles changed very abruptly like a step function, leading to excessive ping-ponging and late braking.  This feature reduces the ping-ponging and varies the braking strength linearly with gap closure speed (the faster the gap closes between you and the lead car, the harder the braking).

- <b>Remember last distance bar interval</b>:  On startup, the car will bring up the last distance interval used before the car was turned off.  For example:  If you were at X bars before you stopped the car or shut the Eon down, the next time you start the car, the distance setting will be X bars.  

- <b>OTA Updates turned on</b>:  Previously I had turned off OTA updates for safety reasons - I didn't want anyone to get an unexpected result when I made changes.  It appears that many more users want OTA updates for convenience so I have turned this feature back on.  IMPORTANT: If you DO NOT want OTA updates then create a file called "/data/no_ota_updates" and it will not perform OTA updates as long as that file exists.

- <b>Increase acceleration profile when lead car pulls away too quickly or no lead car</b>:  OP has two acceleration profiles, one occurs when following a lead car, and one without a lead car.  Oddly the acceleration profile when following is greater than when not following.  So sometimes a lead car will pull away so quickly, that the car goes from following to not following mode and the acceleration profile actually drops.  I've made the acceleration profiles the same so that the the car doesn't stop accelerating at the same rate when the lead car rips away quickly from a stop. 

- <b>FOUR (new) Step adjustable follow distance</b>:  The default behaviour for following distance is 1.8s of following distance.  It is not adjustable.  This typically causes, in some traffic conditions, the user to be constantly cut off by other drivers, and 1.8s of follow distance instantly becomes much shorter (like 0.2-0.5s).  I wanted to reintroduce honda 'stock-like' ACC behaviour back into the mix to prevent people from getting cutoff so often.  Here is a summary of follow distance in seconds:  <b>1 bar = 0.9s, 2 bars = 1.3s, 3 bars = 1.8, 4 bars = 2.5s of follow distance</b>. Thanks to @arne182, whose code I built upon.

- <b>Reduce speed dependent lane width to 2.85 to 3.5 (from 3.0 to 3.7) [meters]</b>:  This has the effect of making the car veer less towards a disappearing lane line because it assumes that the lane width is less.  It may also improve curb performance.

- <b>Display km/h for set speed in ACC HUD</b>:  For Nidec Hondas, Openpilot overrides Honda's global metric settings and displays mph no matter what.  This change makes the ACC HUD show km/h or mph and abides by the metric setting on the Eon.  I plan on upstreaming this change to comma in the near future.

- <b>Kill the video uploader when the car is running</b>:  Some people like to tether the Eon to a wifi hotspot on their cellphone instead of purchasing a dedicated SIM card to run on the Eon.  When this occurs default comma code will upload large video files even while you are driving chewing up your monthly data limits.  This change stops the video from uploading when the car is running.  *caution* when you stop the car, the videos will resume uploading on your cellular hotspot if you forget to disconnect it.

- <b>Increase brightness of Eon screen</b>:  After the NEOS 8 upgrade some have reported that the screen is too dim.  I have boosted the screen brightness to compensate for this.

- <b>Battery limit charging</b>:  The default comma code charges the Eon to 100% and keeps it there.  LiIon batteries such as the one in the Eon do not like being at 100% or low states of charge for extended periods (this is why when you first get something with a LiIon battery it is always near 50% - it is also why Tesla owners don't charge their cars to 100% if they can help it).  By keeping the charge between 60-70% this will prolong the life of the battery in your Eon.  *NOTE* after your battery gets to 70% the LED will turn from yellow to RED and stay there.  Rest assured that while plugged in the battery will stay between 60-70%.  You can (and should) verify this by plugging the Eon in, SSHing into the Eon and performing a 'tmux a' command to monitor what the charging does.  When you disconnect your Eon, be sure to shut it down properly to keep it in the happy zone of 60-70%.  You can also look at the battery icon to ensure the battery is approximately 60-70% by touching near the left of the eon screen.  Thanks to @csouers for the initial iteration of this.

- <b>Tuned braking at city street speeds (Nidecs only)</b>:  Some have described the default braking when slowing to a stop can be very 'late'.  I have introduced a change in MPC settings to slow the car down sooner when the radar detects deceleration in the lead car.  Different profiles are used for 1 bar and 2 bar distances, with a more aggressive braking profile applied to 1 bar distance.  Additionally lead car stopped distance is increased so that you stop a little farther away from the car in front for a greater margin of error.  Thanks to @arne182 for the MPC and TR changes which I built upon.

- <b>Fixed grinding sound when braking with Pedal (Pilots only)</b>:  Honda Pilots with pedals installed may have noticed a loud ripping / grinding noise accompanied by oscillating pressure on the brake when the brake is pressed especially at lower speeds.  This occurs because OP disengages too late when the brake is pressed and the user ends up fighting with OP for the right brake position.  This fix detects brake pressure sooner so that OP disengages sooner so that the condition is significantly reduced.  If you are on another model and this is happening this fix may also work for you so please message me on Slack or Discord @kegman.

- <b>Smoother acceration from stop (Pedal users)</b>:  The default acceleration / gas profile when pedal is installed may cause a head snapping "lurch" from a stop which can be quite jarring.  This fix smoothes out the acceleration when coming out of a stop.

- <b>Dev UI</b>:  Thanks to @zeeexaris who made this work post 0.5.7 - displays widgets with steering information and temperature as well as lead car velocity and distance.  Very useful when entering turns to know how tight the turn is and more certainty as to whether you have to intervene.  Also great when PID tuning.

- <b>Gernby's Resonant Feed Forward Steering</b>:  This is still a work in progress.  Some cars respond very well while there is more variance with other cars.  You may need to tweak some parameters to make it work well but once it's dialed in it makes the wheel very stiff and more impervious to wind / bumps and in some cases makes car centering better (such as on the PilotAwesomeness branch).  Give it a try and let @gernby know what you find.  Gernby's steering is available on kegman-plusGernbySteering, kegman-plusPilotAwesomeness.  

- <b>Steering off when blinkers on</b>:  The default behaviour when changing lanes is the user overrides the wheel, a bunch of steering required alarms sound and the user lets go of the wheel.  I didn't like fighting the wheel so when the blinkers are on I've disabled the OP steering.  Note that the blinker stock must be fully left or right or held in position for the steering to be off.  The "3 blink" tap of the stock does not deactivate steering for long enough to be noticeable.

- <b>LKAS button toggles steering</b>:  Stock Openpilot deactivates the LKAS button.  In some cases while driving you may have to fight the wheel for a long period of time.  By pressing the LKAS button you can toggle steering off or on so that you don't have to fight the wheel, which can get tiring and probably isn't good for the EPS motor.  When LKAS is toggled off OP still controls gas and brake so it's more like standard ACC.

- <b>Honda Pilot and Ridgeline PID</b>:  I wasn't happy with the way Honda Pilot performed on curves where the car often would hug the inside line of the turn and this was very hazardous in 2 lane highways where it got very close to the oncoming traffic.  Also, on crowned roads (where the fast lane slants to the left and where the slow lane slants to the right), the car would not overcome the gravity of the slanted road and "hug" in the direction of the slant.  After many hours of on the road testing, I have mitigated this issue.  When combined with Gernby's steering it is quite a robust setup.  This combination is found in kegman-plusPilotAwesomeness.  Apparently this branch works well with RIDGELINES too!


Enjoy everyone.
