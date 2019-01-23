This is a fork of comma's openpilot, and contains tweaks for Hondas and some GM vehicles 

I will attempt to detail the changes in each of the branches here:

<b>kegman</b> - this is the default branch which does not include Gernby's resonant feed forward steering (i.e. it's comma's default steering)

<b>kegman-plusGernbySteering</b> - this branch is everything in the kegman branch PLUS a Gernby's LATEST feed forward steering.  This also includes working code for GM cars.  (thx to @jamezz for the code and @cryptokylan for submitting the GM stuff!)

<b>kegman-plusPilotAwesomeness</b> - <u>If you have a Honda Pilot, OR Honda Ridgeline use this branch.</u>  It has everything in kegman branch, uses my PID tuning + a magical older version of Gernby's FF steering which just happened to work very well across all driving conditions including slanted (crowned roads), wind gusts, road bumps, centering on curves, and keeping proper distance from curbs.  I have yet to test a combination of FF steering and PID tuning that can beat the performance of this for Honda Pilots.

<b>kegman-plusPilotAwesomeness2</b> - <u>If you have a Honda Pilot, OR Honda Ridgeline you can use this branch as well.</u>  It uses the latest Gernby Resonant steering and I spent a few hours using his OpenPilot Dashboard and staged resonant rolling branch to figure out the optimal values for resistive, inductive and reactive parameters.  You should notice a stiffer steering wheel, which in theory should let to even better response.  On slower curves when you give the steering a nudge it may "stick" and maintain the steering angle.  Let me know how it goes and which Honda Pilot branch you prefer.

<b>testing-GernbyPRcandidate</b> - this has been MERGED with kegman-plusGernbySteering


List of changes and tweaks (latest changes at the top:
- <b>Display km/h for set speed in ACC HUD</b>:  For Nidec Hondas, Openpilot overrides Honda's global metric settings and displays mph no matter what.  This change makes the ACC HUD show km/h or mph and abides by the metric setting on the Eon.  I plan on upstreaming this change to comma in the near future.

- <b>Kill the video uploader when the car is running</b>:  Some people like to tether the Eon to a wifi hotspot on their cellphone instead of purchasing a dedicated SIM card to run on the Eon.  When this occurs default comma code will upload large video files even while you are driving chewing up your monthly data limits.  This change stops the video from uploading when the car is running.  *caution* when you stop the car, the videos will resume uploading on your cellular hotspot if you forget to disconnect it.

- <b>Increase brightness of Eon screen</b>:  After the NEOS 8 upgrade some have reported that the screen is too dim.  I have boosted the screen brightness to compensate for this.

- <b>Battery limit charging</b>:  The default comma code charges the Eon to 100% and keeps it there.  LiIon batteries such as the one in the Eon do not like being at 100% or low states of charge for extended periods (this is why when you first get something with a LiIon battery it is always near 50% - it is also why Tesla owners don't charge their cars to 100% if they can help it).  By keeping the charge between 60-70% this will prolong the life of the battery in your Eon.  *NOTE* after your battery gets to 70% the LED will turn from yellow to RED and stay there.  Rest assured that while plugged in the battery will stay between 60-70%.  You can (and should) verify this by plugging the Eon in, SSHing into the Eon and performing a 'tmux a' command to monitor what the charging does.  When you disconnect your Eon, be sure to shut it down properly to keep it in the happy zone of 60-70%.  You can also look at the battery icon to ensure the battery is approximately 60-70% by touching near the left of the eon screen.  Thanks to @csouers for the initial iteration of this.

- <b>Tuned braking at city street speeds (Nidecs only)</b>:  Some have described the default braking when slowing to a stop can be very 'late'.  I have introduced a change in MPC settings to slow the car down sooner when the radar detects deceleration in the lead car.  Different profiles are used for 1 bar and 2 bar distances, with a more aggressive braking profile applied to 1 bar distance.  Additionally lead car stopped distance is increased so that you stop a little farther away from the car in front for a greater margin of error.  Thanks to @arne182 for the MPC and TR changes which I built upon.

- <b>Fixed grinding sound when braking with Pedal (Pilots only)</b>:  Honda Pilots with pedals installed may have noticed a loud ripping / grinding noise accompanied by oscillating pressure on the brake when the brake is pressed especially at lower speeds.  This occurs because OP disengages too late when the brake is pressed and the user ends up fighting with OP for the right brake position.  This fix detects brake pressure sooner so that OP disengages sooner so that the condition is significantly reduced.  If you are on another model and this is happening this fix may also work for you so please message me on Slack or Discord @kegman.

- <b>Smoother acceration from stop (Pedal users)</b>:  The default acceleration / gas profile when pedal is installed may cause a head snapping "lurch" from a stop which can be quite jarring.  This fix smoothes out the acceleration when coming out of a stop.

- <b>Dev UI</b>:  Thanks to @zeeexaris who made this work post 0.5.7 - displays widgets with steering information and temperature as well as lead car velocity and distance.  Very useful when entering turns to know how tight the turn is and more certainty as to whether you have to intervene.  Also great when PID tuning.

- <b>Gernby's Resonant Feed Forward Steering</b>:  This is still a work in progress.  Some cars respond very well while there is more variance with other cars.  You may need to tweak some parameters to make it work well but once it's dialed in it makes the wheel very stiff and more impervious to wind / bumps and in some cases makes car centering better (such as on the PilotAwesomeness branch).  Give it a try and let @gernby know what you find.  Gernby's steering is available on kegman-plusGernbySteering, kegman-plusPilotAwesomeness.  

- <b>Steering off when blinkers on</b>:  The default behaviour when changing lanes is the user overrides the wheel, a bunch of steering required alarms sound and the user lets go of the wheel.  I didn't like fighting the wheel so when the blinkers are on I've disabled the OP steering.  Note that the blinker stock must be fully left or right or held in position for the steering to be off.  The "3 blink" tap of the stock does not deactive steering for long enough to be noticeable.

- <b>LKAS button toggles steering</b>:  Stock Openpilot deactivates the LKAS button.  In some cases while driving you may have to fight the wheel for a long period of time.  By pressing the LKAS button you can toggle steering off or on so that you don't have to fight the wheel, which can get tiring and probably isn't good for the EPS motor.  When LKAS is toggled off OP still controls gas and brake so it's more like standard ACC.

- <b>3 Step adjustable follow distance</b>:  The default behaviour for following distance is 1.8s of following distance.  It is not adjustable.  This typically causes, in some traffic conditions, the user to be constantly cut off by other drivers, and 1.8s of follow distance instantly becomes much shorter (like 0.2-0.5s).  I wanted to reintroduce honda 'stock-like' ACC behaviour back into the mix to prevent people from getting cutoff so often.  Here is a summary of follow distance in seconds:  1 bar = 1s, 2 bars = 2s, 3 bars = 3s of follow distance. Thanks to @arne182, whose code I built upon, distance adjustment is back.

- <b>Honda Pilot and Ridgeline PID</b>:  I wasn't happy with the way Honda Pilot performed on curves where the car often would hug the inside line of the turn and this was very hazardous in 2 lane highways where it got very close to the oncoming traffic.  Also, on crowned roads (where the fast lane slants to the left and where the slow lane slants to the right), the car would not overcome the gravity of the slanted road and "hug" in the direction of the slant.  After many hours of on the road testing, I have mitigated this issue.  When combined with Gernby's steering it is quite a robust setup.  This combination is found in kegman-plusPilotAwesomeness.  Apparently this branch works well with RIDGELINES too!


Enjoy everyone.
