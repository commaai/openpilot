This is a fork of comma's openpilot, and contains tweaks for (mostly) Hondas

I will attempt to detail the changes in each of the branches here:

<b>kegman</b> - this is the default branch which does not include Gernby's resonant feed forward steering (i.e. it's comma's default steering)

<b>kegman-plusGernbySteering</b> - this branch is everything in the kegman branch PLUS a previous version of Gernby's feed forward steering which worked reasonably well

<b>kegman-plusPilotAwesomeness</b> - <u>If you have a Honda Pilot, use this branch.</u>  It has everything in kegman branch, uses my PID tuning + a magical older version of Gernby's FF steering which just happened to work very well across all driving conditions including slanted (crowned roads), wind gusts, road bumps, centering on curves, and keeping proper distance from curbs.  I have yet to test a combination of FF steering and PID tuning that can beat the performance of this for Honda Pilots.

<b>testing-GernbyPRcandidate</b> - this is kegman branch + Gernby's latest resonant feed forward steering which Gernby is planning to submit to comma as a pull request to have it included as part of base code.


The fork has been confirmed to work for Bosch and Nidec based Hondas. 

List of changes and tweaks (latest changes at the top:
- Display km/h for set speed in ACC HUD:  For Nidec Hondas, Openpilot overrides Honda's global metric settings and displays mph no matter what.  This change makes the ACC HUD show km/h or mph and abides by the metric setting on the Eon.  I plan on upstreaming this change to comma in the near future.

- Kill the video uploader when the car is running:  Some people like to tether the Eon to a wifi hotspot on their cellphone instead of purchasing a dedicated SIM card to run on the Eon.  When this occurs default comma code will upload large video files even while you are driving chewing up your monthly data limits.  This change stops the video from uploading when the car is running.  *caution* when you stop the car, the videos will resume uploading on your cellular hotspot if you forget to disconnect it.

- Increase brightness of Eon screen:  After the NEOS 8 upgrade some have reported that the screen is too dim.  I have boosted the screen brightness to compensate for this.

- Battery limit charging:  The default comma code charges the Eon to 100% and keeps it there.  LiIon batteries such as the one in the Eon do not like being at 100% or low states of charge for extended periods (this is why when you first get something with a LiIon battery it is always near 50% - it is also why Tesla owners don't charge their cars to 100% if they can help it).  By keeping the charge between 60-70% this will prolong the life of the battery in your Eon.  *NOTE* after your battery gets to 70% the LED will turn from yellow to RED and stay there.  Rest assured that while plugged in the battery will stay between 60-70%.  You can (and should) verify this by plugging the Eon in, SSHing into the Eon and performing a 'tmux a' command to monitor what the charging does.  When you disconnect your Eon, be sure to shut it down properly to keep it in the happy zone of 60-70%.  You can also look at the battery icon to ensure the battery is approximately 60-70% by touching near the left of the eon screen.

more to come....


