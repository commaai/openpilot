This is a fork of comma's openpilot, and contains tweaks for (mostly) Hondas

I will attempt to detail the changes here:

Branch Detail:
<b>kegman</b> - this is the default branch which does not include Gernby's resonant feed forward steering (i.e. it's comma's default steering)

<b>kegman-plusGernbySteering</b> - this branch is everything in the kegman branch PLUS a previous version of Gernby's feed forward steering which worked reasonably well

<b>kegman-plusPilotAwesomeness</b> - <u>If you have a Honda Pilot, use this branch.</u>  It has everything in kegman branch, uses my PID tuning + a magical older version of Gernby's FF steering which just happened to work very well across all driving conditions including slanted (crowned roads), wind gusts, road bumps, centering on curves, and keeping proper distance from curbs.  I have yet to test a combination of FF steering and PID tuning that can beat the performance of this for Honda Pilots.

<b>testing-GernbyPRcandidate</b> - this is kegman branch + Gernby's latest resonant feed forward steering which Gernby is planning to submit to comma as a pull request to have it included as part of base code.
