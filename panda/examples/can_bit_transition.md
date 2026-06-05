# How to use can_bit_transition.py to reverse engineer a single bit field

Let's say our goal is to find a brake pedal signal (caused by your foot pressing the brake pedal vs adaptive cruise control braking).

The following process will allow you to quickly find bits that are always 0 during a period of time (when you know you were not pressing the brake with your foot) and always 1 in a different period of time (when you know you were pressing the brake with your foot).

Open up a drive in cabana where you can find a place you used the brake pedal and another place where you did not use the brake pedal (and you can identify when you were on the brake pedal and when you were not).  You may want to go out for a drive and put something in front of the camera after you put your foot on the brake and take it away before you take your foot off the brake so you can easily identify exactly when you had your foot on the brake based on the video in cabana.  This is critical because this script needs the brake signal to always be high the entire time for one of the time frames.  A 10 second time frame worked well for me.

I found a drive where I knew I was not pressing the brake between timestamp 50.0 thru 65.0 and I was pressing the brake between timestamp 69.0 thru 79.0. Determine what the timestamps are in cabana by plotting any message and putting your mouse over the plot at the location you want to discover the timestamp.  The tool tip on mouse hover has the format: timestamp: value

Now download the log from cabana (Save Log button) and run the script passing in the timestamps
(replace csv file name with cabana log you downloaded and time ranges with your own)
```
./can_bit_transition.py ./honda_crv_ex_2017_can-1520354796875.csv 50.0-65.0 69.0-79.0
```

The script will output bits that were always low in the first time range and always high in the second time range (and vice versa)
```
id 17c 0 -> 1 at byte 4 bitmask 1
id 17c 0 -> 1 at byte 6 bitmask 32
id 221 1 -> 0 at byte 0 bitmask 4
id 1be 0 -> 1 at byte 0 bitmask 16
```

Now I go back to cabana and graph the above bits by searching for the message by id, double clicking on the appropriate bit, and then selecting "show plot". I already knew that message id 0x17c is both user brake and adaptive cruise control braking combined, so I plotted one of those signals along side 0x221 and 0x1be.  By replaying a drive I could see that 0x221 was not a brake signal (when high at random times that did not correspond to braking).  Next I looked at 0x1be and I found it was the brake pedal signal I was looking for (went high whenever I pressed the brake pedal and did not go high when adaptive cruise control was braking).