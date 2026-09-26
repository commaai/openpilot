# How to use can_unique.py to reverse engineer a single bit field

Let's say our goal is to find the CAN message indicating that the driver's door is either open or closed.
The following process is great for simple single-bit messages.
However for frequently changing values, such as RPM or speed, Cabana's graphical plots are probably better to use.


First record a few minutes of background CAN messages with all the doors closed and save it in background.csv:
```
./can_logger.py
mv output.csv background.csv
```
Then run can_logger.py for a few seconds while performing the action you're interested, such as opening and then closing the
front-left door and save it as door-fl-1.csv
Repeat the process and save it as door-f1-2.csv to have an easy way to confirm any suspicions.

Now we'll use can_unique.py to look for unique bits:
```
$ ./can_unique.py door-fl-1.csv background*
id 820 new one  at byte 2 bitmask 2
id 520 new one  at byte 3 bitmask 7
id 520 new zero at byte 3 bitmask 8
id 520 new one  at byte 5 bitmask 6
id 520 new zero at byte 5 bitmask 9
id 559 new zero at byte 6 bitmask 4
id 804 new one  at byte 5 bitmask 2
id 804 new zero at byte 5 bitmask 1

$ ./can_unique.py door-fl-2.csv background*
id 672 new one  at byte 3 bitmask 3
id 820 new one  at byte 2 bitmask 2
id 520 new one  at byte 3 bitmask 7
id 520 new zero at byte 3 bitmask 8
id 520 new one  at byte 5 bitmask 6
id 520 new zero at byte 5 bitmask 9
id 559 new zero at byte 6 bitmask 4
```

One of these bits hopefully indicates that the driver's door is open.
Let's go through each message ID to figure out which one is correct.
We expect any correct bits to have changed in both runs.
We can rule out 804 because it only occurred in the first run.
We can rule out 672 because it only occurred in the second run.
That leaves us with these message IDs: 820, 520, 559. Let's take a closer look at each one.

```
$ fgrep ,559, door-fl-1.csv |head
0,559,00ff0000000024f0
0,559,00ff000000004464
0,559,00ff0000000054a9
0,559,00ff0000000064e3
0,559,00ff00000000742e
0,559,00ff000000008451
0,559,00ff00000000949c
0,559,00ff00000000a4d6
0,559,00ff00000000b41b
0,559,00ff00000000c442
```
Message ID 559 looks like an incrementing value, so it's not what we're looking for.

```
$ fgrep ,520, door-fl-2.csv
0,520,26ff00f8a1890000
0,520,26ff00f8a2890000
0,520,26ff00f8a2890000
0,520,26ff00f8a1890000
0,520,26ff00f8a2890000
0,520,26ff00f8a1890000
0,520,26ff00f8a2890000
0,520,26ff00f8a1890000
0,520,26ff00f8a2890000
0,520,26ff00f8a1890000
0,520,26ff00f8a2890000
0,520,26ff00f8a1890000
```
Message ID 520 oscillates between two values. However I only opened and closed the door once, so this is probably not it.

```
$ fgrep ,820, door-fl-1.csv
0,820,44000100a500c802
0,820,44000100a500c803
0,820,44000300a500c803
0,820,44000300a500c802
0,820,44000300a500c802
0,820,44000300a500c802
0,820,44000100a500c802
0,820,44000100a500c802
0,820,44000100a500c802
```
Message ID 820 looks promising! It starts off at 44000100a500c802 when the door is closed.
When the door is open it goes to 44000300a500c802.
Then when the door is closed again, it goes back to 44000100a500c802.
Let's confirm by looking at the data from our other run:
```
$ fgrep ,820, door-fl-2.csv
0,820,44000100a500c802
0,820,44000300a500c802
0,820,44000100a500c802
```
Perfect! We now know that message id 820 at byte 2 bitmask 2 is set if the driver's door is open.
If we repeat the process with the front passenger's door,
then we'll find that message id 820 at byte 2 bitmask 4 is set if the front-right door is open.
This confirms our finding because it's common for similar signals to be near each other.
