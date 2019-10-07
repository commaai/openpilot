<td><a href="https://www.youtube.com/watch?v=emKcwWaTjyg" title="Youtube" rel="noopener"><img src="https://i.imgur.com/eEX1qmB.png"></a></td>

Welcome to openpilot 0.6.4 DEVEL with OLD_CAR support and ZORRO CurvatureFactorLearner
======

[openpilot](http://github.com/commaai/openpilot) is an open source driving agent. Currently, it performs the functions of Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS).  It's about on par with Tesla Autopilot and GM Super Cruise, and better than [all other manufacturers](http://www.thedrive.com/tech/5707/the-war-for-autonomous-driving-part-iii-us-vs-germany-vs-japan).

This OLD_CAR Branch brings openpilot to almost every car. Follow this readme to get an overview how it works.

Big thank you goes to @wocsor. He developed the whole thing and modified the code.

The openpilot codebase has been written to be concise and to enable rapid prototyping. We look forward to your contributions - improving real vehicle automation has never been easier.

OVERVIEW
=======================

To make openpilot work in an old car, we need to retrofit actuators from supported cars like toyota corolla 2018. Some small ECU needs to be build DIY.

Brain:
* [EON and Pada](#eon-and-panda)


Steering:
* [EPS - electric power steering](#eps)
* [VSS - vehicle speed sensor](#vss)
* [Buttons](#buttons)
* [Cruise_ECU](#cruise_ecu)
* [Steering angle sensor](#steering-angle-sensor)

Throttle:
* [Cruise Control Actuator](#cruise-control-actuator)
* [Potentiometer](#potentiometer)
* [Throttle_ECU](#throttle_ecu)

Radar: 
* [Radar sensor](#radar)

Brake (not finished yet):
* [ABS PUMP / OSCC Module](#POLYSYNC-OSCC-Brake-module-and-Prius-Actuator)

Community
 * [Community](#commutity)

# BRAIN

## EON and Panda

![enter image description here](https://i.imgur.com/RBqQvoZ.jpg)

First, we need a brain to control everything. 
This is EON. An extremely powerful piece of hardware which runs openpilot. 
We also need Panda, which connects EON to the OBD2 port of the car. So that EON can talk to the car via CAN-Bus. 
Oh wait! - we do not have a CAN-Bus network in our car. Don't worry we will build it DIY.
For more informations to EON or PANDA visit [comma.ai.](https://comma.ai)

I have used a cheap [OBD2 wire connector](https://www.amazon.com/iKKEGOL-Connector-Diagnostic-Extension-Pigtail/dp/B07F16HC12/ref=sr_1_15?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&keywords=obd2%20cable&qid=1560506720&s=gateway&sr=8-15) to pinout the panda. 


# STEERING


## Eps

I have used an EPS (electronic power steering) out of a Toyota corolla 2018.
This is already supported by Openpilot so we do not have to port it from sketch.

![electric steering column out of a corolla](https://i.imgur.com/PUOQNph.png)

It is  important, that it provides LKAS (lane keep assistent). The steering column and motor might be the same like in older corollas. But the ECU is different. So make sure to buy the ECU with LKAS ( "KV" on the sticker).
![EPS ECU COROLLA 2018 WITH LKAS](https://i.imgur.com/Bl3FpBX.png)

This is how to wire the steering ECU:

![enter image description here](https://i.imgur.com/w6tnlDq.png)

z11 and z7 connectors will be connected to the EPS Motor.

Now it's time to retrofit the steering column. Since every car is slightly different, you need to be a little creative. 
Im my case, I have cut my stock column in half and welded both ends to the corolla steering column.
If you already have a hydraulic power steering, you might want to disable that. Otherwise you would have a power steering on top of a power steering, and your steering wheel will never return to center by itself.

This is how my conversion looks like: 

![enter image description here](https://i.imgur.com/TTxdILC.jpg)

![enter image description here](https://i.imgur.com/349kMvt.png)

Now we have a working power steering in our car! 
Unfortunately it goes into failsafe, which means that it will disable LKAS. 
Cruise_ECU will take care of this issue.

----
## Vss

Eon needs to know how fast we are driving. Therefore we need to add a sensor which measure the "speed" of the car. Most cars already provide such a signal already. For example for the radio. If you have such a signal, you can grab that. In my case I have added a hall sensor to the rotary disc of the speedometer. This counts 4000 signal each km. Cruis_Ecu will calculate that signal with some math. NOTE: you need to adjust the "counts per km" of your specific sensor in Cruise_ECU code.


----
## Buttons

Since we do not have original toyota buttons, - guess what - we need to build it ourself.
Be creative, it is simple task. Pull-down buttons, which will be connected to Cruise_ECU.

![enter image description here](https://i.imgur.com/V3gqlWY.png)

![enter image description here](https://i.imgur.com/LdcZqPN.jpg)

----
## Cruise_Ecu

Cruise_ECU Hardware is an [Arduino Uno](https://www.amazon.com/Elegoo-EL-CB-001-ATmega328P-ATMEGA16U2-Arduino/dp/B01EWOE0UU/ref=sr_1_2?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&keywords=arduino%20uno&qid=1560514638&s=gateway&sr=8-2) with a [CAN bus shield](https://www.amazon.com/MakerFocus-CAN-Bus-Shield-V1-2/dp/B06XWQ4WF9/ref=sr_1_2?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&keywords=arduino%20uno%20canbus%20shield&qid=1560514663&s=gateway&sr=8-2) attached to it.
![CRUISE_ECU](https://i.imgur.com/CnysIXP.png)


It handles the following functions: 

 1. Cruise_ECU sends some CAN messages on the bus, which let the EPS thinks that it is still in a corolla.
 The EPS is happy and does not go into failsafe. LKAS is ready to take control of it !!! 
 
 2. Cruise_ECU calculates the current speed by reading the VSS. It sends a 0xb4 and 0xaa message on the CAN-Bus. EON and Throttle_ECU can read and use those messages.
 
 3. Cruise_ECU digitalReads the buttons and sends the associated CAN messages to the bus. It let's us enable and disable
 Openpilot. We can also increase or decrease the set speed.
 
 5. It provides some safety function. It will disable immediately, if it looses CAN safety checksum.
 
 This is how you wire Cruise_ECU: 
![CRUISE_ECU_PINOUT](https://i.imgur.com/9Mnr5qg.jpg)

Note: The LED stuff is optional. "Interrupt to Throttle_ecu" is an extra safety feature, not necessary but recommended.

Attach a switch to the brake pedal and wire it to D7 (button_cancel). Openpilot will disengage when pressing the brake pedal.
If you have a manual transmition, do the same for clutch pedal.

[Download Cruise ECU Code.](https://github.com/Lukilink/Cruise_ECU)


----
## Steering Angle Sensor

I am using the stock steering angle sensor out of a toyota corolla / rav4.
I highly recommend buying it with the hair spring attached. Also we do not need the hair spring, it takes care of the sensor while shipping.
Fortunatley the sensor provides it's own ECU. Therefor it is like plug and play. 

![enter image description here](https://i.imgur.com/8hsyrax.png)


![enter image description here](https://i.imgur.com/CwXuUUv.jpg)
	
The stock Toyota sensor is laggy and not very precise. 
Zorrobyte has started to build a high precision, fast like hell, sensor.
It´s cheap and provides a stunning performance. Zorrobyte invented a super clever mounting position. 
Check out his [Github](https://github.com/zorrobyte/betterToyotaAngleSensorForOP). 
I will definetely upgrade to zorro_angle_sensor.

---

# THROTTLE

## Cruise Control Actuator

Add an electric cruise control actuator to your throttle.
Choose what ever brand you like. They are all very similar and it is easy to get one cheap out of a 90th car on ebay. 
It needs to have an electric motor, and something similar to a clutch.
The "clutch" is basically a solenoid, which disconnects everything mechanically. Pretty nice safety feature :) 
If you already have stock cruise control in your car, take that one!


## Potentiometer

To measure the position of the throttle we use the stock potentiometer. Almost every throttle has a potentiometer attached.
We read that amount in Throttle_ECU. 

Note: Throttle_ECU sketch must be adjusted for your specific potentiometer. Therefore you need to "measure" the min and max value of your potentiometer with analog_read_to_serial. If you have your min and max values you can set those in Throttle_ECU sketch.

## Throttle_ECU

Throttle ECU hardware is an [Arduino Uno](https://www.amazon.com/Elegoo-EL-CB-001-ATmega328P-ATMEGA16U2-Arduino/dp/B01EWOE0UU/ref=sr_1_3?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&keywords=arduino%20uno&qid=1560516407&s=gateway&sr=8-3) with a [CAN-bus shield](https://www.amazon.com/MakerFocus-CAN-Bus-Shield-V1-2/dp/B06XWQ4WF9/ref=sr_1_2?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&keywords=arduino%20uno%20canbus%20shield&qid=1560514663&s=gateway&sr=8-2) and a [H-bridge](https://www.amazon.com/CJRSLRB-3Packs-Controller-H-Bridge-Arduino/dp/B07BMTQMKN/ref=sr_1_2_sspa?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&keywords=H-bridge&qid=1560516427&s=gateway&sr=8-2-spons&psc=1) attached.

![enter image description here](https://i.imgur.com/EClutor.png)

It handles the following functions: 

 1. Throttle_ECU receives the angle_requests from EON and runs the cruise actuator motor and solenoid.
 
 2. It calculates the acceleration amount depending on the current speed. That means it accelerates faster when driving fast, and more smooth and carefully when driving slow. 
 
 4. It provides some safety function. It will disengage immediately, if it looses CAN-safety messages. 

Download [Throttle_ECU Code](https://github.com/Lukilink/Cruise_ECU).

![Throttle_ECU](https://i.imgur.com/FgJhgx8.png)

---
# RADAR

## Radar


![enter image description here](https://i.imgur.com/0dD9zPy.png)

Similar to the steering angle sensor, the radar out of a corolla / rav4 provides its own ECU. 
Therefore it is pretty easy to install. 

![enter image description here](https://i.imgur.com/soMhXAJ.png)

You may need to fingerprint after you have installed the radar. 

---

# BRAKE

Brake is not finished yet. Therefore I will not go to much in detail. 

## POLYSYNC OSCC Brake module and Prius Actuator

![enter image description here](https://i.imgur.com/fcA75LR.png)

[Polysync /oscc](https://github.com/PolySync/oscc)  build a board with an Arduino mega attached to control a toyota prius ABS actuator. 

This actuator will be placed on top of the stock brake system.

To do:  merge / port the OSCC DBC to openpilot

[ossc.dbc](https://github.com/PolySync/oscc/blob/master/api/include/can_protocols/oscc.dbc)

I would appreciate if someone could help us with that.
If you want to know more about that. Feel free to contact us on Github.

---

# COMMUNITY

Community is the most important thing on this project.

The [legendary Arne Fork] does support old_cars now.
Big thanks to Arne182 for his awesome work. 


Comma [Twitter you should follow](https://twitter.com/comma_ai).

Also, we have a several thousand people community on [Discord](https://discord.comma.ai).



Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>
