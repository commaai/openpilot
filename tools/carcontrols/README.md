Debug car controls
-------------

**Hardware needed**: [panda](panda.comma.ai), [giraffe](https://comma.ai/shop/products/giraffe/), joystick

Use the panda's OBD-II port to connect with your car and a usb cable to connect the panda to your pc.
Also, connect a joystick to your pc.

`joystickd.py` runs a deamon that reads inputs from a joystick and publishes them over zmq.
`boardd` sends the CAN messages from your pc to the panda.
`debug_controls` is a mocked version of `controlsd.py` and uses input from a joystick to send controls to your car.

Make sure the conditions are met in the panda to allow controls (e.g. cruise control engaged). You can also make a modification to the panda code to always allow controls.

Usage:
```
python carcontrols/joystickd.py

# In another terminal:
BASEDIR=$(pwd) selfdrive/boardd/boardd

# In another terminal:
python carcontrols/debug_controls.py

```
![Imgur](steer.gif)
