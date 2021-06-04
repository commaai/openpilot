# Joystick

**Hardware needed**: [comma two devkit](https://comma.ai/shop/products/comma-two-devkit), laptop, joystick (optional)

With joystickd, you can connect your laptop to your comma two over the network and debug controls using one a joystick or your keyboard, however a joystick is recommended for more precise control.

Using a keyboard:
---

To get started you'll want to first ssh into your comma two devkit. On your comma two, start joystickd with the following command: `tools/joystick/joystickd.py --keyboard`. The available buttons and axes will print showing their key mappings.

In general, the WASD keys control gas and brakes and steering torque in 5% increments.

Using a joystick:
---

In order to use a joystick over the network, we need to run joystickd locally from your laptop and have it send `testJoystick` ZMQ packets over the network to the comma two. First connect a compatible joystick to your PC; joystickd uses [inputs](https://pypi.org/project/inputs) which supports many common gamepads and joysticks.

1. Connect a joystick to your laptop, tell cereal to publish using ZMQ, and start joystickd:
   ```shell
   export ZMQ=1
   tools/joystick/joystickd.py
   ```
2. Start a WiFi hotspot on your comma two, connect your laptop to it, and open a new ssh shell.
3. On your comma two, run unbridge with your laptop's IP address. This republishes `testJoystick` sent from your laptop so that openpilot can receive the messages:
   ```shell
   cereal/messaging/bridge --reverse -ip {LAPTOP_IP}
   ```
4. Finally, since we aren't running joystickd on the comma two, we need to write a parameter to let controlsd know to start in debug mode:
   ```shell
   echo -n "1" > /data/params/d/JoystickDebugMode
   ```

---
Now start your car and openpilot should go into debug mode with an alert!

- `joystickd.py` runs a deamon that reads inputs from a keyboard or joystick and publishes them over zmq or msgq.
- openpilot's [`controlsd`](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/controlsd.py) reads a parameter that joystickd sets on startup and switches into a debug mode, receiving steering and acceleration inputs from the joystick instead of from the standard controllers.

Make sure the conditions are met in the panda to allow controls (e.g. cruise control engaged). You can also make a modification to the panda code to always allow controls.

![Imgur](steer.gif)
