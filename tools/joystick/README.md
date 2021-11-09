# Joystick

**Hardware needed**: [comma two devkit](https://comma.ai/shop/products/comma-two-devkit), laptop, joystick (optional)

With joystickd, you can connect your laptop to your comma device over the network and debug controls using a joystick or keyboard, however a joystick is recommended for more precise control.

Using a keyboard
---

To get started, ssh into your comma device and start joystickd with the following command:

```shell
tools/joystick/joystickd.py --keyboard
```

The available buttons and axes will print showing their key mappings. In general, the WASD keys control gas and brakes and steering torque in 5% increments.

Using a joystick
---

In order to use a joystick over the network, we need to run joystickd locally from your laptop and have it send `testJoystick` ZMQ packets over the network to the comma device. First connect a compatible joystick to your PC; joystickd uses [inputs](https://pypi.org/project/inputs) which supports many common gamepads and joysticks.

1. Connect your laptop to your comma device's hotspot and open a new ssh shell. Since joystickd is being run on your laptop, we need to write a parameter to let controlsd know to start in joystick debug mode:
   ```shell
   # on your comma device
   echo -n "1" > /data/params/d/JoystickDebugMode
   ```
2. Run bridge with your laptop's IP address. This republishes the `testJoystick` packets sent from your laptop so that openpilot can receive them:
   ```shell
   # on your comma device
   cereal/messaging/bridge {LAPTOP_IP} testJoystick
   ```
3. Finally, start joystickd on your laptop and tell it to publish ZMQ packets over the network:
   ```shell
   # on your laptop
   export ZMQ=1
   tools/joystick/joystickd.py
   ```

---
Now start your car and openpilot should go into debug mode with an alert on startup! The status of the axes will display on the alert, while button statuses print in the shell.

Make sure the conditions are met in the panda to allow controls (e.g. cruise control engaged). You can also make a modification to the panda code to always allow controls.

![Imgur](steer.gif)
