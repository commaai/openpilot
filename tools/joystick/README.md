# Joystick

**Hardware needed**: device running openpilot, laptop, joystick (optional)

With joystickd, you can connect your laptop to your comma device over the network and debug controls using a joystick or keyboard.
joystickd uses [inputs](https://pypi.org/project/inputs) which supports many common gamepads and joysticks.

## Usage

The car must be off, and openpilot must be offroad before starting `joystickd`.

### Using a keyboard

SSH into your comma device and start joystickd with the following command:

```shell
tools/joystick/joystickd.py --keyboard
```

The available buttons and axes will print showing their key mappings. In general, the WASD keys control gas and brakes and steering torque in 5% increments.

### Joystick on your comma three

Plug the joystick into your comma three aux USB-C port. Then, SSH into the device and start `joystickd.py`.

### Joystick on your laptop

In order to use a joystick over the network, we need to run joystickd locally from your laptop and have it send `testJoystick` packets over the network to the comma device.

1. Connect a joystick to your PC.
2. Connect your laptop to your comma device's hotspot and open a new SSH shell. Since joystickd is being run on your laptop, we need to write a parameter to let controlsd know to start in joystick debug mode:
   ```shell
   # on your comma device
   echo -n "1" > /data/params/d/JoystickDebugMode
   ```
3. Run bridge with your laptop's IP address. This republishes the `testJoystick` packets sent from your laptop so that openpilot can receive them:
   ```shell
   # on your comma device
   cereal/messaging/bridge {LAPTOP_IP} testJoystick
   ```
4. Start joystickd on your laptop in ZMQ mode.
   ```shell
   # on your laptop
   export ZMQ=1
   tools/joystick/joystickd.py
   ```

---
Now start your car and openpilot should go into joystick mode with an alert on startup! The status of the axes will display on the alert, while button statuses print in the shell.

Make sure the conditions are met in the panda to allow controls (e.g. cruise control engaged). You can also make a modification to the panda code to always allow controls.

![](https://github.com/commaai/openpilot/assets/8762862/e640cbca-cb7a-4dcb-abce-b23b036ad8e7)
