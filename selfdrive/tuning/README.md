# Tuning Tools

Tools for testing and tuning openpilot's lateral and longitudinal control, plus a joystick debug mode.

## Joystick

**Hardware needed**: device running openpilot, laptop, joystick (optional)

With joystick_control, you can connect your laptop to your comma device over the network and debug controls using a joystick or keyboard.
joystick_control uses [inputs](https://pypi.org/project/inputs) which supports many common gamepads and joysticks.

### Usage

The car must be off, and openpilot must be offroad before starting `joystick_control`.

### Using a keyboard

SSH into your comma device and start joystick_control with the following command:

```shell
selfdrive/tuning/joystick_control.py --keyboard
```

The available buttons and axes will print showing their key mappings. In general, the WASD keys control gas and brakes and steering torque in 5% increments.

### Joystick on your comma three

Plug the joystick into your comma three aux USB-C port. Then, SSH into the device and start `joystick_control.py`.

### Joystick on your laptop

In order to use a joystick over the network, we need to run joystick_control locally from your laptop and have it send `testJoystick` packets over the network to the comma device.

1. Connect a joystick to your PC.
2. Connect your laptop to your comma device's hotspot and open a new SSH shell. Since joystick_control is being run on your laptop, we need to write a parameter to let controlsd know to start in joystick debug mode:
   ```shell
   # on your comma device
   echo -n "1" > /data/params/d/JoystickDebugMode
   ```
3. Run bridge with your laptop's IP address. This republishes the `testJoystick` packets sent from your laptop so that openpilot can receive them:
   ```shell
   # on your comma device
   cereal/messaging/bridge {LAPTOP_IP} testJoystick
   ```
4. Start joystick_control on your laptop in ZMQ mode.
   ```shell
   # on your laptop
   export ZMQ=1
   selfdrive/tuning/joystick_control.py
   ```

---
Now start your car and openpilot should go into joystick mode with an alert on startup! The status of the axes will display on the alert, while button statuses print in the shell.

Make sure the conditions are met in the panda to allow controls (e.g. cruise control engaged). You can also make a modification to the panda code to always allow controls.

![](https://github.com/commaai/openpilot/assets/8762862/e640cbca-cb7a-4dcb-abce-b23b036ad8e7)

## Longitudinal Maneuvers

Test your vehicle's longitudinal control tuning with this tool. The tool will test the vehicle's ability to follow a few longitudinal maneuvers and includes a tool to generate a report from the route.

<details><summary>Sample snapshot of a report.</summary><img width="600px" src="https://github.com/user-attachments/assets/d18d0c7d-2bde-44c1-8e86-1741ed442ad8"></details>

### Instructions

1. Check out a development branch such as `master` on your comma device.
2. Locate either a large empty parking lot or road devoid of any car or foot traffic. Flat, straight road is preferred. The full maneuver suite can take 1 mile or more if left running, however it is recommended to disengage openpilot between maneuvers and turn around if there is not enough space.
3. Turn off the vehicle and set this parameter which will signal to openpilot to start the longitudinal maneuver daemon:

   ```sh
   echo -n 1 > /data/params/d/LongitudinalManeuverMode
   ```

4. Turn your vehicle back on. You will see the "Longitudinal Maneuver Mode" alert:

   ![videoframe_6652](https://github.com/user-attachments/assets/e9d4c95a-cd76-4ab7-933e-19937792fa0f)

5. Ensure the road ahead is clear, as openpilot will not brake for any obstructions in this mode. Once you are ready, press "Set" on your steering wheel to start the tests. The tests will run for about 4 minutes. If you need to pause the tests, press "Cancel" on your steering wheel. You can resume the tests by pressing "Resume" on your steering wheel. 

   **Note:** For GM cars, it is recommended to hold down the resume button for all low-speed tests (starting, stopping and creep) to avoid the car entering standstill.

   ![cog-clip-00 01 11 250-00 01 22 250](https://github.com/user-attachments/assets/c312c1cc-76e8-46e1-a05e-bb9dfb58994f)

6. When the testing is complete, you'll see an alert that says "Maneuvers Finished." Complete the route by pulling over and turning off the vehicle.

   ![fin2](https://github.com/user-attachments/assets/c06960ae-7cfb-44af-beaa-4dc28848e49f)

7. Visit https://connect.comma.ai and locate the route(s). They will stand out with lots of orange intervals in their timeline. Ensure "All logs" show as "uploaded."

   ![image](https://github.com/user-attachments/assets/cfe4c6d9-752f-4b24-b421-4b90a01933dc)

8. Gather the route ID and then run the report generator. The file will be exported to the same directory:

    ```sh
    $ python selfdrive/tuning/generate_longitudinal_report.py 57048cfce01d9625/0000010e--5b26bc3be7 'pcm accel compensation'

    processing report for LEXUS_ES_TSS2
    plotting maneuver: start from stop, runs: 4
    plotting maneuver: creep: alternate between +1m/s^2 and -1m/s^2, runs: 2
    plotting maneuver: gas step response: +1m/s^2 from 20mph, runs: 2

    Report written to /home/batman/openpilot/selfdrive/tuning/longitudinal_reports/LEXUS_ES_TSS2_57048cfce01d9625_0000010e--5b26bc3be7.html
    ```

You can reach out on [Discord](https://discord.comma.ai) if you have any questions about these instructions or the tool itself.

## Lateral Maneuvers

> [!WARNING]
> Use caution when using this tool.

Test your vehicle's lateral control tuning with this tool. The tool will test the vehicle's ability to follow a few lateral maneuvers and includes a tool to generate a report from the route.

### Instructions

1. Check out a development branch such as `master` on your comma device.
2. The full maneuver suite runs at 20 and 30 mph.
3. Enable "Lateral Maneuver Mode" in Settings > Developer on the device while offroad. Alternatively, set the parameter manually:

   ```sh
   echo -n 1 > /data/params/d/LateralManeuverMode
   ```

4. Turn your vehicle back on. You will see "Lateral Maneuver Mode".

5. Ensure the area ahead is clear, as openpilot will command lateral acceleration steps in this mode. Once you are ready, set ACC manually to the target speed shown on screen and let openpilot stabilize lateral. After 1 seconds of steady straight driving, the maneuver will begin automatically. openpilot lateral control stays engaged between maneuvers normally while waiting for the next maneuver's readiness conditions. The maneuver will be aborted and repeated if speed is out of range, steering is touched or openpilot disengages.

6. When the testing is complete, you'll see an alert that says "Maneuvers Finished." Complete the route by pulling over and turning off the vehicle.

7. Visit https://connect.comma.ai and locate the route(s). They will stand out with lots of orange intervals in their timeline. Ensure "All logs" show as "uploaded."

   ![image](https://github.com/user-attachments/assets/cfe4c6d9-752f-4b24-b421-4b90a01933dc)

8. Gather the route ID and then run the report generator. The file will be exported to the same directory:

    ```sh
    $ python selfdrive/tuning/generate_lateral_report.py 98395b7c5b27882e/000001cc--5a73bde686

    processing report for KIA_EV6
    plotting maneuver: step right 20mph, runs: 3
    plotting maneuver: step left 20mph, runs: 3
    plotting maneuver: sine 0.5Hz 20mph, runs: 3
    plotting maneuver: step right 30mph, runs: 3

    Opening report: /home/batman/openpilot/selfdrive/tuning/lateral_reports/KIA_EV6_98395b7c5b27882e_000001cc--5a73bde686.html
    ```

You can reach out on [Discord](https://discord.comma.ai) if you have any questions about these instructions or the tool itself.
