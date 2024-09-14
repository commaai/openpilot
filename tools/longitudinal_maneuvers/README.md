# Longitudinal Maneuvers Testing Tool

Test your vehicle's longitudinal control adherence with this tool. The tool will test the vehicle's ability to follow a few simple longitudinal maneuvers and includes a tool to generate a report afterwards.

Sample snapshot of a report:

![longitudinal test tool output](https://github.com/user-attachments/assets/4cddb012-5fc9-4207-ab40-71e0b3812218)

Sample Test Video:

https://github.com/user-attachments/assets/0aec84ce-2a24-44c9-a929-1b1073b00039

(A full uninterrupted test is currently about 4 minutes long).

## Step-by-Step Instructions

These instructions are fairly detailed but may not cover every possible issue. If you have questions, please ask on Discord.

1. Either run derivatives of the `master` branch such as `nightly` or `master-ci` that includes this tool on your comma device or even consider run the `master` branch itself (remember to pull the submodules!) on your comma device.
2. Make sure you are able to SSH into the comma device.
3. Make sure you can SSH from a laptop or mobile device that you want accompanying you in the car and you should know how to SSH into your comma device "on the road" from your mobile device. Example workable setups include:
   * Have comma prime and [use the SSH proxy](https://docs.comma.ai/how-to/connect-to-comma/) from your laptop or mobile device.
   * Tether to your phone and use the phone's hotspot. Connect your laptop or use your mobile device to SSH to the comma device whose IP you may find under Network -> Advanced.
4. Locate a good place and good time to execute your test. Flat, large parking lots or flat, long, empty roads near industrial areas and/or railroads in times of low traffic are good places to start. Use Google Maps or Google Earth to find a good spot. These places and times should be free of traffic and pedestrians. If you can locate a place with 0.8 miles or 1.3 km of relatively flat and straight road, that would be perfect, but it is not necessary. *You can interrupt the suite of tests and resume it after repositioning your vehicle such as turning it around if need be.*
5. Go to your test starting location and time. Turn off the vehicle.
6. With the comma device now in "offroad" mode, meaning not showing the road, SSH into the device and run the following command:

   ```sh
   echo -n 1 > /data/params/d/LongitudinalManeuverMode
   ```
7. Turn your vehicle back on. You will see this screen:

   ![videoframe_6652](https://github.com/user-attachments/assets/e9d4c95a-cd76-4ab7-933e-19937792fa0f)

8. Check if the road ahead is clear and that the area is safe to drive. If so, press "Set" on your steering wheel to start the tests. The tests will run for about 4 minutes. If you need to pause the tests, press "Cancel" on your steering wheel. You can resume the tests by pressing "Resume" on your steering wheel.
9. When the testing is complete, you'll see this:

   ![fin2](https://github.com/user-attachments/assets/c06960ae-7cfb-44af-beaa-4dc28848e49d)

   Pull over and stop the vehicle. Turn off the vehicle.

   Once the vehicle is off and your vehicle enters "offroad" mode, `LongitudinalManeuverMode` will automatically be unset. The next time your vehicle enters "onroad" mode, you will be in openpilot's normal mode.

   If you want to run another test, before your start your vehicle again, repeat the SSH command to enable `LongitudinalManeuverMode` again.

10. Once you are satisfied you have collected enough data, retire to your workstation and begin the process to generate a report.
11. Ensure your comma device is connected to suitable non-mobile Wi-Fi.
12. Visit https://connect.comma.ai and locate "Longitudinal Maneuvers Mode" routes. They will stand out with lots of orange intervals in their timeline:

    ![image](https://github.com/user-attachments/assets/cfe4c6d9-752f-4b24-b421-4b90a01933dc)
13. Under "Files", make sure "All Logs" is uploaded. If not, click on All Logs to upload them and wait for the uploads to complete.
14. Under "More Info" under each route, you will find a route ID. This route ID will be an argument to the script that generates the report:

    ![image](https://github.com/user-attachments/assets/50a92cce-c426-460e-ae36-f84146d329cc)

    Remove the /# at the end to get the route ID. E.g. `fe18f736cb0d7813/00000303--48e2583fa7/2` becomes `fe18f736cb0d7813/00000303--48e2583fa7`.

15. To generate a report for each route, run the following command inside an openpilot development environment:

    ```sh
    cd tools/longitudinal_maneuvers
    python tools/longitudinal_maneuvers/generate_report.py <the route id from the previous step>
    ```

    If you don't have a local environment handy, be aware that creating a GitHub Codespaces off of the `openpilot` repository will give you a suitable development environment to run the command above.

    ![image](https://github.com/user-attachments/assets/345f5076-b114-4e5e-8503-338a8ad24d5e)


16. Open the generated report(s) in your browser. You have a few choices in how to share them. It is possible to save the report as a PDF or take whole page snapshots. If the generated HTML is small enough, you can also upload it to a [JSFiddle](https://jsfiddle.net) and share that. Here is an example: https://jsfiddle.net/oydqw481/.

