Shane's Stock Additions (0.7)
=====

This branch is simply stock openpilot with some additions to help it drive as smooth as possible on my 2017 Toyota Corolla.


Highlight Features
====

1. **(NOT YET ADDED) Dynamic gas**: This aims to provide a smoother driving experience in stop and go traffic by modifying the maximum gas that can be applied based on your current velocity and the relative velocity of the lead car. It'll also of course increase the maximum gas when the lead is accelerating to help you get up to speed quicker than stock. And smoother; this eliminates the jerking you get from stock openpilot with comma pedal. Better tuning will be next.
2. **Dynamic follow**: This is my dynamic follow from 0.5, where it changes your TR (following distance) dynamically based on multiple vehicle factors, as well as data from the lead vehicle. [Here's an old write up from a while ago explaining how it works exactly. Some of it might be out of date, but how it functions is the same.](https://github.com/ShaneSmiskol/openpilot/blob/dynamic-follow/README.md)
3. **(NOT YET ADDED) Two PID loops to control gas and brakes independently (new!)**: If you have a Toyota Corolla with a comma pedal, you'll love this addition. Two longitudinal PID loops are set up in `longcontrol.py` so that one is running with comma pedal tuning to control the gas, and the other is running stock non-pedal tuning for better braking control. In the car, this feels miles better than stock openpilot, and nearly as good as your stock Toyota cruise control before you pulled out your DSU! It won't accelerate up to stopped cars and brake at the last moment anymore.
3. **Custom wheel offset to reduce lane hugging**: Stock openpilot doesn't seem to be able to identify your car's true angle offset. With the `LaneHugging` module you can specify a custom angle offset to be added to your desired steering angle. Simply find the angle your wheel is at when you're driving on a straight highway. By default, this is disabled, to enable you can:
    - Use the `opEdit` class in the root directory of openpilot. To use it, simply open an `ssh` shell and enter the commands below:
        ```python
        cd /data/openpilot
        python op_edit.py
        ```
        You'll be greeted with a list of your parameters you can explore, enter the number corresponding to `lane_hug_direction`. Your options are to enter `'left'` or `'right'` for whichever direction your car has a tendency to hug toward. `None` will disable the feature.
        Finally you'll need to enter your absolute angle offset (negative will be converted to positive) with the `opParams` parameter: `lane_hug_angle_offset`.
4. **Custom following distance**: Using the `following_distance` parameter in `opParams`, you can specify a custom TR value to always be used. Afraid of technology and want to give yourself the highest following distance out there? Try out 2.7s! Are you daredevil and don't care about pissing off the car you're tailgating ahead? Try 0.9s!
    - Again, you can use `opEdit` to change this:
        ```python
        cd /data/openpilot
        python op_edit.py
        ```
        Then enter the number for the `following_distance` parameter and set to a float or integer between `0.9` and `2.7`. `None` will use dynamic follow!
5. **Customize this branch (opEdit Parameter class)**: This is a handy tool to change your `opParams` parameters without diving into any json files or code. You can specify parameters to be used in any fork's operation that supports `opParams`. First, ssh in to your EON and make sure you're in `/data/openpilot`, then start `opEdit`:
    ```python
    cd /data/openpilot
    python op_edit.py
    ```
   A list of parameters that you can change are located [here](https://github.com/ShaneSmiskol/openpilot/blob/stock_additions-07/common/op_params.py#L29).

    Parameters are stored at `/data/op_params.json`