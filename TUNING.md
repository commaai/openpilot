Welcome to openpilot - Tuning mod
======

This openpilot mod allows you to dynamically modify variables used by openpilot.
The purpose of this mod is to make it easier to tweak certain variables instead of
having to modify code, recompile, reboot after every change.

To use this mod you need to do 2 things:

1. Create a file called **/sdcard/tuning/params.txt** on your EON.

You will need to specify which variables you want the Tuning mod to manage by adding them to that file.  See the file called **selfdrive/ui/params.example.txt** for an example.

2. Modify OpenPilot code that uses the variable so that it is read from this file instead of hard coded.  This is left for the user to figure out and implement.  To help you get started with this step you can use the following python code to read in the variables stored by this mod:

```
import imp
f = open("/sdcard/tuning/params.txt")
tuning = imp.load_source('tuning', '', f)
f.close()
````

This will read all of the parameters that you've defined in your **params.txt** file and store them into a python variable called `tuning`.  Then you need to assign the real variables that openpilot uses with the corresponding parameter value.  For example, if openpilot has a hard coded value for `ret.steerKp=[0.2]` Assuming your **params.txt** file contains this parameter:

`
MySteerKp=[0.25]
`

You would then replace that line in the openpilot python file with `ret.steerKp=tuning.MySteerKp`.

This mod can manage up to 10 different variables with each variable having a maximum of 3 element values in it.

For questions or info about this mod, visit the comma slack channel #mod-tuning

To change the "scale" of the steps tap the "Steps" box.  It will cycle through different scales.  The step scales are: \[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5\]

![screenshot](https://i.imgur.com/G7j2vQY.jpg)

CHANGE LOG:

v0.0.4
========================
 * Add ability to specify "presets" in params.txt file (See selfdrive/ui/params.example.txt).  You can have up to 10 "preset" values.  NOTE: The parameter list for each preset MUST match.  When you edit a preset value the values for the current preset will be written to a file called **/sdcard/tuning/tune.txt**.  This is now the file you should be loading into your python code.

v0.0.3
========================
 * Increase precision to 6 decimal places

v0.0.2
========================
 * Added 'press and hold' on the +/- buttons
 * Swapped positions of +/- buttons
 
v0.0.1
========================
 * Initial version

To Do:
- [ ] Ability to handle hex values
- [ ] Suppress baseui from toggling zoom view when tapping
