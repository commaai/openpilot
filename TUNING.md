Welcome to openpilot - Tuning mod
======

This openpilot mod allows you to dynamically modify variables used by openpilot.
The purpose of this mod is to make it easier to tweak certain variables instead of
having to modify code, recompile, reboot after every change.

To use this mod you need to do 2 things:

1. Create a file called **/sdcard/tuning/params.txt** on your EON.

You will need to specify which variables you want the Tuning mod to manage by adding them to that file.  See the file called **selfdrive/ui/params.example.txt** for an example.

2. Modify OpenPilot code that uses the variable so that it is read from this file instead of hard coded.  This is left for the user to figure out and implement.

This mod can manage up to 10 different variables with each variable having a maximum of 3 element values in it.

For questions or info about this mod, visit the comma slack channel #mod-tuning

To change the "scale" of the steps tap the "Steps" box.  It will cycle through different scales.  The step scales are: \[0.001, 0.01, 0.1, 1, 5\]

![screenshot](https://i.imgur.com/G7j2vQY.jpg)

CHANGE LOG:

v0.0.1 - Initial version

To Do:
- [ ] Ability to handle hex values
