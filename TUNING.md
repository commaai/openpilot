[![](https://i.imgur.com/UetIFyH.jpg)](#)

Welcome to openpilot - Tuning mod
======

This OpenPilot mod allows you to dynamically modify variables used by OpenPilot.
The purpose of this mod is to make it easier to tweak certain variables instead of
having to modify code, recompile, reboot after every change.

To use this mod you need to do 2 things:

1. Create a file called **/sdcard/tuning/params.txt** on your EON.

Copy the file called **selfdrive/ui/params.example.txt** into **/sdcard/tuning** (create the directory if needed).
You will need to specify which variables you want the Tuning mod to manage by adding them to that file.

2. Modify OpenPilot code that uses the variable so that it is read from this file instead of hard coded.  This is left for the user to figure out and implement.

For questions or info about this mod, visit the comma slack channel #mod-tuning

CHANGE LOG:

v0.0.1 - Initial version

