#!/usr/bin/bash

# then, connect to computer:
# adb connect 192.168.5.11:5555

setprop service.adb.tcp.port 5555
stop adbd
start adbd
