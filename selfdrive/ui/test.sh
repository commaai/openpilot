#!/usr/bin/bash

# display off
./set_power 0
sleep 2
echo "turning display on"
./set_power 2
sleep 1
./set_power 0
