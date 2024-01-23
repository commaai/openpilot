#!/bin/bash

#start pulseaudio daemon
sudo pulseaudio -D > /dev/null 2>&1

# create a virtual null audio and set it to default device
sudo pactl load-module module-null-sink sink_name=virtual_audio > /dev/null 2>&1
sudo pactl set-default-sink virtual_audio > /dev/null 2>&1
