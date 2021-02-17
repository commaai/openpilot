#!/bin/bash

cd ~/openpilot/tools/nui

# vision, boardd, sensorsd
ALLOW=frame,can,ubloxRaw,health,sensorEvents,gpsNMEA,gpsLocationExternal ./nui "02ec6bea180a4d36/2019-10-25--10-18-09"
