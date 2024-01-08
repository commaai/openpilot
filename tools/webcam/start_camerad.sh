#!/bin/bash
export DUAL="0"
export BLOCK="logmessaged,ui,logcatd,proclogd,micd,timezoned,dmonitoringmodeld,modeld,mapsd,navmodeld,sensord,soundd,locationd,boardd,calibrationd,torqued,controlsd,deleter,dmonitoringd,qcomgpsd,ubloxd,pigeond,plannerd,radard,thermald,tombstoned,updated,uploader,statsd,bridge,webrtcd,webjoystick,pandad"
export USE_WEBCAM="1"
export YUV_BUFFER_COUNT="20"

#Change camera index according to your setting
export CAMERA_ROAD_ID="0"
export CAMERA_DRIVER_ID="1"
export CAMERA_WIDE_ID="2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
python3 /home/bongb/openpilot/selfdrive/manager/manager.py
#python3 $DIR/camerad.py
