#!/usr/bin/bash

if [ $1 -eq 1 ]; then
  printf %s "1" > /data/params/d/CommunityFeaturesToggle
  printf %s "1" > /data/params/d/dp_atl
  printf %s "0" > /data/params/d/dp_uploader
  printf %s "0" > /data/params/d/dp_logger
  printf %s "0" > /data/params/d/dp_athenad
  printf %s "0" > /data/params/d/dp_accel_profile_ctrl
  printf %s "0" > /data/params/d/dp_following_profile_ctrl
  printf %s "0" > /data/params/d/dp_gear_check
fi
if [ $1 -eq 0 ]; then
  printf %s "0" > /data/params/d/dp_atl
  cd /data/openpilot || exit
  git reset --hard @{u}
  git clean -xdf
fi
reboot
