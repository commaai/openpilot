#!/usr/bin/bash

HOME=~/one

if [ ! -d $HOME ]; then
  HOME=/data/chffrplus
fi

camera_test () {
  printf "Running camera test...\n"

  cd $HOME/selfdrive/visiond

  if [ ! -e visiond ]; then
    make > /dev/null
  fi

  CAMERA_TEST=1 ./visiond > /dev/null
  V4L_SUBDEVS=$(find -L /sys/class/video4linux/v4l-subdev* -maxdepth 1 -name name -exec cat {} \;)
  CAMERA_COUNT=0
  for SUBDEV in $V4L_SUBDEVS; do
    if [ "$SUBDEV" == "imx298" ] || [ "$SUBDEV" == "ov8865_sunny" ]; then
      CAMERA_COUNT=$((CAMERA_COUNT + 1))
    fi
  done

  if [ "$CAMERA_COUNT" == "2" ]; then
    printf "Camera test: SUCCESS!\n"
  else
    printf "One or more cameras are missing! Camera count: $CAMERA_COUNT\n"
    exit 1
  fi
}

sensor_test () {
  printf "Running sensor test...\n"

  cd $HOME/selfdrive/sensord

  if [ ! -e sensord ]; then
    make > /dev/null
  fi

  SENSOR_TEST=1 LD_LIBRARY_PATH=/system/lib64:$LD_LIBRARY_PATH ./sensord
  SENSOR_COUNT=$?

  if [ "$SENSOR_COUNT" == "40" ]; then
    printf "Sensor test: SUCCESS!\n"
  else
    printf "One or more sensors are missing! Sensor count: $SENSOR_COUNT, expected 40\n"
    exit 1
  fi
}

wifi_test () {
  printf "Running WiFi test...\n"

  su -c 'svc wifi enable'
  WIFI_STATUS=$(getprop wlan.driver.status)

  if [ "$WIFI_STATUS" == "ok" ]; then
    printf "WiFi test: SUCCESS!\n"
  else
    printf "WiFi isn't working! Driver status: $WIFI_STATUS\n"
    exit 1
  fi
}

modem_test () {
  printf "Running modem test...\n"

  BASEBAND_VERSION=$(getprop gsm.version.baseband | awk '{print $1}')

  if [ "$BASEBAND_VERSION" == "MPSS.TH.2.0.c1.9.1-00010" ]; then
    printf "Modem test: SUCCESS!\n"
  else
    printf "Modem isn't working! Detected baseband version: $BASEBAND_VERSION\n"
    exit 1
  fi
}

fan_test () {
  printf "Running fan test...\n"

  i2cget -f -y 7 0x67 0 1>/dev/null 2>&1
  IS_NORMAL_LEECO=$?

  if [ "$IS_NORMAL_LEECO" == "0" ]; then
    /tmp/test_leeco_alt_fan.py > /dev/null
  else
    /tmp/test_leeco_fan.py > /dev/null
  fi

  printf "Fan test: the fan should now be running at full speed, press Y or N\n"

  read -p "Is the fan running [Y/n]?\n" fan_running
  case $fan_running in
    [Nn]* )
      printf "Fan isn't working! (user says it isn't working)\n"
      exit 1
      ;;
  esac

  printf "Turning off the fan ...\n"
  if [ "$IS_NORMAL_LEECO" == "0" ]; then
    i2cset -f -y 7 0x67 0xa 0
  else
    i2cset -f -y 7 0x3d 0 0x1
  fi
}

camera_test
printf "\n"

sensor_test
printf "\n"

wifi_test
printf "\n"

modem_test
printf "\n"

fan_test
