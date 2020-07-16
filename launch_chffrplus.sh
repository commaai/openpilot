#!/usr/bin/bash

if [ -z "$BASEDIR" ]; then
  export BASEDIR="/data/openpilot"
fi

source "$BASEDIR/launch_env.sh"

function launch {
  # Wifi scan
  wpa_cli IFNAME=wlan0 SCAN

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The BASEDIR init file has to exist, with a newer modtime than anything in
  #    the BASEDIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${BASEDIR}/.overlay_init" ]; then
    find ${BASEDIR}/.git -newer ${BASEDIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${BASEDIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $BASEDIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $BASEDIR
          cd $BASEDIR

          # Partial mitigation for symlink-related filesystem corruption
          # Ensure all files match the repo versions after update
          git reset --hard
          git submodule foreach --recursive git reset --hard

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          cd "${BASEDIR}" && exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # Android and other system processes are not permitted to run on CPU 3
  # NEOS installed app processes can run anywhere
  echo 0-2 > /dev/cpuset/background/cpus
  echo 0-2 > /dev/cpuset/system-background/cpus
  [ -d "/dev/cpuset/foreground/boost/cpus" ] && echo 0-2 > /dev/cpuset/foreground/boost/cpus  # Not present in < NEOS 15
  echo 0-2 > /dev/cpuset/foreground/cpus
  echo 0-2 > /dev/cpuset/android/cpus
  echo 0-3 > /dev/cpuset/app/cpus

  # Configure non-default interrupt affinities for NEOS platforms
  # Slight differences between OP3T and LeEco for I2C, SPS, and USB, trying both for now
  # TODO: abstract this into per-platform and per-mainboard startup scripts
  #
  # Move RIL and other possibly long-running I/O interrupts onto core 1
  # Move USB to core 3 for better realtime handling
  IRQ_AFFINE_CORE_0=""
  IRQ_AFFINE_CORE_1="13 16 25 78"  # I2C, NGD, qcom,smd-modem
  IRQ_AFFINE_CORE_2="6 15 26 33 35"  # SPS, MDSS, ufshcd (flash storage), wlan_pci (WiFi)
  IRQ_AFFINE_CORE_3="733 736"  # USB for LeEco and OP3T mainboards respectively

  for CORE in {0..3}
  do
    CORE_IRQ_LIST=IRQ_AFFINE_CORE_$CORE
    for IRQ in ${!CORE_IRQ_LIST}
    do
      if [ -d "/proc/irq/$IRQ" ]; then
        echo "Setting IRQ affinity: IRQ $IRQ to core $CORE"
        echo $CORE > /proc/irq/$IRQ/smp_affinity_list
      fi
    done
  done

  # Check for NEOS update
  if [ "$(< /VERSION)" != "$REQUIRED_NEOS_VERSION" ]; then
    if [ -f "$BASEDIR/scripts/continue.sh" ]; then
      cp "$BASEDIR/scripts/continue.sh" "/data/data/com.termux/files/continue.sh"
    fi

    if [ ! -f "$BASEDIR/prebuilt" ]; then
      echo "Clearing build products and resetting scons state prior to NEOS update"
      git clean -xdf
      git submodule foreach --recursive git clean -xdf
      rm -rf /tmp/scons_cache
    fi

    "$BASEDIR/installer/updater/updater" "file://$BASEDIR/installer/updater/update.json"
  fi

  # Remove old NEOS update files
  if [ -d /data/neoupdate ]; then
    rm -rf /data/neoupdate
  fi

  # One-time fix for a subset of OP3T with gyro orientation offsets.
  # Remove and regenerate qcom sensor registry. Only done on OP3T mainboards.
  # Performed exactly once. The old registry is preserved just-in-case, and
  # doubles as a flag denoting we've already done the reset.
  # TODO: we should really grow per-platform detect and setup routines
  if ! $(grep -q "letv" /proc/cmdline) && [ ! -f "/persist/comma/op3t-sns-reg-backup" ]; then
    echo "Performing OP3T sensor registry reset"
    mv /persist/sensors/sns.reg /persist/comma/op3t-sns-reg-backup &&
      rm -f /persist/sensors/sensors_settings /persist/sensors/error_log /persist/sensors/gyro_sensitity_cal &&
      echo "restart" > /sys/kernel/debug/msm_subsys/slpi &&
      sleep 5  # Give Android sensor subsystem a moment to recover
  fi

  # handle pythonpath
  ln -sfn "$(pwd)" /data/pythonpath
  export PYTHONPATH="$PWD"

  # start manager
  cd selfdrive && ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
