#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$BASEDIR" ]; then
  BASEDIR="/data/openpilot"
fi

if [ -z "$PASSIVE" ]; then
  export PASSIVE="1"
fi

STAGING_ROOT="/data/safe_staging"

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
          exec "${LAUNCHER_LOCATION}"
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

  # Collect RIL and other possibly long-running I/O interrupts onto CPU 1
  echo 1 > /proc/irq/78/smp_affinity_list # qcom,smd-modem (LTE radio)
  echo 1 > /proc/irq/33/smp_affinity_list # ufshcd (flash storage)
  echo 1 > /proc/irq/35/smp_affinity_list # wifi (wlan_pci)
  # USB traffic needs realtime handling on cpu 3
  [ -d "/proc/irq/733" ] && echo 3 > /proc/irq/733/smp_affinity_list # USB for LeEco
  [ -d "/proc/irq/736" ] && echo 3 > /proc/irq/736/smp_affinity_list # USB for OP3T

  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

  # Remove old NEOS update file
  # TODO: move this code to the updater
  if [ -d /data/neoupdate ]; then
    rm -rf /data/neoupdate
  fi

  # Check for NEOS update
  if [ $(< /VERSION) != "14" ]; then
    if [ -f "$DIR/scripts/continue.sh" ]; then
      cp "$DIR/scripts/continue.sh" "/data/data/com.termux/files/continue.sh"
    fi

    if [ ! -f "$BASEDIR/prebuilt" ]; then
      echo "Clearing build products and resetting scons state prior to NEOS update"
      cd $BASEDIR && scons --clean
      rm -rf /tmp/scons_cache
      rm -r $BASEDIR/.sconsign.dblite
    fi
    "$DIR/installer/updater/updater" "file://$DIR/installer/updater/update.json"
  else
    if [[ $(uname -v) == "#1 SMP PREEMPT Wed Jun 10 12:40:53 PDT 2020" ]]; then
      "$DIR/installer/updater/updater" "file://$DIR/installer/updater/update_kernel.json"
    fi
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
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # start manager
  cd selfdrive
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
