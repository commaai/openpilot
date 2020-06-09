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

          # The mv changed our working directory to /data/safe_staging/old_openpilot
          cd "${BASEDIR}"

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

  # Configure interrupt affinities for NEOS platforms
  # Slight differences between OP3T and LeEco for I2C, SPS, and USB, trying both for now
  # TODO: abstract this into per-platform and per-mainboard startup scripts
  #
  # Move RIL and other possibly long-running I/O interrupts onto core 1
  # Move USB to core 3 for better realtime handling
  IRQ_AFFINE_CORE_0=""
  IRQ_AFFINE_CORE_1="13 16 25 33 35 78"  # I2C, NGD, ufshcd (flash storage), wlan_pci (WiFi)
  IRQ_AFFINE_CORE_2="6 15 26"  # SPS, MDSS
  IRQ_AFFINE_CORE_3="733 736"  # USB for LeEco and OP3T mainboards respectively

  for CORE in {0..3}
    do
      CORE_IRQS=IRQ_AFFINE_CORE_$CORE
      for IRQ in ${!CORE_IRQS}
        do
          if [ -d "/proc/irq/$IRQ" ]; then
            echo "Setting IRQ affinity: IRQ $IRQ to core $CORE"
            echo $CORE > /proc/irq/$IRQ/smp_affinity_list
          fi
        done
    done

  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

  # Remove old NEOS update file
  # TODO: move this code to the updater
  if [ -d /data/neoupdate ]; then
    rm -rf /data/neoupdate
  fi

  # Check for NEOS update
  if [ $(< /VERSION) != "15" ]; then
    if [ -f "$DIR/scripts/continue.sh" ]; then
      cp "$DIR/scripts/continue.sh" "/data/data/com.termux/files/continue.sh"
    fi

    "$DIR/installer/updater/updater" "file://$DIR/installer/updater/update.json"
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
