#!/usr/bin/env bash

SETUP_DONE="/data/.openpilot_setup_done"
PARAMS_D="/data/params/d"
INSTALLER_BIN="/tmp/installer"
INSTALLER_STAGED="/data/installer_staged"
INSTALLER_DONE="/data/installer_done"

cd /data/openpilot
touch prebuilt
rm -f "$INSTALLER_DONE"
if [[ ! -f "$SETUP_DONE" ]]; then
  rm -f "$PARAMS_D/HasAcceptedTerms" "$PARAMS_D/CompletedTrainingVersion"
  rm -f /data/.openpilot_cache

  # Pre-stage built installer to a separate path (setup will copy to /tmp/installer)
  BUILT_INSTALLER="selfdrive/ui/installer/installers/installer_openpilot"
  rm -f "$INSTALLER_BIN"
  if [[ -f "$BUILT_INSTALLER" ]]; then
    cp "$BUILT_INSTALLER" "$INSTALLER_STAGED"
    chmod +x "$INSTALLER_STAGED"
  fi

  export PYTHONPATH="$PWD"
  python system/ui/mici_setup.py &
  SETUP_PID=$!

  # Wait for installer binary to appear (like AGNOS)
  while [[ ! -f "$INSTALLER_BIN" ]]; do
    sleep 0.1
  done

  # Run installer on top of setup (no flash)
  chmod +x "$INSTALLER_BIN"
  "$INSTALLER_BIN" &

  # Wait for installer to signal done
  while [[ ! -f "$INSTALLER_DONE" ]]; do
    sleep 0.5
  done

  kill $SETUP_PID 2>/dev/null
  rm -f "$INSTALLER_BIN" "$INSTALLER_DONE" "$INSTALLER_STAGED"
  touch "$SETUP_DONE"
fi
exec ./launch_openpilot.sh
