#!/usr/bin/env bash

SETUP_DONE="/data/.openpilot_setup_done"
PARAMS_D="/data/params/d"
INSTALLER_BIN="/tmp/installer"
INSTALLER_DONE="/data/installer_done"

cd /data/openpilot
rm -f "$INSTALLER_DONE"
if [[ ! -f "$SETUP_DONE" ]]; then
  rm -f "$PARAMS_D/HasAcceptedTerms" "$PARAMS_D/CompletedTrainingVersion"
  rm -f /data/.openpilot_cache

  # Pre-stage built installer so setup skips the download
  BUILT_INSTALLER="selfdrive/ui/installer/installers/installer_openpilot"
  if [[ -f "$BUILT_INSTALLER" ]]; then
    cp "$BUILT_INSTALLER" "$INSTALLER_BIN"
    chmod +x "$INSTALLER_BIN"
  fi

  export PYTHONPATH="$PWD"
  python system/ui/mici_setup.py

  # Simulate AGNOS: run installer in background, wait for it to signal done
  rm -f "$INSTALLER_DONE"
  if [[ -x "$INSTALLER_BIN" ]]; then
    "$INSTALLER_BIN" &
    while [[ ! -f "$INSTALLER_DONE" ]]; do
      sleep 0.5
    done
    rm -f "$INSTALLER_BIN" "$INSTALLER_DONE"
  fi

  touch "$SETUP_DONE"
fi
exec ./launch_openpilot.sh
