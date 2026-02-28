#!/usr/bin/env bash

SETUP_DONE="/data/.openpilot_setup_done"
PARAMS_D="/data/params/d"
INSTALLER_BIN="/tmp/installer"

cd /data/openpilot
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

  # Simulate AGNOS: run installer after setup exits
  if [[ -x "$INSTALLER_BIN" ]]; then
    "$INSTALLER_BIN"
    rm -f "$INSTALLER_BIN"
  fi

  touch "$SETUP_DONE"
fi
exec ./launch_openpilot.sh
