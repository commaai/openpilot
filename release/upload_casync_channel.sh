#!/bin/bash

CASYNC_DIR="${CASYNC_DIR:=/tmp/casync}"

OPENPILOT_CHANNELS="https://commadist.blob.core.windows.net/openpilot-channels/"

SAS="$(python -c 'from tools.lib.azure_container import get_container_sas;print(get_container_sas("commadist","openpilot-channels"))')"

azcopy cp "$CASYNC_DIR*" "$OPENPILOT_CHANNELS?$SAS" --recursive
