#!/bin/bash

CASYNC_DIR="${CASYNC_DIR:=/tmp/casync}"

OPENPILOT_RELEASES="https://commadist.blob.core.windows.net/openpilot-releases/"

SAS="$(python -c 'from tools.lib.azure_container import get_container_sas;print(get_container_sas("commadist","openpilot-releases"))')"

azcopy cp "$CASYNC_DIR*" "$OPENPILOT_RELEASES?$SAS" --recursive
