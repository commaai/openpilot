#!/usr/bin/env bash
#
# Applies the default car fingerprint on first boot (installation time).
#
# If the user has not already set a manual car selection
# (ForceCarFingerprint param), this reads selfdrive/car/default_car.conf
# and writes it so card.py picks it up instead of auto-detecting.
#
# Safe to run on every boot: it only acts when no selection exists yet.

set -e

PARAMS_DIR="/data/params/d"
PARAM_KEY="ForceCarFingerprint"
PARAM_FILE="${PARAMS_DIR}/${PARAM_KEY}"
CONF_FILE="$(dirname "$0")/default_car.conf"

# already set by the user? keep their choice
if [ -f "${PARAM_FILE}" ] && [ -s "${PARAM_FILE}" ]; then
  exit 0
fi

# read default car from config
if [ ! -f "${CONF_FILE}" ]; then
  exit 0
fi

CAR_FP=$(tr -d '[:space:]' < "${CONF_FILE}")
if [ -z "${CAR_FP}" ]; then
  exit 0
fi

mkdir -p "${PARAMS_DIR}"
printf '%s' "${CAR_FP}" > "${PARAM_FILE}"

echo "[apply_default_car] set ForceCarFingerprint=${CAR_FP}"
