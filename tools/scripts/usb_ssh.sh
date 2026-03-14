#!/usr/bin/env bash
set -euo pipefail

DEVICE_IP="192.168.42.2"
SSH_USER="comma"
TIMEOUT=30

# Allow Ctrl-C to exit cleanly during the ping wait
trap 'echo; exit 130' INT

echo "waiting for comma device at ${DEVICE_IP}..."
elapsed=0
while ! nc -z -w 1 "${DEVICE_IP}" 22 &>/dev/null; do
  elapsed=$((elapsed + 1))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "Timed out waiting for device. Is it connected via USB?" >&2
    exit 1
  fi
  sleep 0.2
done

# SSH into the device
# -o StrictHostKeyChecking=no: different devices share the same IP
# -o UserKnownHostsFile=/dev/null: don't pollute known_hosts with this ephemeral IP
exec ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR \
  "${SSH_USER}@${DEVICE_IP}" "$@"
