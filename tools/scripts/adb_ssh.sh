#!/usr/bin/env bash
set -euo pipefail

# Forward all openpilot service ports
while IFS=' ' read -r name port; do
  adb forward "tcp:${port}" "tcp:${port}" > /dev/null
done < <(python3 - <<'PY'
from openpilot.cereal.services import SERVICE_LIST

FNV_PRIME = 0x100000001b3
FNV_OFFSET_BASIS = 0xcbf29ce484222325
START_PORT = 8023
MAX_PORT = 65535
PORT_RANGE = MAX_PORT - START_PORT
MASK = 0xffffffffffffffff

def fnv1a(endpoint: str) -> int:
  h = FNV_OFFSET_BASIS
  for b in endpoint.encode():
    h ^= b
    h = (h * FNV_PRIME) & MASK
  return h

ports = set()
for name in SERVICE_LIST.keys():
  port = START_PORT + fnv1a(name) % PORT_RANGE
  ports.add((name, port))

for name, port in sorted(ports):
  print(f"{name} {port}")
PY
)

# Forward SSH on a free local port.
SSH_PORT=$(adb forward tcp:0 tcp:22)

# SSH!
ssh comma@localhost -p ${SSH_PORT} "$@"
