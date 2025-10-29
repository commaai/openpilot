#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

if [ -f /TICI ]; then
  echo "This should be run from the laptop!"
  exit 1
fi

device() {
  tools/scripts/adb_ssh.sh "$@"
}

echo -e "${BOLD}${GREEN}Prepping device...${NC}"
echo -e "${BOLD}${GREEN}==================${NC}\n"
device "exit 0"
device "tmux list-sessions > /dev/null 2>&1 && tmux kill-server || true"
device "tmux new-session -s laptop -e PYTHONPATH='/data/pythonpath' -e FINGERPRINT='TOYOTA_COROLLA_TSS2' -e ZMQ=1 -e BLOCK=modeld -d 'source /etc/profile && env && /data/openpilot/system/manager/manager.py'"

# run modeld
echo -e "${BOLD}${GREEN}Prepping laptop...${NC}"
echo -e "${BOLD}${GREEN}==================${NC}\n"
source tools/install_python_dependencies.sh
scons

set -x

adb forward --remove tcp:58537
adb forward --remove tcp:50972
adb forward --remove tcp:49244

tools/camerastream/compressed_vipc.py --silent 127.0.0.1 &

ZMQ=1 selfdrive/modeld/modeld.py
