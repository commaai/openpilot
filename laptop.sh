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

set -x

# laptop must have
device() {
  tools/scripts/adb_ssh.sh "$@"
}

echo -e "${BOLD}${GREEN}Prepping device...${NC}"
echo -e "${BOLD}${GREEN}==================${NC}\n"
device "exit 0"
device "tmux list-sessions > /dev/null 2>&1 && tmux kill-server || true"
device "tmux new-session -s laptop -e BLOCK=modeld -d /usr/comma/comma.sh"


# run modeld
#scons
echo -e "${BOLD}${GREEN}Prepping laptop...${NC}"
echo -e "${BOLD}${GREEN}==================${NC}\n"
source tools/install_python_dependencies.sh

pkill -f bridge || true

# broadcast to ZMQ
cereal/message/bridge
t

ZMQ=1 selfdrive/modeld/modeld.py
