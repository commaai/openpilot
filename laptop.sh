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

set -x

echo -e "${BOLD}${GREEN}Prepping device...${NC}"
echo -e "${BOLD}${GREEN}==================${NC}\n"
device "sudo ip link set usb0 up"
device "sudo ip addr flush dev usb0"
device "sudo ip addr add 192.168.64.1/24 dev usb0"
device "tmux list-sessions > /dev/null 2>&1 && tmux kill-server || true"
device "tmux new-session -s laptop -e PYTHONPATH='/data/pythonpath' -e FINGERPRINT='TOYOTA_COROLLA_TSS2' -e BLOCK=modeld -d 'source /etc/profile && /data/continue.sh'"
device "tmux new-window -t laptop 'sudo systemctl restart lte'"
device "tmux new-window -t laptop 'cd /data/openpilot/cereal/messaging && ./bridge'"
device "tmux new-window -t laptop 'cd /data/openpilot/cereal/messaging && ./bridge 192.168.64.2 modelV2,drivingModelData,cameraOdometry'"

get_usb_ncm_iface() {
  for i in /sys/class/net/*; do
    iface=${i##*/}
    [[ -f "$i/type" && $(<"$i/type") -eq 1 ]] || continue
    readlink -f "$i" | grep -q '/usb' || continue
    if ethtool -i "$iface" 2>/dev/null | grep -qE 'driver: (cdc_ncm|huawei_cdc_ncm|cdc_mbim)'; then
      echo "$iface"
      return 0
    fi
  done
  return 1
}
iface=$(get_usb_ncm_iface) && echo "NCM interface: $iface"
sudo ip addr add 192.168.64.2/30 dev $iface || true
sudo ip link set dev $iface up
ping -c 5 192.168.64.1

# run modeld
echo -e "${BOLD}${GREEN}Prepping laptop...${NC}"
echo -e "${BOLD}${GREEN}==================${NC}\n"
source .venv/bin/activate
tools/camerastream/compressed_vipc.py --silent --server camerad 192.168.64.1 &
selfdrive/modeld/modeld.py
