#!/usr/bin/env bash
set -euo pipefail

# Quick QEMU VM launcher for testing openpilot on different distros
# Usage: tools/scripts/vm.sh [distro]
# Boots live ISO desktops — ephemeral, nothing persists.

DIR="$HOME/Downloads"

declare -A ISOS=(
  [arch]="https://geo.mirror.pkgbuild.com/iso/latest/archlinux-x86_64.iso"
  [debian]="https://cdimage.debian.org/debian-cd/current-live/amd64/iso-hybrid/debian-live-12.10.0-amd64-gnome.iso"
  [fedora]="https://download.fedoraproject.org/pub/fedora/linux/releases/42/Workstation/x86_64/iso/Fedora-Workstation-Live-42-1.1.x86_64.iso"
  [manjaro]="https://download.manjaro.org/gnome/26.0.2/manjaro-gnome-26.0.2-260206-linux618.iso"
  [ubuntu]="https://releases.ubuntu.com/24.04/ubuntu-24.04.2-desktop-amd64.iso"
)

# Distros whose ISOs require UEFI boot
NEEDS_UEFI="manjaro"

DISTRO="${1:-}"
if [ -z "$DISTRO" ]; then
  echo "Select a distro:"
  names=($(echo "${!ISOS[@]}" | tr ' ' '\n' | sort))
  for i in "${!names[@]}"; do echo "  $((i+1))) ${names[$i]}"; done
  read -p "> " choice
  DISTRO="${names[$((choice-1))]}"
fi

URL="${ISOS[$DISTRO]:-}"
if [ -z "$URL" ]; then echo "Unknown distro: $DISTRO (available: ${!ISOS[*]})"; exit 1; fi

ISO="$DIR/$(basename "$URL")"
[ -f "$ISO" ] || { echo "Downloading $(basename "$ISO")..."; curl -fL --retry 3 --retry-delay 5 -C - --progress-bar -o "$ISO" "$URL"; }

EXTRA=()
if [[ " $NEEDS_UEFI " == *" $DISTRO "* ]]; then
  EXTRA+=(-bios /usr/share/ovmf/OVMF.fd)
fi

echo "Launching $DISTRO live desktop..."
echo "  Clipboard: install spice-vdagent in guest for copy/paste"

qemu-system-x86_64 \
  -enable-kvm -cpu host -smp 4 -m 20G \
  "${EXTRA[@]}" \
  -cdrom "$ISO" -boot d \
  -vga qxl -spice port=5930,disable-ticketing=on \
  -device virtio-serial -chardev spicevmc,id=vdagent,name=vdagent \
  -device virtserialport,chardev=vdagent,name=com.redhat.spice.0 \
  -usb -device usb-tablet \
  -device virtio-net-pci,netdev=net0,romfile= \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 &

sleep 1
remote-viewer spice://localhost:5930
kill %1 2>/dev/null
