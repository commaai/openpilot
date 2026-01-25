#!/bin/bash
# Setup Bluetooth-enabled kernel for AGNOS/comma 3X
# This script builds and flashes a kernel with Bluetooth support

set -e

AGNOS_BUILDER="/Volumes/agnos/builder"
KERNEL_DIR="$AGNOS_BUILDER/agnos-kernel-sdm845"
DEFCONFIG="$KERNEL_DIR/arch/arm64/configs/tici_defconfig"
SSH_HOST="${SSH_HOST:-comma}"

# Bluetooth kernel configs to add
BT_CONFIGS=(
  "CONFIG_BT=y"
  "CONFIG_BT_BREDR=y"
  "CONFIG_BT_LE=y"
  "CONFIG_BT_RFCOMM=y"
  "CONFIG_BT_RFCOMM_TTY=y"
  "CONFIG_BT_HIDP=y"
  "CONFIG_BT_HCIUART=y"
  "CONFIG_BT_HCIUART_H4=y"
  "CONFIG_BT_HCIUART_QCA=y"
  "CONFIG_BT_QCA=y"
)

echo "=== Bluetooth Kernel Setup for AGNOS ==="
echo ""

# Check if AGNOS builder exists
if [ ! -d "$AGNOS_BUILDER" ]; then
  echo "ERROR: AGNOS builder not found at $AGNOS_BUILDER"
  echo "Mount the AGNOS builder volume first"
  exit 1
fi

# Check if defconfig exists
if [ ! -f "$DEFCONFIG" ]; then
  echo "ERROR: Kernel defconfig not found at $DEFCONFIG"
  echo "Make sure the kernel submodule is initialized:"
  echo "  cd $AGNOS_BUILDER && git submodule update --init agnos-kernel-sdm845"
  exit 1
fi

echo "Step 1: Checking/Adding Bluetooth configs to defconfig..."
for config in "${BT_CONFIGS[@]}"; do
  config_name="${config%%=*}"
  if grep -q "^$config_name=" "$DEFCONFIG" || grep -q "^# $config_name is not set" "$DEFCONFIG"; then
    # Config exists, update it
    sed -i '' "s/^# $config_name is not set/$config/" "$DEFCONFIG" 2>/dev/null || true
    sed -i '' "s/^$config_name=.*/$config/" "$DEFCONFIG" 2>/dev/null || true
  else
    # Config doesn't exist, add it
    echo "$config" >> "$DEFCONFIG"
  fi
  echo "  ✓ $config"
done

echo ""
echo "Step 2: Building kernel (this takes ~5-10 min with ccache)..."
cd "$AGNOS_BUILDER"
./build_kernel.sh

if [ ! -f "$AGNOS_BUILDER/output/boot.img" ]; then
  echo "ERROR: Kernel build failed - boot.img not found"
  exit 1
fi

echo ""
echo "Step 3: Copying boot.img to device..."
scp "$AGNOS_BUILDER/output/boot.img" "$SSH_HOST:/tmp/boot.img"

echo ""
echo "Step 4: Detecting current boot slot..."
BOOT_SLOT=$(ssh "$SSH_HOST" "cat /proc/cmdline | grep -o 'slot_suffix=_[ab]' | cut -d= -f2")
if [ -z "$BOOT_SLOT" ]; then
  BOOT_SLOT="_a"
  echo "  Could not detect slot, defaulting to boot_a"
else
  echo "  Current boot slot: boot$BOOT_SLOT"
fi

echo ""
echo "Step 5: Flashing kernel to boot$BOOT_SLOT partition..."
ssh "$SSH_HOST" "sudo dd if=/tmp/boot.img of=/dev/disk/by-partlabel/boot$BOOT_SLOT bs=4M && sync"

echo ""
echo "Step 6: Rebooting device..."
ssh "$SSH_HOST" "sudo reboot" || true

echo ""
echo "=== Done! ==="
echo "Device is rebooting. After reboot, verify with:"
echo "  ssh $SSH_HOST 'zcat /proc/config.gz | grep CONFIG_BT'"
echo "  ssh $SSH_HOST 'ls -la /dev/ttyHS1'"
echo ""
echo "Then enable BLE in openpilot settings or run:"
echo "  ssh $SSH_HOST 'echo -n 1 > /data/params/d/EnableBLE'"
