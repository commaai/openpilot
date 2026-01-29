#!/usr/bin/env bash
set -e

# AGNOS Build Script for Bluetooth-enabled kernel
# Works locally (macOS with Docker) and in GitHub Actions
#
# Prerequisites:
#   - Docker (OrbStack on macOS recommended)
#   - For local macOS: Case-sensitive APFS volume at /Volumes/agnos
#   - For upload: R2 credentials in environment or .env file
#
# Usage:
#   AGNOS_VERSION=16-bt2 ./scripts/build-agnos.sh --system           # Build kernel + system
#   AGNOS_VERSION=16-bt2 ./scripts/build-agnos.sh --system --upload  # Build and upload to R2
#   AGNOS_VERSION=16-bt2 ./scripts/build-agnos.sh --upload-only      # Upload existing build to R2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration - AGNOS_VERSION must be provided
if [ -z "$AGNOS_VERSION" ]; then
  echo "Error: AGNOS_VERSION must be set"
  echo "Usage: AGNOS_VERSION=16-bt2 $0 [options]"
  exit 1
fi
BASE_AGNOS_VERSION="${AGNOS_VERSION%%-*}"  # Extract base version (16 from 16-bt2)

# Determine build directory
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS: Use case-sensitive volume
  BUILD_DIR="/Volumes/agnos/builder"
  if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: AGNOS builder not found at $BUILD_DIR"
    echo "Please set up a case-sensitive APFS volume:"
    echo "  diskutil apfs addVolume disk3 \"Case-sensitive APFS\" agnos"
    echo "  cd /Volumes/agnos"
    echo "  git clone https://github.com/commaai/agnos-builder.git builder"
    echo "  cd builder && git submodule update --init"
    exit 1
  fi
else
  # Linux/CI: Use workspace directory
  BUILD_DIR="${AGNOS_BUILD_DIR:-$REPO_ROOT/agnos-builder}"
  if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning agnos-builder..."
    git clone --depth 1 https://github.com/commaai/agnos-builder.git "$BUILD_DIR"
    cd "$BUILD_DIR"
    git submodule update --init agnos-kernel-sdm845
  fi
fi

KERNEL_DIR="$BUILD_DIR/agnos-kernel-sdm845"
OUTPUT_DIR="$BUILD_DIR/output"

echo "=== AGNOS Build Configuration ==="
echo "Version: $AGNOS_VERSION"
echo "Base AGNOS: $BASE_AGNOS_VERSION"
echo "Build dir: $BUILD_DIR"
echo "================================="

# Step 1: Apply Bluetooth kernel config
apply_bt_kernel_config() {
  echo ""
  echo "=== Applying Bluetooth kernel configuration ==="

  DEFCONFIG="$KERNEL_DIR/arch/arm64/configs/tici_defconfig"

  # Check if BT is already enabled
  if grep -q "CONFIG_BT=y" "$DEFCONFIG"; then
    echo "Bluetooth already enabled in kernel config"
  else
    echo "Adding Bluetooth options to tici_defconfig..."
    cat >> "$DEFCONFIG" << 'EOF'

# Bluetooth support (added by build-agnos.sh)
CONFIG_BT=y
CONFIG_BT_BREDR=y
CONFIG_BT_RFCOMM=y
CONFIG_BT_BNEP=y
CONFIG_BT_HIDP=y
CONFIG_BT_HS=y
CONFIG_BT_LE=y
CONFIG_BT_QCA=y
CONFIG_BT_HCIUART=y
CONFIG_BT_HCIUART_QCA=y
CONFIG_MSM_BT_POWER=y
CONFIG_BTFM_SLIM=y
CONFIG_BTFM_SLIM_WCN3990=y
EOF
  fi

  # Increase MAX_PATCH_FILE_SIZE for WCN3990 firmware (170KB+ TLV files)
  BTQCA="$KERNEL_DIR/drivers/bluetooth/btqca.c"
  if grep -q "MAX_PATCH_FILE_SIZE (100\*1024)" "$BTQCA"; then
    echo "Increasing MAX_PATCH_FILE_SIZE to 256KB..."
    sed -i.bak 's/MAX_PATCH_FILE_SIZE (100\*1024)/MAX_PATCH_FILE_SIZE (256*1024)/' "$BTQCA"
  fi

  # Enable BT UART in device tree
  DTSI="$KERNEL_DIR/arch/arm64/boot/dts/qcom/comma_common.dtsi"
  if grep -q "qupv3_se6_4uart" "$DTSI" && ! grep -q 'status = "ok"' "$DTSI"; then
    echo "Enabling Bluetooth UART in device tree..."
    # This assumes the dtsi has the qupv3_se6_4uart entry but status is disabled
    sed -i.bak 's/&qupv3_se6_4uart {/&qupv3_se6_4uart {\n\tstatus = "ok";/' "$DTSI"
  elif ! grep -q "qupv3_se6_4uart" "$DTSI"; then
    echo "Adding Bluetooth UART to device tree..."
    cat >> "$DTSI" << 'EOF'

/* Bluetooth UART - SE6 at 0x898000 */
&qupv3_se6_4uart {
	status = "ok";
};
EOF
  else
    echo "Bluetooth UART already configured in device tree"
  fi
}

# Step 1.5: Ensure bluez is in base_setup.sh for system image
ensure_bluez_in_system() {
  echo ""
  echo "=== Ensuring bluez is in system image ==="

  BASE_SETUP="$BUILD_DIR/userspace/base_setup.sh"
  if ! grep -q "bluez" "$BASE_SETUP"; then
    echo "Adding bluez to base_setup.sh..."
    # Add bluez to the apt-fast install list (after wireless-tools)
    sed -i.bak 's/wireless-tools \\/wireless-tools \\\n    bluez \\/' "$BASE_SETUP"
  else
    echo "bluez already in base_setup.sh"
  fi
}

# Step 1.6: Install D-Bus policy for BLE GATT server
install_ble_dbus_policy() {
  echo ""
  echo "=== Installing BLE D-Bus policy ==="

  DBUS_DIR="$BUILD_DIR/userspace/files"
  POLICY_FILE="$DBUS_DIR/comma-ble.conf"

  if [ -f "$POLICY_FILE" ]; then
    echo "BLE D-Bus policy already exists"
    return
  fi

  cat > "$POLICY_FILE" << 'DBUSEOF'
<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-BUS Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <policy context="default">
    <allow send_interface="org.bluez.GattCharacteristic1"/>
    <allow send_interface="org.bluez.GattDescriptor1"/>
    <allow send_interface="org.bluez.GattService1"/>
    <allow send_interface="org.bluez.LEAdvertisement1"/>
    <allow send_interface="org.freedesktop.DBus.ObjectManager"/>
    <allow send_interface="org.freedesktop.DBus.Properties"/>
    <allow send_interface="org.freedesktop.DBus.Introspectable"/>
    <allow receive_interface="org.bluez.GattCharacteristic1"/>
    <allow receive_interface="org.bluez.GattDescriptor1"/>
    <allow receive_interface="org.bluez.GattService1"/>
    <allow receive_interface="org.bluez.LEAdvertisement1"/>
    <allow receive_interface="org.freedesktop.DBus.ObjectManager"/>
    <allow receive_interface="org.freedesktop.DBus.Properties"/>
    <allow receive_interface="org.freedesktop.DBus.Introspectable"/>
  </policy>
</busconfig>
DBUSEOF

  echo "BLE D-Bus policy created at $POLICY_FILE"
  echo "NOTE: This file must be copied to /etc/dbus-1/system.d/ in the system image"
  echo "Add to Dockerfile.agnos or hardware_setup.sh: COPY ./userspace/files/comma-ble.conf /etc/dbus-1/system.d/"
}

# Step 1.7: Extract and install QCA Bluetooth firmware from bluetooth.img
install_bt_firmware() {
  echo ""
  echo "=== Installing QCA Bluetooth firmware ==="

  BT_IMG="$BUILD_DIR/firmware/bluetooth.img"
  FW_DIR="$BUILD_DIR/userspace/files/lib/firmware/qca"

  if [ ! -f "$BT_IMG" ]; then
    echo "Warning: bluetooth.img not found, skipping firmware install"
    return
  fi

  mkdir -p "$FW_DIR"

  # Extract CRBTFW21.TLV and CRNV21.BIN from bluetooth.img FAT16 partition
  # and install as rampatch_02140201.bin / nvm_02140201.bin for the kernel driver
  BT_IMG="$BT_IMG" FW_DIR="$FW_DIR" python3 << 'PYEOF'
import struct, os

bt_img = os.environ["BT_IMG"]
fw_dir = os.environ["FW_DIR"]

with open(bt_img, 'rb') as f:
    data = f.read()

sector_size = struct.unpack('<H', data[11:13])[0]
sectors_per_cluster = data[13]
reserved_sectors = struct.unpack('<H', data[14:16])[0]
num_fats = data[16]
root_entries = struct.unpack('<H', data[17:19])[0]
sectors_per_fat = struct.unpack('<H', data[22:24])[0]
cluster_size = sector_size * sectors_per_cluster
root_dir_offset = (reserved_sectors + num_fats * sectors_per_fat) * sector_size
root_dir_sectors = (root_entries * 32 + sector_size - 1) // sector_size
data_start = root_dir_offset + root_dir_sectors * sector_size

def read_fat():
    fat_offset = reserved_sectors * sector_size
    return {i: struct.unpack('<H', data[fat_offset+i*2:fat_offset+i*2+2])[0]
            for i in range(sectors_per_fat * sector_size // 2)}

def cluster_to_offset(c):
    return data_start + (c - 2) * cluster_size

def read_file(start_cluster, size, fat):
    result, cluster, remaining = b'', start_cluster, size
    while remaining > 0 and cluster < 0xFFF8:
        offset = cluster_to_offset(cluster)
        chunk = min(remaining, cluster_size)
        result += data[offset:offset+chunk]
        remaining -= chunk
        cluster = fat[cluster]
    return result

fat = read_fat()

# Map of source filename (uppercase) -> dest filename
FILE_MAP = {
    'CRBTFW21.TLV': 'rampatch_02140201.bin',
    'CRNV21.BIN': 'nvm_02140201.bin',
}

for i in range(root_entries):
    offset = root_dir_offset + i * 32
    entry = data[offset:offset+32]
    if entry[0] == 0x00: break
    if entry[0] == 0xE5 or entry[11] == 0x0F: continue
    name = entry[:8].rstrip(b' ').decode('ascii', errors='ignore')
    if entry[11] & 0x10 and name == 'IMAGE':
        cluster = struct.unpack('<H', entry[26:28])[0]
        dir_data = read_file(cluster, cluster_size * 4, fat)
        for j in range(len(dir_data) // 32):
            e2 = dir_data[j*32:j*32+32]
            if e2[0] == 0x00: break
            if e2[0] == 0xE5 or e2[11] == 0x0F: continue
            n = e2[:8].rstrip(b' ').decode('ascii', errors='ignore')
            ext = e2[8:11].rstrip(b' ').decode('ascii', errors='ignore')
            c = struct.unpack('<H', e2[26:28])[0]
            s = struct.unpack('<I', e2[28:32])[0]
            full = f'{n}.{ext}' if ext else n
            if full in FILE_MAP and s > 0:
                file_data = read_file(c, s, fat)
                dest = os.path.join(fw_dir, FILE_MAP[full])
                with open(dest, 'wb') as f2:
                    f2.write(file_data)
                print(f'  {full} -> {FILE_MAP[full]} ({s} bytes)')
PYEOF

  echo "Firmware installed to $FW_DIR"
  ls -la "$FW_DIR"
}

# Step 1.6: Patch package_ota.py to skip userdata partitions we don't build
patch_package_ota() {
  echo ""
  echo "=== Patching package_ota.py ==="

  PACKAGE_SCRIPT="$BUILD_DIR/scripts/package_ota.py"

  # Comment out userdata partitions (we don't build those)
  if grep -q "^  Partition('userdata" "$PACKAGE_SCRIPT"; then
    echo "Commenting out userdata partitions..."
    sed -i.bak \
      -e "s|^  Partition('userdata_90'|  # Partition('userdata_90'|" \
      -e "s|^  Partition('userdata_89'|  # Partition('userdata_89'|" \
      -e "s|^  Partition('userdata_30'|  # Partition('userdata_30'|" \
      "$PACKAGE_SCRIPT"
  else
    echo "userdata partitions already commented out"
  fi
}

# Step 2: Update VERSION file
update_version() {
  echo ""
  echo "=== Updating VERSION to $AGNOS_VERSION ==="
  echo "$AGNOS_VERSION" > "$BUILD_DIR/VERSION"
}

# Step 3: Build kernel
build_kernel() {
  echo ""
  echo "=== Building kernel ==="
  cd "$BUILD_DIR"
  ./build_kernel.sh

  if [ ! -f "$OUTPUT_DIR/boot.img" ]; then
    echo "Error: boot.img not found after build"
    exit 1
  fi

  echo "Kernel built: $OUTPUT_DIR/boot.img"
  ls -la "$OUTPUT_DIR/boot.img"
}

# Step 4: Build system (optional, takes longer)
build_system() {
  echo ""
  echo "=== Building system image ==="
  cd "$BUILD_DIR"
  GIT_HASH=$(git rev-parse HEAD) ./build_system.sh

  if [ ! -f "$OUTPUT_DIR/system.img" ]; then
    echo "Error: system.img not found after build"
    exit 1
  fi

  echo "System built: $OUTPUT_DIR/system.img"
  ls -la "$OUTPUT_DIR/system.img"
}

# Step 5: Package OTA
package_ota() {
  echo ""
  echo "=== Packaging OTA files ==="
  cd "$BUILD_DIR"

  # Set our R2 URL as the update URL
  export AGNOS_UPDATE_URL="${R2_PUBLIC_URL:-https://pub-0e2c2429a38c4224bf993e0f6773839b.r2.dev}"
  export AGNOS_STAGING_UPDATE_URL="$AGNOS_UPDATE_URL"

  # Ensure package_ota.py is patched
  patch_package_ota

  python3 scripts/package_ota.py

  echo "OTA files created in $OUTPUT_DIR/ota/"
  ls -la "$OUTPUT_DIR/ota/"
}

# Step 6: Upload to R2
upload_to_r2() {
  echo ""
  echo "=== Uploading to Cloudflare R2 ==="

  # Load credentials from .env if available (uses AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
  if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
  fi

  # R2 config - uses same AWS_* env vars as infra
  R2_BUCKET="${R2_BUCKET:-asius-agnos}"
  R2_ENDPOINT="${R2_ENDPOINT:-https://558df022e422781a34f239d7de72c8ae.r2.cloudflarestorage.com}"
  R2_PUBLIC_URL="${R2_PUBLIC_URL:-https://pub-0e2c2429a38c4224bf993e0f6773839b.r2.dev}"

  if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: R2 credentials not set. Required env vars:"
    echo "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
    echo ""
    echo "These are the same credentials used for Pulumi infra deployment."
    exit 1
  fi

  OTA_DIR="$OUTPUT_DIR/ota"

  echo "Uploading OTA files to R2..."
  for file in "$OTA_DIR"/*.xz "$OTA_DIR"/*.img "$OTA_DIR"/*.json; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      echo "  Uploading $filename..."
      aws s3 cp "$file" "s3://$R2_BUCKET/$filename" \
        --endpoint-url "$R2_ENDPOINT" \
        --no-progress
    fi
  done

  echo ""
  echo "Upload complete!"
  echo "OTA manifest available at: $R2_PUBLIC_URL/ota.json"
}

# Step 7: Generate agnos.json for openpilot
generate_manifest() {
  echo ""
  echo "=== Generating agnos.json for openpilot ==="

  OTA_JSON="$OUTPUT_DIR/ota/ota.json"
  DEST="$REPO_ROOT/system/hardware/tici/agnos.json"

  if [ -f "$OTA_JSON" ]; then
    cp "$OTA_JSON" "$DEST"
    echo "Updated: $DEST"
  else
    echo "Warning: OTA manifest not found at $OTA_JSON"
  fi
}

# Main
main() {
  UPLOAD=false
  UPLOAD_ONLY=false
  BUILD_SYSTEM=false

  for arg in "$@"; do
    case $arg in
      --upload)
        UPLOAD=true
        ;;
      --upload-only)
        UPLOAD_ONLY=true
        ;;
      --system)
        BUILD_SYSTEM=true
        ;;
      --help)
        echo "Usage: AGNOS_VERSION=<version> $0 [options]"
        echo ""
        echo "Environment:"
        echo "  AGNOS_VERSION   Required. Version string (e.g., 16-bt2)"
        echo ""
        echo "Options:"
        echo "  --system       Build system image (required for full OTA)"
        echo "  --upload       Upload to R2 after building"
        echo "  --upload-only  Only upload existing build (skip build steps)"
        echo "  --help         Show this help"
        echo ""
        echo "Example:"
        echo "  AGNOS_VERSION=16-bt2 $0 --system --upload"
        exit 0
        ;;
    esac
  done

  if [ "$UPLOAD_ONLY" = true ]; then
    upload_to_r2
    echo ""
    echo "=== Upload complete ==="
    exit 0
  fi

  apply_bt_kernel_config
  ensure_bluez_in_system
  install_ble_dbus_policy
  install_bt_firmware
  update_version
  build_kernel

  if [ "$BUILD_SYSTEM" = true ]; then
    build_system
    package_ota
    generate_manifest
  fi

  if [ "$UPLOAD" = true ]; then
    upload_to_r2
  fi

  echo ""
  echo "=== Build complete ==="
  echo "Output files:"
  ls -la "$OUTPUT_DIR/"
}

main "$@"
