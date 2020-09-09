#!/usr/bin/env bash
set -eu

if [ $# -ge 1 ] && [ ! -z "$1" ]; then
  if [ "$1" != "" -a "$1" != "-h" -a "$1" != "--help" -a "$1" != "-vmpath" ]; then
    echo "install_ubuntu_ARM.sh: invalid option $1"
    echo "Try ./install_ubuntu_ARM.sh --help"
    exit 1
  fi

  if [ "$1" == "-h" -o "$1" == "--help" ]; then
    echo "Usage: ./install_ubuntu_ARM.sh [-vmpath [path to store VM files]]"
    echo ""
    echo "Optional arguments:"
    echo "  -vmpath     specifies the path where .img and qcow2 files will be stored. Current working dir by default"
    exit 0
  fi

  if [ "$1" == "-vmpath" ]; then
    if [ -z "$2" ]; then
      echo "install_ubuntu_ARM.sh: invalid option $1"
      echo "Try ./install_ubuntu_ARM.sh --help"
      exit 1
    else
      install_dir="$2"
    fi
  else
    install_dir="$(pwd)/"
  fi
else
  install_dir="$(pwd)/"
fi

echo ""
echo -e "\e[1;36m Current settings... \e[0m"
echo ""
echo -e "\e[1;32m VM Installation dir: $install_dir \e[0m"
echo ""

id=ubuntu-16.04.6-server-arm64   
img="${install_dir}${id}.img.qcow2"
img_snapshot="${install_dir}${id}.img.snapshot.qcow2"
iso="${install_dir}${id}.iso"
flash0="${install_dir}${id}-flash0.img"
flash1="${install_dir}${id}-flash1.img"

#
# 1 - Get to OS iso image.
#
# http://cdimage.ubuntu.com/ubuntu/releases/16.04/release/ubuntu-16.04.6-server-arm64.iso
#

if [ ! -f "$iso" ]; then
  wget "http://cdimage.ubuntu.com/ubuntu/releases/16.04/release/${id}.iso"
fi


#
# 2 - create a blank disc image
#
if [ ! -f "$img" ]; then
  qemu-img create -f qcow2 "$img" 40G
fi

#
# 3 - create snapshot image
#
if [ ! -f "$img_snapshot" ]; then
  qemu-img \
    create \
    -b "$img" \
    -f qcow2 \
    "$img_snapshot" \
  ;
fi

#
# 4 - Assemble flash images
#
if [ ! -f "$flash0" ]; then
  dd if=/dev/zero of="$flash0" bs=1M count=64
  dd if=/usr/share/qemu-efi/QEMU_EFI.fd of="$flash0" conv=notrunc
fi
if [ ! -f "$flash1" ]; then
  dd if=/dev/zero of="$flash1" bs=1M count=64
fi

#
# 5 - Invoke qemu with a cdrom mounted
#
qemu-system-aarch64_5.0 \
  -cpu cortex-a57 \
  -device virtio-scsi-device \
  -device scsi-cd,drive=cdrom \
  -device virtio-blk-device,drive=hd0 \
  -drive "file=${iso},id=cdrom,if=none,media=cdrom" \
  -drive "if=none,file=${img_snapshot},id=hd0" \
  -m 4G \
  -machine virt \
  -nographic \
  -pflash "$flash0" \
  -pflash "$flash1" \
  -smp 4 \
  -device virtio-net-device,netdev=hostnet0,mac=3c:a0:67:3e:fe:72 \
  -netdev user,id=hostnet0,hostfwd=tcp::2222-:22 \
;
