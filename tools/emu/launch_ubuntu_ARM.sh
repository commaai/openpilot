#!/usr/bin/env bash

set -eu

qemu_disk_image_option=""

if [ $# -ge 1 ] && [ ! -z "$1" ]; then
  if [ "$1" != "" -a "$1" != "-h" -a "$1" != "--help" -a "$1" != "-display" -a "$1" != "-vmpath" -a "$1" != "-temp" ]; then
    echo "launch_ubuntu_ARM.sh: invalid option $1"
    echo "Try ./launch_ubuntu_ARM.sh --help"
    exit 1
  fi

  if [ "$1" == "-h" -o "$1" == "--help" ]; then
    echo "Usage: ./launch_ubuntu_ARM.sh [-display [none|vga] -vmpath [path to store VM files] -temp]"
    echo ""
    echo "Optional arguments:"
    echo "  -display    specifies the display mode. Possible values are: none (default),"
    echo "              vga (requires a VNC connection)."
    echo "  -vmpath     specifies the path where .img and qcow2 files will be stored. Current working dir by default"
    echo "  -temp       tells qemu to do not persist the changes in the snapshot image"
    exit 0
  fi

  if [ "$1" == "-temp" ]; then
    qemu_disk_image_option="-snapshot"
  elif [ $# -eq 3 ] && [ ! -z "$3" ]; then
    if [ "$3" == "-temp" ]; then
      qemu_disk_image_option="-snapshot"
    fi
  elif [ $# -eq 5 ] && [ ! -z "$5" ]; then
    if [ "$5" == "-temp" ]; then
      qemu_disk_image_option="-snapshot"
    fi
  fi

  if [ "$1" == "-display" ]; then
    if [ "$2" != "none" -a "$2" != "vga" ]; then
      echo "launch_ubuntu_ARM.sh: invalid option $1"
      echo "Try ./launch_ubuntu_ARM.sh --help"
      exit 1
    else
      display="$2"
    fi

    if [ $# -ge 4 ] && [ ! -z "$3" ]; then
      if [ "$3" == "-vmpath" ]; then
        if [ -z "$4" ]; then
          echo "launch_ubuntu_ARM.sh: invalid option $1"
          echo "Try ./launch_ubuntu_ARM.sh --help"
          exit 1
        else
          install_dir="$4"
        fi
      else
        install_dir="$(pwd)/"
      fi
    else
      install_dir="$(pwd)/"
    fi
  elif [ "$1" == "-vmpath" ]; then
    if [ -z "$2" ]; then
      echo "launch_ubuntu_ARM.sh: invalid option $1"
      echo "Try ./launch_ubuntu_ARM.sh --help"
      exit 1
    else
      install_dir="$2"
    fi

    if [ $# -ge 4 ] && [ ! -z "$3" ]; then
      if [ "$3" == "-display" ]; then
        if [ -z "$4" ]; then
          echo "launch_ubuntu_ARM.sh: invalid option $1"
          echo "Try ./launch_ubuntu_ARM.sh --help"
          exit 1
        else
          display="$4"
        fi
      else
        display="none"
      fi
    else
      display="none"
    fi
  else
    display="none"
    install_dir="$(pwd)/"
  fi
else
  display="none"
  install_dir="$(pwd)/"
fi

echo ""
echo -e "\e[1;36m Current settings... \e[0m"
echo ""
echo -e "\e[1;32m Display mode: $display \e[0m"
echo -e "\e[1;32m VM Installation dir: $install_dir \e[0m"
echo ""

id=ubuntu-16.04.6-server-arm64   
img="${install_dir}${id}.img.qcow2"
img_snapshot="${install_dir}${id}.img.snapshot.qcow2"
flash0="${install_dir}${id}-flash0.img"
flash1="${install_dir}${id}-flash1.img"

if [ $display == "vga" ]; then
  qemu_display_options_1="-device cirrus-vga"
  qemu_display_options_2="-vga std"
else
  qemu_display_options_1="-nographic"
  qemu_display_options_2=""
fi

qemu-system-aarch64_5.0 \
  -cpu cortex-a57 \
  -device virtio-blk-device,drive=hd0 \
  -drive "if=none,file=${img_snapshot},id=hd0" $qemu_disk_image_option \
  -m 4G \
  -machine virt \
  -pflash "$flash0" \
  -pflash "$flash1" \
  -smp 4 \
  -device virtio-net-device,netdev=hostnet0,mac=3c:a0:67:3e:fe:72 \
  -netdev user,id=hostnet0,hostfwd=tcp::2222-:22 \
  -device usb-ehci -device usb-kbd -device usb-mouse -usb \
  -device usb-tablet \
  $qemu_display_options_1 \
  $qemu_display_options_2 \
;
