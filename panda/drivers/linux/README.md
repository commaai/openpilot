# Linux driver
Installs the panda linux kernel driver using DKMS.

This will allow the panda to work with tools such as `can-utils`

## Prerequisites
 - `apt-get install dkms gcc linux-headers-$(uname -r) make sudo`

## Installation
 - `make all`
 - `make link` (optional, setup to build/install when kernel is updated)

## Uninstall
 - `make clean`

## Usage

You will need to bring it up using `sudo ifconfig can0 up` or
`sudo ip link set dev can0 up`, depending on your platform.

Note that you may have to setup udev rules for Linux
``` bash
sudo tee /etc/udev/rules.d/11-panda.rules <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddee", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```
