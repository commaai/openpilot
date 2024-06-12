Welcome to the jungle
======

Firmware for the Panda Jungle testing board.
Available for purchase at the [comma shop](https://comma.ai/shop/panda-jungle).

## udev rules

To make the jungle usable without root permissions, you might need to setup udev rules for it.
On ubuntu, this should do the trick:
``` bash
sudo tee /etc/udev/rules.d/12-panda_jungle.rules <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## updating the firmware
Updating the firmware is easy! In the `board/jungle/` folder, run:
``` bash
./flash.py
```

If you somehow bricked your jungle, you'll need a [comma key](https://comma.ai/shop/products/comma-key) to put the microcontroller in DFU mode for the V1.
For V2, the onboard button serves this purpose. When powered on while holding the button to put it in DFU mode, running `./recover.sh` in `board/` should unbrick it.
