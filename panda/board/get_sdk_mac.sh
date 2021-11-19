#!/bin/bash
# Need formula for gcc
sudo easy_install pip
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew tap ArmMbed/homebrew-formulae
brew install python dfu-util arm-none-eabi-gcc
pip install --user libusb1 pycryptodome requests
