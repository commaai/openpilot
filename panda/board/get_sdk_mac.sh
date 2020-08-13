#!/bin/bash
# Need formula for gcc
sudo easy_install pip
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew tap ArmMbed/homebrew-formulae
brew install python dfu-util arm-none-eabi-gcc
pip install --user libusb1 pycryptodome requests
