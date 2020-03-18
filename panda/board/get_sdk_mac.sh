#!/bin/bash
# Need formula for gcc
brew tap ArmMbed/homebrew-formulae
brew install python dfu-util arm-none-eabi-gcc
pip install --user libusb1 pycrypto requests
