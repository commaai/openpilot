#!/bin/bash

# from http://www.esp8266.com/wiki/doku.php?id=setup-osx-compiler-esp8266

brew install gnu-sed --with-default-names
brew tap homebrew/dupes
brew install gperf
brew install grep
brew install autoconf
brew install binutils
brew install gawk
brew install wget
brew install automake
brew install libtool
brew install help2man

brew uninstall gperf

hdiutil create esp-open-sdk.dmg -volname "esp-open-sdk" -size 10g -fs "Case-sensitive HFS+"
hdiutil mount esp-open-sdk.dmg
ln -s /Volumes/esp-open-sdk esp-open-sdk
cd esp-open-sdk

git init
git remote add origin https://github.com/pfalcon/esp-open-sdk.git
git fetch origin
git checkout 03f5e898a059451ec5f3de30e7feff30455f7cec
git submodule init
git submodule update --recursive

make STANDALONE=y

