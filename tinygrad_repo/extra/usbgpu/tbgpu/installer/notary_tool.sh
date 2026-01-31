#!/bin/bash
set -e

ditto -c -k --keepParent ./build/Release/TinyGPU.app ./build/Release/TinyGPU.zip
xcrun notarytool submit ./build/Release/TinyGPU.zip --keychain-profile "hgwJFhdheiIEy82nDN" --wait
