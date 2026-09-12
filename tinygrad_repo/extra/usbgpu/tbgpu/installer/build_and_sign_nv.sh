#!/bin/bash
set -e

xcodebuild clean build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO -alltargets -configuration Release build

cp "../profiles/edriver_rel_2.provisionprofile" "./build/Release/TinyGPU.app/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext/embedded.provisionprofile"
cp "../profiles/installer_provisioning.provisionprofile" "./build/Release/TinyGPU.app/Contents/embedded.provisionprofile"

codesign \
    --sign "Developer ID Application: tinygrad, Corp. (9YG3G8543N)" \
    --entitlements ./TinyGPUDriverExtension/TinyGPUDriver.NV.Release.entitlements \
    --verbose \
    --options runtime \
    --timestamp \
    --force \
    ./build/Release/TinyGPU.app/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext

codesign \
    --sign "Developer ID Application: tinygrad, Corp. (9YG3G8543N)" \
    --entitlements ./macOS/macOS.entitlements \
    --options runtime \
    --verbose \
    --timestamp \
    --force \
    ./build/Release/TinyGPU.app

codesign --verify --deep --strict --verbose=4 ./build/Release/TinyGPU.app/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext

codesign --verify --deep --strict --verbose=4 ./build/Release/TinyGPU.app

spctl -a -vv ./build/Release/TinyGPU.app

spctl -a -vv ./build/Release/TinyGPU.app/Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext
