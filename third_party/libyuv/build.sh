#!/usr/bin/env bash
set -e

git clone https://chromium.googlesource.com/libyuv/libyuv
cd libyuv
git reset --hard 4a14cb2e81235ecd656e799aecaaf139db8ce4a2
cmake .

## To create universal binary on Darwin:
## ```
## lipo -create -output Darwin/libyuv.a path-to-x64/libyuv.a path-to-arm64/libyuv.a
## ```