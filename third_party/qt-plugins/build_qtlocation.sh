#!/usr/bin/env sh

# Qtlocation plugin with extra fields parsed from api response
cd /tmp
git clone https://github.com/commaai/qtlocation.git
cd qtlocation
qmake
make -j$(nproc)
