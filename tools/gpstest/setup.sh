#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d LimeGPS ]; then
    git clone https://github.com/osqzss/LimeGPS.git
    cd LimeGPS
    sed -i 's/LimeSuite/LimeSuite -L..\/lib/' makefile
    make
    cd ..
fi
