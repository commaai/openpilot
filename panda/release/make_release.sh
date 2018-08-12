#!/bin/bash

if [ ! -d "../../pandaextra" ]; then
  echo "No release cert found, cannot build release."
  echo "You probably aren't looking to do this anyway."
  exit
fi

export RELEASE=1

# make ST + bootstub
pushd .
cd ../board
make clean
make obj/panda.bin
make obj/bootstub.panda.bin
popd

# make ESP
pushd .
cd ../boardesp
make clean
make user1.bin
make user2.bin
popd

# make release
mkdir obj
make -f ../common/version.mk
make obj/gitversion.h
RELEASE_NAME=$(python -c "import sys;sys.stdout.write(open('obj/gitversion.h').read().split('\"')[1])")
echo -en $RELEASE_NAME > /tmp/version
rm -rf obj

# make zip file
pushd .
cd ..
zip -j release/panda-$RELEASE_NAME.zip ~/one/panda/board/obj/bootstub.panda.bin ~/one/panda/board/obj/panda.bin ~/one/panda/boardesp/user?.bin ~/one/panda/boardesp/esp-open-sdk/ESP8266_NONOS_SDK_V1.5.4_16_05_20/bin/boot_v1.5.bin /tmp/version
popd

