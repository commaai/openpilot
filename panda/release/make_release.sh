#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
export CERT=$DIR/../../xx/pandaextra/certs/release

if [ ! -f "$CERT" ]; then
  echo "No release cert found, cannot build release."
  echo "You probably aren't looking to do this anyway."
  exit
fi

export RELEASE=1
export BUILDER=DEV

cd $DIR/../board
scons -u -c
rm obj/*
scons -u
cd obj
RELEASE_NAME=$(awk '{print $1}' version)
rm panda.bin panda_h7.bin
mv panda.bin.signed panda.bin
mv panda_h7.bin.signed panda_h7.bin
zip -j ../../release/panda-$RELEASE_NAME.zip version panda.bin bootstub.panda.bin panda_h7.bin bootstub.panda_h7.bin
