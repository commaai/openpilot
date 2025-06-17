#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
export CERT=/home/batman/xx/pandaextra/certs/release

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
zip -j ../../release/panda-$RELEASE_NAME.zip version panda.bin.signed bootstub.panda.bin panda_h7.bin.signed bootstub.panda_h7.bin
