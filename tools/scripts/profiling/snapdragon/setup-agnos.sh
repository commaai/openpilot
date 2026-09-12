#!/usr/bin/env bash
set -e

# TODO: there's probably a better way to do this

cd SnapdragonProfiler/service
mv android real_android
ln -s agl/ android
