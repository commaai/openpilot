#!/bin/bash

# TODO: there's probably a better way to do this

cd SnapdragonProfiler/service
mv android real_android
ln -s iot_rb5_lu/ android
