#!/usr/bin/bash

#TODO: check if we need to open pipenv shell (yes on PC, no on EON)
#TODO: ensure in correct dir
scons -j$(nproc)
