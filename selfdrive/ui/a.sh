#!/usr/bin/env bash

cd ../..
scons -j40|| { echo 'my_command failed' ; exit 1; }
cd selfdrive/ui
./ui  
