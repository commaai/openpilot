#!/usr/bin/env bash
sudo ifconfig can0 up
make
./cantest
