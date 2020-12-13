#!/bin/bash

cd /openpilot/selfdrive/modeld/

#wait a bit till everything gets ready
sleep 10

while true; do
	sleep 3
	PROC=$(ps -A | grep -w _modeld)
	if ! [ -n "$_modeld" ] ; then
	    ./modeld
	fi
done
