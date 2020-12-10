#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#CARLA PythonAPI stuff
EGGFILE=carla-0.9.7-py3.5-linux-x86_64.egg
CARLAFILE=CARLA_0.9.7.tar.gz

if [ ! -f "$EGGFILE" ]; then
	echo "Found $EGGFILE not found, checking for $CARLAFILE..."
	if [ ! -f "$CARLAFILE" ]; then
		echo "$CARLAFILE not found, downloading..."
		curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$CARLAFILE
	else
		echo "Found $CARLAFILE, checking integrity..."		
		echo "30d374202eb5c75591af1eff3bd2b38f  $CARLAFILE" | md5sum --check --status
		if [ $? != 0 ]; then
			echo "$CARLAFILE md5 doesn't match, deleting $CARLAFILE and exiting."
			rm -f $CARLAFILE
			exit 1
		fi
	fi
	echo "Extracting CARLA PythonAPI package..."
	mkdir -p $DIR/assets
	cd $DIR/assets
	tar -xvf ../$CARLAFILE PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg --strip-components=3
	echo "05c82fc1203efe9e68910dcf9672bbfe  $EGGFILE" | md5sum --check
	cd $DIR
	rm -f $CARLAFILE
fi


#Docker stuff
cd $DIR/../../

docker pull commaai/openpilot-base:latest
docker build \
  --cache-from commaai/openpilot-sim:latest \
  -t commaai/openpilot-sim:latest \
  -f tools/sim/Dockerfile.sim .
