#!/bin/bash -e
FILE=CARLA_0.9.7.tar.gz
if [ ! -f $FILE ]; then
  curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$FILE
fi
mkdir -p carla
cd carla
tar xvf ../$FILE
easy_install PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg

