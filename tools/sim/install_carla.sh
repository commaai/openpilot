#!/usr/bin/env bash

cd /tmp
FILE=CARLA_0.9.7.tar.gz
rm -f $FILE
curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$FILE

rm -rf carla_tmp
mkdir -p carla_tmp
cd carla_tmp
tar xvf ../$FILE
easy_install PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg || true
cd ..
rm -rf /tmp/$FILE
rm -rf carla_tmp
