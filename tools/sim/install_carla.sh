#!/usr/bin/env bash

cd /tmp
FILE=CARLA_0.9.11.tar.gz
rm -f $FILE
curl -O https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/$FILE

rm -rf carla_tmp
mkdir -p carla_tmp
cd carla_tmp
tar xvf ../$FILE PythonAPI/
easy_install PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg || true

cd ..
rm -rf /tmp/$FILE
rm -rf carla_tmp
