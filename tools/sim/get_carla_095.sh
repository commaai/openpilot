#!/bin/bash -e

FILE=CARLA_0.9.5.tar.gz
if [ ! -f $FILE ]; then
  curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$FILE
fi
if [ ! -d carla ]; then
  rm -rf carla_tmp
  mkdir -p carla_tmp
  cd carla_tmp
  tar xvf ../$FILE
  cd ../
  mv carla_tmp carla
fi
