#!/bin/bash

if [ -z $BASEDIR ]; then
  export BASEDIR="./extra/datasets/"
fi

cd $BASEDIR
if [ -d "kits19" ]; then
  echo "kits19 dataset is already available"
else
  echo "Downloading and preparing kits19 dataset at $BASEDIR"

  git clone https://github.com/neheller/kits19
  cd kits19
  pip3 install -r requirements.txt
  python3 -m starter_code.get_imaging

  echo "Done"
fi
