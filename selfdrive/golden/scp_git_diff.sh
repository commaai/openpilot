#!/bin/bash

source ./selfdrive/golden/guess_op_ip.sh

echo $OP_IP

PORT=8022
RSA_FILE=~/.ssh/op.rsa

files=`git diff --name-only`
for file in $files; do
  #echo $file
  set -x
  scp -r -P $PORT -i $RSA_FILE $file root@$OP_IP:/data/openpilot/$file
  set +x
done