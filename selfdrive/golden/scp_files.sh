#!/bin/bash

source ./selfdrive/golden/guess_op_ip.sh

echo $OP_IP

PORT=8022
RSA_FILE=~/.ssh/op.rsa

TARGET=$1
if [ "$2" ]; then
    TARGET=$2
fi

set -x

scp -r -P $PORT -i $RSA_FILE $1 root@$OP_IP:/data/openpilot/$TARGET