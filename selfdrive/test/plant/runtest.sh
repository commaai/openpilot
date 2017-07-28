#!/bin/bash

export OPTEST=1
export OLD_CAN=1

pushd ../../controls
./controlsd.py &
pid1=$!
./radard.py &
pid2=$!
trap "trap - SIGTERM && kill $pid1 && kill $pid2" SIGINT SIGTERM EXIT
popd
mkdir -p out
MPLBACKEND=svg ./runtracks.py out
