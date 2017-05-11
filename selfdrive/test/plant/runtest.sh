#!/bin/bash
pushd ../../controls
./controlsd.py &
pid1=$!
./radard.py &
pid2=$!
./plannerd.py &
pid3=$!
trap "trap - SIGTERM && kill $pid1 && kill $pid2 && kill $pid3" SIGINT SIGTERM EXIT
popd
mkdir -p out
MPLBACKEND=svg ./runtracks.py out
