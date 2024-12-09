#!/bin/sh
cd ..
while :; do
  ./camerad &
  pid="$!"
  sleep 2
  kill -2 $pid
  wait $pid
done
