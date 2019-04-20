#!/bin/sh
finish() {
  echo "exiting orbd"
  pkill -SIGINT -P $$
}

trap finish EXIT

while true; do
  ./orbd &
  wait $!
done

