#!/bin/bash
cd ../board
make clean

while true; do
  make ota
  sleep 10
done

