#!/usr/bin/env bash

# sudo apt install scour

for svg in $(find icons/ -type f | grep svg$); do
  # scour doesn't support overwriting input file
  scour $svg --remove-metadata $svg.tmp
  mv $svg.tmp $svg
done
