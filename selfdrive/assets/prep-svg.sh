#!/usr/bin/env bash
set -e

# sudo apt install scour

for svg in $(find icons/ -type f | grep svg$); do
  # scour doesn't support overwriting input file
  scour $svg --remove-metadata $svg.tmp
  mv $svg.tmp $svg

  # convert to PNG
  convert -background none -resize 400% -density 384 $svg "${svg%.svg}.png"
done
