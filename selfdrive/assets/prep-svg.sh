#!/usr/bin/env bash
set -e

for svg in $(find icons/ images/ -type f | grep svg$); do
  bunx svgo $svg --multipass --pretty --indent 2

  # convert to PNG
  # sudo apt install inkscape
  convert -background none -resize 400% -density 384 $svg "${svg%.svg}.png"
done
