#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

for svg in $(find $DIR -type f | grep svg$); do
  bunx svgo $svg --multipass --pretty --indent 2

  # convert to PNG
  # sudo apt install inkscape
  convert -background none -density 384 "$svg" -resize 512x512 "${svg%.svg}.png"
done
