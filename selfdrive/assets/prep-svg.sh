#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

for svg in $(find $DIR -type f | grep svg$); do
  bunx svgo $svg --multipass --pretty --indent 2

  # convert to PNG
  # sudo apt install inkscape
  convert -background none -resize 400% -density 384 $svg "${svg%.svg}.png"
done
