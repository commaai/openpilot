#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# sudo apt install inkscape

for svg in $(find $DIR -type f | grep svg$); do
  bunx svgo $svg --multipass --pretty --indent 2

  # convert to PNG
  png="${svg%.svg}.png"
  width=$(inkscape --query-width "$svg")
  height=$(inkscape --query-height "$svg")
  if (( $(echo "$width > $height" | bc -l) )); then
    export_dim="--export-width=512"
  else
    export_dim="--export-height=512"
  fi
  inkscape "$svg" --export-filename="$png" $export_dim

  optipng -o7 -strip all "$png"
done
