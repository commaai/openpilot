#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ICONS_DIR="$DIR/icons"
BOOTSTRAP_SVG="$DIR/../../third_party/bootstrap/bootstrap-icons.svg"

ICON_IDS=(
  arrow-down
  arrow-right
  backspace
  capslock
  shift
)
ICON_FILL_COLOR="#fff"

# extract bootstrap icons
for id in "${ICON_IDS[@]}"; do
  svg="${ICONS_DIR}/${id}.svg"
  perl -0777 -ne "print \$& if /<symbol[^>]*id=\"$id\"[^>]*>.*?<\/symbol>/s" "$BOOTSTRAP_SVG" \
  | sed "s/<symbol/<svg fill=\"$ICON_FILL_COLOR\"/; s/<\/symbol>/<\/svg>/" > "$svg"
done

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
  inkscape "$svg" --export-filename="$png" "$export_dim"

  optipng -o7 -strip all "$png"
done

# cleanup bootstrap SVGs
for id in "${ICON_IDS[@]}"; do
  rm "${ICONS_DIR}/${id}.svg"
done
