#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

if [ ! -f fontbm ]; then
  rm -rf /tmp/fontbm
  git clone git@github.com:vladimirgamalyan/fontbm.git /tmp/fontbm

  cd /tmp/fontbm
  mkdir -p build
  cd build
  cmake ../
  make -j8

  cp fontbm $DIR/
fi

for file in *.ttf; do
  name="${file%.ttf}"
  ./fontbm --font-file $file --output $name
done
