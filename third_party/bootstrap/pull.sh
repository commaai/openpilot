#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d icons/ ]; then
  git clone https://github.com/twbs/icons/
fi

cd icons
git fetch --all
cp bootstrap-icons.svg ../
