#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d icons/ ]; then
  git clone https://github.com/twbs/icons/
fi

cd icons
git fetch --all
git checkout d5aa187483a1b0b186f87adcfa8576350d970d98
cp bootstrap-icons.svg ../
