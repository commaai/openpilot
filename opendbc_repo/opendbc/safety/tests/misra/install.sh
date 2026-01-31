#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
: "${CPPCHECK_DIR:=$DIR/cppcheck/}"

# skip if we're running in parallel with test_mutation.py
if [ ! -z "$OPENDBC_ROOT" ]; then
  exit 0
fi

if [ ! -d "$CPPCHECK_DIR" ]; then
  git clone https://github.com/danmar/cppcheck.git $CPPCHECK_DIR
fi

cd $CPPCHECK_DIR

VERS="2.19.1"
if [ "$(git describe --tags --always)" != "$VERS" ]; then
  git fetch --all --tags --force
  git checkout $VERS
fi

#make clean
make MATCHCOMPILTER=yes CXXFLAGS="-O2" -j8
