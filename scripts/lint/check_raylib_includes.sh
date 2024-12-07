#!/usr/bin/env bash

FAIL=0

if grep -n '#include "third_party/raylib/include/raylib\.h"' $@ | grep -v '^system/ui/raylib/raylib\.h'; then
  echo -e "Bad raylib include found! Use '#include \"system/ui/raylib/raylib.h\"' instead\n"
  FAIL=1
fi

exit $FAIL
