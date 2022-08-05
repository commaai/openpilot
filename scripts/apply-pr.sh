#!/bin/bash

if [ $# -eq 0 ]; then
  echo "usage: $0 <pull-request-number>"
  exit 1
fi

BASE="https://github.com/commaai/openpilot/pull/"
PR_NUM="$(echo $1 | grep -o -E '[0-9]+')"

curl -L $BASE/$PR_NUM.patch | git apply
