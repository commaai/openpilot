#!/usr/bin/env bash
set -e

if [ $# -eq 0 ]; then
  echo "usage: $0 <pull-request-number>"
  exit 1
fi

BASE="https://github.com/commaai/openpilot/pull/"
PR_NUM="$(echo $1 | grep -o -E '[0-9]+')"
BRANCH=tmp-pr${PR_NUM}

git branch -D -f $BRANCH || true
git fetch -u -f origin pull/$PR_NUM/head:$BRANCH
git switch $BRANCH
git reset --hard FETCH_HEAD
