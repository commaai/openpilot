#!/usr/bin/env bash

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASEDIR"
pwd

PY_FILES=$(git ls-files 'layouts/*.py' 'widgets/*.py')

xgettext -L Python \
  --keyword=tr \
  --keyword=trn:1,2 \
  --keyword=pgettext:1c,2 \
  --from-code=UTF-8 \
  --flag=tr:1:python-brace-format \
  --flag=trn:1:python-brace-format --flag=trn:2:python-brace-format \
  -o translations/app.pot \
  $PY_FILES

msginit \
  -l es \
  --no-translator \
  --input translations/app.pot \
  --output-file translations/app.po
