#!/usr/bin/env bash

git stash -q --keep-index
$(git rev-parse --show-toplevel)/flake8_openpilot.sh
RESULT=$?
if [ $RESULT -eq 0 ]; then
    IGNORE_REGEXP=$(echo -n "^(?!"; echo $(cat release/files_common release/files_common | tr '\n' ' ') | tr ' ' '\n' | grep "py$" | tr '\n' '|' | sed s'/|$//'; echo ").*")
    git-pylint-commit-hook --ignore $IGNORE_REGEXP
    RESULT=$?
fi

git stash pop -q

[ $RESULT -ne 0 ] && exit 1
exit 0
