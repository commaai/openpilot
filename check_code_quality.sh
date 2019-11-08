#!/bin/bash

# Only pyflakes checks (--select=F)
flake8 --select=F $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda")
RESULT=$?
if [ $RESULT -eq 0 ]; then
    pylint $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda")
    RESULT=$? & 3
fi

[ $RESULT -ne 0 ] && exit 1
exit 0
