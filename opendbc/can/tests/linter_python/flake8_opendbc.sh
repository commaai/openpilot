#!/usr/bin/env bash

RESULT=$(python3 -m flake8 --select=F $(find ../../../ -type f | grep "\.py$"))
if [[ $RESULT  ]]; then
	echo "Pyflakes found errors in the code. Please fix and try again"
	echo "$RESULT"
	exit 1
fi
