#!/usr/bin/env bash

# only pyflakes check (--select=F)
RESULT=$(python3 -m flake8 --select=F $(eval echo $(cat <(find cereal) <(find opendbc) release/files_common release/files_common | tr '\n' ' ') | tr ' ' '\n' | grep "\.py$"))
if [[ $RESULT  ]]; then
	echo "Pyflakes found errors in the code. Please fix and try again"
	echo "$RESULT"
	exit 1
fi
