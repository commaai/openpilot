#!/usr/bin/env bash

python3 -m pylint --disable=R,C,W $(eval echo <(find cereal) <(find opendbc) $(cat release/files_common release/files_common | tr '\n' ' ') | tr ' ' '\n' | grep "\.py$")

exit_status=$?
(( res = exit_status & 3 ))

if [[ $res != 0  ]]; then
	echo "Pylint found errors in the code. Please fix and try again"
	exit 1
fi
