#!/bin/bash

export DUAL="0"
export BLOCK="${BLOCK},camerad"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
python3 $DIR/camerad.py
