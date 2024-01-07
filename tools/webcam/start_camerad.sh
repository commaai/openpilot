#!/bin/bash
export DUAL="0"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export BLOCK=camerad
python3 $DIR/camerad.py
