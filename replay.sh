#!/bin/bash
set -e

source .venv/bin/activate.fish
cd tools/replay
./replay '0c7f0c7f0c7f0c7f|2021-10-13--13-00-00' --dcam --ecam
