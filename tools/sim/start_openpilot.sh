#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

cd ../../

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"

./launch_openpilot.sh