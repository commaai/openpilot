#!/bin/bash
# usage: replay_many.sh ROUTE OUTPREFIX SEG [SEG ...]   runs segments in parallel (5 at a time), one npz per segment
# ONNX picks the model (default: the tapped master model in the data dir)
CODE=$(cd "$(dirname "$0")" && pwd)
DATA=${WEIGHT_LAB_DATA:-/home/batman/xx/junk/shane/weight_edit}
ROUTE=$1; PREFIX=$2; shift 2
cd /home/batman/openpilot
printf '%s\n' "$@" | xargs -P 5 -I{} sh -c "~/xx/.venv/bin/python $CODE/replay.py $ROUTE/{} --onnx ${ONNX:-$DATA/tapped2.onnx} --out $DATA/seg/${PREFIX}_{}.npz > $DATA/seg/${PREFIX}_{}.log 2>&1 && echo done {} || echo FAIL {}"
