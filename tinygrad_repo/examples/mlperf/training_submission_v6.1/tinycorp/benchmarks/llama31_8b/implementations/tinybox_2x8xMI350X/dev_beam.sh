#!/usr/bin/env bash
export NODES=${NODES:-"tinyamd3 tinyamd4"}
export REMOTE=${REMOTE:-"local,$(echo $NODES | tr ' ' '\n' | grep -vx "$(hostname)" | head -1):6667"}
export DEV=PCI+AMD REMOTE_TIMEOUT=600 ALLREDUCE_NODE_NDEVS=8 DP=16 BS=32 EVAL_BS=16
exec bash "$(dirname "$0")/../tinybox_8xMI350X/dev_beam.sh"
