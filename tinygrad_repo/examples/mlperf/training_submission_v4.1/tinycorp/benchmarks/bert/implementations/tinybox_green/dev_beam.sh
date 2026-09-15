#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=66 EVAL_BS=6

export BEAM=4 BEAM_UOPS_MAX=2000 BEAM_UPCAST_MAX=64 BEAM_LOCAL_MAX=512
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/wiki"

export BENCHMARK=10 DEBUG=2

python3 examples/mlperf/model_train.py
