#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export DEFAULT_FLOAT="HALF" SUM_DTYPE="HALF" GPUS=6 BS=96 EVAL_BS=96

export BEAM=3 BEAM_UOPS_MAX=4000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/wiki"

export WANDB=1 PARALLEL=0

RUNMLPERF=1 python3 examples/mlperf/model_train.py