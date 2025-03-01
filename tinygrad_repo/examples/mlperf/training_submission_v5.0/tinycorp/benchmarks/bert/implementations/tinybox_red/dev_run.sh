#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=78 EVAL_BS=78

export BEAM=3 BEAM_UOPS_MAX=3000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/wiki"

export WANDB=1 PARALLEL=0

RUNMLPERF=1 python3 examples/mlperf/model_train.py