#!/bin/bash

export PYTHONPATH="." AMD=1
export MODEL="retinanet"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=96 EVAL_BS=96
export BASEDIR="/raid/datasets/openimages"

# export RESET_STEP=0

export TRAIN_BEAM=2 IGNORE_JIT_FIRST_BEAM=1 BEAM_UOPS_MAX=1500 BEAM_UPCAST_MAX=64 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=0

export BENCHMARK=5 DEBUG=2

python examples/mlperf/model_train.py
