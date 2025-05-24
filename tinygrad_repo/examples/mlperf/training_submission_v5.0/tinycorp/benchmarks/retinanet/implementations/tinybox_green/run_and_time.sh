#!/bin/bash

export PYTHONPATH="." NV=1
export MODEL="retinanet"
export SUBMISSION_PLATFORM="tinybox_green"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=96 EVAL_BS=96

export TRAIN_BEAM=2 BEAM_UOPS_MAX=1500 BEAM_UPCAST_MAX=64 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=0
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/openimages"

# pip install -e ".[mlperf]"
export LOGMLPERF=1

export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
LOGFILE="retinanet_green_${DATETIME}_${SEED}.log"

# init
BENCHMARK=10 INITMLPERF=1 python3 examples/mlperf/model_train.py | tee $LOGFILE

# run
PARALLEL=0 RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a $LOGFILE
