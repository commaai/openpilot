#!/bin/bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

export PYTHONPATH="." AMD=1
export MODEL="bert"
export SUBMISSION_PLATFORM="tinybox_red"
export DEFAULT_FLOAT="HALF" SUM_DTYPE="HALF" GPUS=6 BS=90 EVAL_BS=90

export IGNORE_OOB=1

export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/wiki"

# pip install -e ".[mlperf]"
export LOGMLPERF=1

export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
LOGFILE="bert_red_${DATETIME}_${SEED}.log"

export HCQDEV_WAIT_TIMEOUT_MS=100000  # prevents hang?

# init
sleep 5 && sudo rmmod amdgpu || true
BENCHMARK=10 INITMLPERF=1 BERT_LAYERS=2 python3 examples/mlperf/model_train.py | tee $LOGFILE

# run
PARALLEL=0 RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a $LOGFILE
