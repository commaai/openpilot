#!/bin/bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

export PYTHONPATH="." AMD=1
export MODEL="bert"
export SUBMISSION_PLATFORM="tinybox_8xMI300X"
export DEFAULT_FLOAT="HALF" GPUS=8 BS=1024 EVAL_BS=1024

# similar to https://github.com/mlcommons/training_results_v3.1/blob/d06288b2bd675a9d88e0e6181f5bb5626b71ec19/Quanta_Cloud_Technology/results/D54U-3U/bert/result_1.txt#L54
export OPT_BASE_LEARNING_RATE=0.0011 OPT_LAMB_BETA_1=0.60466 OPT_LAMB_BETA_2=0.85437 DECAY=0.1
export TRAIN_STEPS=3900

export BEAM=3 BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1 FREE_INTERMEDIATE=0
export BASEDIR="/raid/datasets/wiki"

# pip install -e ".[mlperf]"
export LOGMLPERF=1

export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
LOGFILE="bert_8xMI300x_${DATETIME}_${SEED}.log"

# init  # TODO: without DEBUG=2 it hangs
BENCHMARK=10 INITMLPERF=1 BERT_LAYERS=2 DEBUG=2 python3 examples/mlperf/model_train.py | tee $LOGFILE

# run
PARALLEL=0 RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a $LOGFILE
