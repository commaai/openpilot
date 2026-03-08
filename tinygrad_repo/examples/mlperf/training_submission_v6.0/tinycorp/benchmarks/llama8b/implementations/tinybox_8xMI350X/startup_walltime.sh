#!/bin/bash
export BENCHMARK=5
export EVAL_BS=0
export FAKEDATA=1
export NULL_ALLOW_COPYOUT=1
export HIP_VISIBLE_DEVICES=""
export DEV=NULL
export JITBEAM=0
export LLAMA_LAYERS=${LLAMA_LAYERS:-"2"}
time examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama8b/implementations/tinybox_8xMI350X/dev_run.sh
