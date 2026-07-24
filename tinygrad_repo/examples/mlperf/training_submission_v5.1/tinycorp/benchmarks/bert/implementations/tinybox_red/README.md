# 1. Problem

This problem uses BERT for NLP.

## Requirements

Install tinygrad and mlperf-logging (uncomment mlperf from setup.py) from branch mlperf_training_v5.0.
```
git clone https://github.com/tinygrad/tinygrad.git
python3 -m pip install -e ".[mlperf]"
```
Also install gdown (for dataset), numpy, tqdm and tensorflow.
```
pip install gdown numpy tqdm tensorflow
```

### tinybox_green
Install the p2p driver per [README](https://github.com/tinygrad/open-gpu-kernel-modules/blob/550.54.15-p2p/README.md)
This is the default on production tinybox green.

# 2. Directions

## Steps to download and verify data

### 1. Download raw data

```
BASEDIR="/raid/datasets/wiki" WIKI_TRAIN=1 VERIFY_CHECKSUM=1 python3 extra/datasets/wikipedia_download.py
```

### 2. Preprocess train and validation data

Note: The number of threads used for preprocessing is limited by available memory. With 128GB of RAM, a maximum of 16 threads is recommended. 

#### Training:
```
BASEDIR="/raid/datasets/wiki" NUM_WORKERS=16 python3 extra/datasets/wikipedia.py pre-train all
```

Generating a specific topic (Between 0 and 499)
```
BASEDIR="/raid/datasets/wiki" python3 extra/datasets/wikipedia.py pre-train 42
```

#### Validation:
```
BASEDIR="/raid/datasets/wiki" python3 extra/datasets/wikipedia.py pre-eval
```
## Running

### tinybox_green

#### Steps to run benchmark
```
examples/mlperf/training_submission_v5.0/tinycorp/benchmarks/bert/implementations/tinybox_green/run_and_time.sh
```

### tinybox_red

#### Steps to run benchmark
```
examples/mlperf/training_submission_v5.0/tinycorp/benchmarks/bert/implementations/tinybox_red/run_and_time.sh
```
### tinybox_8xMI300X

#### Steps to run benchmark
```
examples/mlperf/training_submission_v5.0/tinycorp/benchmarks/bert/implementations/tinybox_8xMI300X/run_and_time.sh
```