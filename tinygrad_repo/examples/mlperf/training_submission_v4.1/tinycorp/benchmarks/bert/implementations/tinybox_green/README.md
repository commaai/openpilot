# 1. Problem

This problem uses BERT for NLP.

## Requirements

Install tinygrad and mlperf-logging from master.
```
git clone https://github.com/tinygrad/tinygrad.git
python3 -m pip install -e ".[mlperf]"
```
Also install tqdm and tensorflow.
```
pip install tqdm tensorflow
```

### tinybox_green
Install the p2p driver per [README](https://github.com/tinygrad/open-gpu-kernel-modules/blob/550.54.15-p2p/README.md)
This is the default on production tinybox green.

### tinybox_red
Disable cwsr + increase mes timeout.
Install the custom amdgpu driver per [README](https://github.com/nimlgen/amdgpu_ubuntu_22_04/blob/v6.1.3/readme.md)

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
examples/mlperf/training_submission_v4.1/tinycorp/benchmarks/bert/implementations/tinybox_green/run_and_time.sh
```

### tinybox_red

#### One time setup

```
examples/mlperf/training_submission_v4.1/tinycorp/benchmarks/bert/implementations/tinybox_red/setup.sh
```

#### Steps to run benchmark
```
examples/mlperf/training_submission_v4.1/tinycorp/benchmarks/bert/implementations/tinybox_red/run_and_time.sh
```