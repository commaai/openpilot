# 1. Problem

This problem uses RetinaNet for SSD.

## Requirements

Install tinygrad and mlperf-logging (uncomment mlperf from setup.py) from branch mlperf_training_v5.0.
```
git clone https://github.com/tinygrad/tinygrad.git
python3 -m pip install -e ".[mlperf]"
```

Also install the following dependencies:
```
pip install tqdm numpy pycocotools boto3 pandas torch torchvision
```

### tinybox_green
Install the p2p driver per [README](https://github.com/tinygrad/open-gpu-kernel-modules/blob/550.54.15-p2p/README.md)
This is the default on production tinybox green.

# 2. Directions

## Steps to download data

Run the following:
```
BASEDIR=/raid/datasets/openimages python3 extra/datasets/openimages.py
```

## Running

### tinybox_green

#### Steps to run benchmark
```
examples/mlperf/training_submission_v5.0/tinycorp/benchmarks/retinanet/implementations/tinybox_green/run_and_time.sh
```
