# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements

Install tinygrad and mlperf-logging from master.
```
git clone https://github.com/tinygrad/tinygrad.git
python3 -m pip install -e ".[mlperf]"
```

### tinybox_green
Install the p2p driver per [README](https://github.com/tinygrad/open-gpu-kernel-modules/blob/550.54.15-p2p/README.md)
This is the default on production tinybox green.

### tinybox_red
Disable cwsr
This is the default on production tinybox red.
```
sudo vi /etc/modprobe.d/amdgpu.conf
cat <<EOF > /etc/modprobe.d/amdgpu.conf
options amdgpu cwsr_enable=0
EOF
sudo update-initramfs -u
sudo reboot

# validate
sudo cat /sys/module/amdgpu/parameters/cwsr_enable #= 0
```

# 2. Directions

## Steps to download and verify data

```
IMGNET_TRAIN=1 python3 extra/datasets/imagenet_download.py
```

## Steps for one time setup

### tinybox_red
```
examples/mlperf/training_submission_v4.0/tinycorp/benchmarks/resnet/implementations/tinybox_red/setup.sh
```

## Steps to run benchmark
```
examples/mlperf/training_submission_v4.0/tinycorp/benchmarks/resnet/implementations/tinybox_red/run_and_time.sh
```
