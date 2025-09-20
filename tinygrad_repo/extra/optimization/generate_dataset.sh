#!/bin/bash
export PAGE_SIZE=1
export PYTHONPATH=.
export LOGOPS=/tmp/ops
export CAPTURE_PROCESS_REPLAY=1
rm $LOGOPS
test/external/process_replay/reset.py

CI=1 python3 -m pytest -n=auto test/test_ops.py test/test_nn.py test/test_winograd.py test/models/test_real_world.py --durations=20
GPU=1 python3 -m pytest test/test_tiny.py

# extract, sort and uniq
extra/optimization/extract_dataset.py
sort -u /tmp/ops > /tmp/sops
ls -lh /tmp/ops /tmp/sops
gzip -k /tmp/sops
# mv /tmp/sops.gz extra/datasets/