#!/bin/bash

set -e
HEAD=$(git rev-parse --abbrev-ref HEAD)
python test/external/process_replay/reset.py
CAPTURE_PROCESS_REPLAY=1 python test/test_ops.py TestOps.test_add
git checkout master
git checkout $HEAD -- test/external/process_replay/process_replay.py
ASSERT_PROCESS_REPLAY=${ASSERT_PROCESS_REPLAY:-1} python test/external/process_replay/process_replay.py
git checkout $HEAD
