#!/bin/bash
python3 test/external/process_replay/reset.py
RUN_PROCESS_REPLAY=1 pytest -n auto test/test_tiny.py test/test_uop_graph.py test/test_ops.py test/test_linearizer.py
while true; do
  if python3 test/test_tiny.py; then
    PYTHONPATH="." python3 test/external/process_replay/process_replay.py
  fi
done