#!/usr/bin/env python3
import argparse
from multiprocessing import Queue, Process

from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument('--timeout', type=int, default=0)
args = parser.parse_args()

from tools.sim.bridge import bridge_keep_alive

# Test connecting to Carla within 1 minute
q: Any = Queue()
p = Process(target=bridge_keep_alive, args=(q,), daemon=True)
p.start()
p.join(args.timeout + 1)  # to ensure script terminates
