#!/usr/bin/env python3
import os
import time
from typing import NoReturn

from common.realtime import set_core_affinity, set_realtime_priority

def main() -> NoReturn:
    # RT shield - ensure CPU 3 always remains available for RT processes
    core = int(os.getenv("CORE", "3"))
    set_core_affinity([core, ])
    set_realtime_priority(1)

    # Use a loop with a sleep time of 1ms instead of 0.000001 to reduce CPU usage
    while True:
        time.sleep(0.001)

if __name__ == "__main__":
    main()
