# Process replay

Process replay is a regression test designed to identify any changes in the output of a process. This test replays a segment through individual processes and compares the output to a known good replay. Each make is represented in the test with a segment.

If the test fails, make sure that you didn't unintentionally change anything. If there are intentional changes, the reference logs will be updated.

Use `test_processes.py` to run the test locally.

Currently the following processes are tested:

* controlsd
* radard
* plannerd
* calibrationd
* ubloxd

### Usage test_processes.py
```
Usage: test_processes.py [-h] [--whitelist-procs PROCS] [--whitelist-cars CARS] [--blacklist-procs PROCS]
                         [--blacklist-cars CARS] [--ignore-fields FIELDS] [--ignore-msgs MSGS] [--timeout TIMEOUT]

Regression test to identify changes in a process's output

optional arguments:
  -h, --help            show this help message and exit
  --whitelist-procs PROCS               Whitelist given processes from the test (e.g. controlsd)
  --whitelist-cars WHITELIST_CARS       Whitelist given cars from the test (e.g. HONDA)
  --blacklist-procs BLACKLIST_PROCS     Blacklist given processes from the test (e.g. controlsd)
  --blacklist-cars BLACKLIST_CARS       Blacklist given cars from the test (e.g. HONDA)
  --ignore-fields IGNORE_FIELDS         Extra fields or msgs to ignore (e.g. carState.events)
  --ignore-msgs IGNORE_MSGS             Msgs to ignore (e.g. carEvents)
  --timeout TIMEOUT                     Set timeout of incoming messages
```
## Forks

openpilot forks can use this test with their own reference logs

To generate new logs:

`./update_refs.py --no-upload`

Then, check in the new logs using git-lfs. Make sure to also include the updated `ref_commit` file.
