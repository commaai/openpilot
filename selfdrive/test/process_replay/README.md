# Process replay

Process replay is a regression test designed to identify any changes in the output of a process. This test replays a segment through individual processes and compares the output to a known good replay. Each make is represented in the test with a segment.

If the test fails, make sure that you didn't unintentionally change anything. If there are intentional changes, the reference logs will be updated.

Use `test_processes.py` to run the test locally.
Use `FILEREADER_CACHE='1' test_processes.py` to cache log files.

Currently the following processes are tested:

* controlsd
* radard
* plannerd
* calibrationd
* dmonitoringd
* locationd
* paramsd
* ubloxd
* torqued

### Usage
```
Usage: test_processes.py [-h] [--whitelist-procs PROCS] [--whitelist-cars CARS] [--blacklist-procs PROCS]
                         [--blacklist-cars CARS] [--ignore-fields FIELDS] [--ignore-msgs MSGS] [--update-refs] [--upload-only]
Regression test to identify changes in a process's output
optional arguments:
  -h, --help            show this help message and exit
  --whitelist-procs PROCS               Whitelist given processes from the test (e.g. controlsd)
  --whitelist-cars WHITELIST_CARS       Whitelist given cars from the test (e.g. HONDA)
  --blacklist-procs BLACKLIST_PROCS     Blacklist given processes from the test (e.g. controlsd)
  --blacklist-cars BLACKLIST_CARS       Blacklist given cars from the test (e.g. HONDA)
  --ignore-fields IGNORE_FIELDS         Extra fields or msgs to ignore (e.g. driverMonitoringState.events)
  --ignore-msgs IGNORE_MSGS             Msgs to ignore (e.g. onroadEvents)
  --update-refs                         Updates reference logs using current commit
  --upload-only                         Skips testing processes and uploads logs from previous test run
```

## Forks

openpilot forks can use this test with their own reference logs, by default `test_proccess.py` saves logs locally.

To generate new logs:

`./test_processes.py`

Then, check in the new logs using git-lfs. Make sure to also update the `ref_commit` file to the current commit.

## API

Process replay test suite exposes programmatic APIs for simultaneously running processes or groups of processes on provided logs. 

```py
def replay_process_with_name(name: Union[str, Iterable[str]], lr: LogIterable, *args, **kwargs) -> List[capnp._DynamicStructReader]:

def replay_process(
  cfg: Union[ProcessConfig, Iterable[ProcessConfig]], lr: LogIterable, frs: Optional[Dict[str, Any]] = None, 
  fingerprint: Optional[str] = None, return_all_logs: bool = False, custom_params: Optional[Dict[str, Any]] = None, disable_progress: bool = False
) -> List[capnp._DynamicStructReader]:
```

Example usage:
```py
from openpilot.selfdrive.test.process_replay import replay_process_with_name
from openpilot.tools.lib.logreader import LogReader

lr = LogReader(...)

# provide a name of the process to replay
output_logs = replay_process_with_name('locationd', lr)

# or list of names
output_logs = replay_process_with_name(['ubloxd', 'locationd'], lr)
```

Supported processes: 
* controlsd
* radard
* plannerd
* calibrationd
* dmonitoringd
* locationd
* paramsd 
* ubloxd
* torqued
* modeld
* dmonitoringmodeld

Certain processes may require an initial state, which is usually supplied within `Params` and persisting from segment to segment (e.g CalibrationParams, LiveParameters). The `custom_params` is dictionary  used to prepopulate `Params` with arbitrary values. The `get_custom_params_from_lr` helper is provided to fetch meaningful values from log files.

```py
from openpilot.selfdrive.test.process_replay import get_custom_params_from_lr

previous_segment_lr = LogReader(...)
current_segment_lr = LogReader(...)

custom_params = get_custom_params_from_lr(previous_segment_lr, 'last')

output_logs = replay_process_with_name('calibrationd', lr, custom_params=custom_params)
```

Replaying processes that use VisionIPC (e.g. modeld, dmonitoringmodeld) require additional `frs` dictionary with camera states as keys and `FrameReader` objects as values.

```py
from openpilot.tools.lib.framereader import FrameReader

frs = {
  'roadCameraState': FrameReader(...),
  'wideRoadCameraState': FrameReader(...),
  'driverCameraState': FrameReader(...),
}

output_logs = replay_process_with_name(['modeld', 'dmonitoringmodeld'], lr, frs=frs)
```

To capture stdout/stderr of the replayed process, `captured_output_store` can be provided.

```py
output_store = dict()
# pass dictionary by reference, it will be filled with standard outputs - even if process replay fails
output_logs = replay_process_with_name(['radard', 'plannerd'], lr, captured_output_store=output_store)

# entries with captured output in format { 'out': '...', 'err': '...' } will be added to provided dictionary for each replayed process
print(output_store['radard']['out']) # radard stdout
print(output_store['radard']['err']) # radard stderr
```
