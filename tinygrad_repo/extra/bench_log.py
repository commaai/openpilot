import time, atexit, uuid
from enum import Enum

from tinygrad.device import Device
from tinygrad.helpers import DEBUG, ContextVar, getenv, GlobalCounters

BENCHMARK_LOG = ContextVar("BENCHMARK_LOG", "")

if BENCHMARK_LOG:
  from influxdb_client_3 import InfluxDBClient3, Point, WriteOptions, write_client_options
  from influxdb_client_3.write_client.client.write_api import WriteType

class BenchEvent(Enum):
  LOAD_WEIGHTS = "load_weights"
  STEP = "step"
  FULL = "full"
  MLPERF_INIT = "mlperf_init"
  MLPERF_RUN = "mlperf_run"
class InstantBenchEvent(Enum):
  GFLOPS = "gflops"

_events = {}
def clear_events():
  for event in BenchEvent:
    _events[event] = {"wall": [], "kernel": []}
  for event in InstantBenchEvent:
    _events[event] = []
clear_events()

class WallTimeEvent:
  def __init__(self, event:BenchEvent):
    self.event = event
  def __enter__(self):
    self.start = time.monotonic()
    return self
  def __exit__(self, *_):
    _events[self.event]["wall"].append(time.monotonic() - self.start)
    return False

class KernelTimeEvent:
  def __init__(self, event:BenchEvent):
    if DEBUG < 2:
      raise Exception("KernelTimeEvent should only be used in DEBUG >= 2")
    self.event = event
  def __enter__(self):
    self.start = GlobalCounters.time_sum_s
    return self
  def __exit__(self, *_):
    _events[self.event]["kernel"].append(GlobalCounters.time_sum_s - self.start)
    return False

def log_event_instant(event:InstantBenchEvent, value:float):
  _events[event].append(value)

if BENCHMARK_LOG:
  INFLUXDB_HOST = getenv("INFLUXDB_HOST", "")
  INFLUXDB_ORG = getenv("INFLUXDB_ORG", "tiny")
  INFLUXDB_TOKEN = getenv("INFLUXDB_TOKEN", "")

  def _create_point(run_id, i, attempt, ref, commit, name, value, run):
    point = Point(BENCHMARK_LOG.value).tag("id", run_id).tag("index", i)
    point = point.tag("device", Device.DEFAULT)
    point = point.tag("attempt", attempt).tag("ref", ref).tag("commit", commit)
    point = point.field(name, value).field("x", run)
    return point

  @atexit.register
  def write_events():
    # see if there are any events to write
    have_events = False
    for event in _events:
      if isinstance(event, BenchEvent):
        for event_type, values in _events[event].items():
          if len(values) > 0:
            have_events = True
      else:
        if len(_events[event]) > 0:
          have_events = True
    if not have_events:
      return

    # pull from github envvars
    ref = getenv("GITHUB_REF_NAME", "")
    commit = getenv("GITHUB_SHA", "")
    run = getenv("GITHUB_RUN_NUMBER", "")
    attempt = getenv("GITHUB_RUN_ATTEMPT", "")

    points = []
    for event in _events:
      run_id = str(uuid.uuid4())
      if isinstance(event, BenchEvent):
        for event_type, values in _events[event].items():
          for i, value in enumerate(values):
            point = _create_point(run_id, i, attempt, ref, commit, f"{event.value}_{event_type}", value, run)
            points.append(point)
      else:
        for i, value in enumerate(_events[event]):
          point = _create_point(run_id, i, attempt, ref, commit, event.value, value, run)
          points.append(point)

    write_options = WriteOptions(write_type=WriteType.synchronous, retry_interval=5000, max_retries=5, max_retry_delay=30000, exponential_base=2)
    wco = write_client_options(write_options=write_options)
    with InfluxDBClient3(
        host=INFLUXDB_HOST,
        org=INFLUXDB_ORG,
        token=INFLUXDB_TOKEN,
        auth_scheme="Basic",
        database="benchmarks",
        write_client_options=wco) as client:
      client.write(points)
