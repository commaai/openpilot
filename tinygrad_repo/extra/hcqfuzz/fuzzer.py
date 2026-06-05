import os, random, subprocess, shlex, datetime, time, signal
from extra.hcqfuzz.tools import create_report, on_start_run, collect_tests, init_log, log
from extra.hcqfuzz.spec import AMSpec

def run_test(dev, test):
  on_start_run(dev, test)

  dev_env = dev.get_exec_state()
  test_env, cmd, timeout = test.get_exec_state()
  env = {**dev_env, **test_env}

  if isinstance(cmd, str): cmd = shlex.split(cmd)
  assert isinstance(cmd, list), "cmd must be list or str"

  if env is None: env = os.environ.copy()
  else:
    env = {k: str(v) for k, v in env.items()}
    env = {**os.environ, **env}

  start_ts = datetime.datetime.now()
  t0 = time.perf_counter()
  log(f"[{start_ts:%Y-%m-%d %H:%M:%S}] running: {test.name()}: {' '.join(cmd)}", end="", flush=True)

  proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  try:
    stdout, stderr = proc.communicate(timeout=timeout)
    ret = proc.returncode
  except KeyboardInterrupt:
    print("\nExiting...", flush=True)
    proc.send_signal(signal.SIGINT)
    try: stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
      proc.kill()
      stdout, stderr = proc.communicate()
    raise
  except subprocess.TimeoutExpired:
    cur_time = datetime.datetime.now()
    log(f"\r[{cur_time:%Y-%m-%d %H:%M:%S}] {test.name()} send SIGKILL", end="", flush=True)

    proc.kill()
    stdout, stderr = proc.communicate()
    ret = -9

  finish_time = datetime.datetime.now()
  elapsed = time.perf_counter() - t0
  if ret != 0:
    log(f"\r[{finish_time:%Y-%m-%d %H:%M:%S}] {test.name()} failed with {ret} after {elapsed:.1f}s", flush=True)
    create_report(dev, test, ret, stdout, stderr)
  else:
    log(f"\r[{finish_time:%Y-%m-%d %H:%M:%S}] {test.name()} exited {ret} after {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
  init_log()
  device_name = "AM"
  dev = AMSpec()

  start_seed = os.environ.get("SEED", 3332)
  random.seed(start_seed)

  log(f"Starting with seed {start_seed}")

  test_set = collect_tests()
  log(f"Found {len(test_set)} tests:")
  for test in test_set: log(f" - {test.name()}")

  while True:
    seed = random.randint(0, 2**31)
    test = random.choice(test_set)

    dev.prepare(seed)
    test.prepare(dev, seed)
    run_test(dev, test)
