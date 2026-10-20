import subprocess
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tinygrad.helpers import getenv

# checks that HCQ drivers can be killed during operation without causing issues

def run_test(i, full_run=False, force_ok=False):
  print(f"\rRunning iteration {i}...", end=" ", flush=True)

  p = subprocess.Popen(["python3", "test/test_tiny.py", "TestTiny.test_plus"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  if not full_run:
    time.sleep(random.uniform(0, 1200) / 1000.0)
    p.kill()
    _, stderr = p.communicate()
  else:
    _, stderr = p.communicate()
    stderr_text = stderr.decode()
    assert ("Ran 1 test in" in stderr_text and "OK" in stderr_text) or (not force_ok and "Failed to take lock file" in stderr_text), stderr_text

if __name__ == "__main__":
  max_workers = getenv("MAX_WORKERS", 4)
  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for i in range(1000000):
      if i % 100 == 0:
        # wait for everything we launched so far
        for f in as_completed(futures):
          try:
            f.result()
          except Exception as e:
            print(f"\nError in iteration: {e}")
        futures = []

        # do a full run in the main proc
        run_test(i, True, force_ok=True)
      else:
        futures.append(executor.submit(run_test, i, bool(getenv("FULL_RUN", 0))))

      # keep list small
      if len(futures) > max_workers * 2:
        futures = [f for f in futures if not f.done()]
