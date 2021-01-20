import time
from multiprocessing import Process

# Note: this does not return any return values of the function, just the exit status
INTERVAL = 0.1
def run_with_timeout(timeout, fn, *kwargs):
  def runner(fn, kwargs):
    try:
      fn(*kwargs)
    except Exception as e:
      print(e)
      raise e

  process = Process(target=runner, args=(fn, kwargs))
  process.start()

  counter = 0
  while process.is_alive():
    time.sleep(INTERVAL)
    counter += 1
    if (counter * INTERVAL) > timeout:
      process.terminate()
      raise TimeoutError("Function timed out!")
  if process.exitcode != 0:
    raise RuntimeError("Test failed with exit code: ", str(process.exitcode))
