#!/usr/bin/env python3
import threading
import traceback
import sys
sys.path.insert(0, 'opendbc_repo')
from opendbc.car.tests.routes import routes
from openpilot.tools.lib.logreader import LogReader

# Get first n routes (or all if n > len(routes))
N_ROUTES = 10
ROUTES = [r.route for r in routes[:N_ROUTES]]

results = []
lock = threading.Lock()


def read_logs(thread_id, route):
  try:
    lr = LogReader(route)
    with lock:
      results.append(f"Thread {thread_id} ({route}): OK")
  except Exception as e:
    with lock:
      tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__, chain=True))
      results.append(f"Thread {thread_id} ({route}):\n{tb_str}")


threads = [threading.Thread(target=read_logs, args=(i, ROUTES[i % len(ROUTES)])) for i in range(10)]
for t in threads:
  t.start()
for t in threads:
  t.join()

for r in results:
  print(r)

successful = sum(1 for r in results if ": OK" in r)
print(f"\n{successful}/{len(threads)} threads successful")
