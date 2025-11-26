#!/usr/bin/env python3
import threading
import traceback
from openpilot.tools.lib.logreader import LogReader

ROUTE = "a703d058f4e05aeb/00000008--f169423024"
results = []
lock = threading.Lock()


def read_logs(thread_id):
  try:
    lr = LogReader(ROUTE)
    with lock:
      results.append(f"Thread {thread_id}: OK")
  except Exception as e:
    import sys
    with lock:
      tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__, chain=True))
      results.append(f"Thread {thread_id}:\n{tb_str}")


threads = [threading.Thread(target=read_logs, args=(i,)) for i in range(10)]
for t in threads:
  t.start()
for t in threads:
  t.join()

for r in results:
  print(r)
