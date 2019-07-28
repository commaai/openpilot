#!/usr/bin/env python
import os
import shutil
import threading
from selfdrive.swaglog import cloudlog
from selfdrive.loggerd.config import ROOT, get_available_percent
from selfdrive.loggerd.uploader import listdir_by_creation_date


def deleter_thread(exit_event):
  while not exit_event.is_set():
    available_percent = get_available_percent()

    if available_percent < 10.0:
      # remove the earliest directory we can
      dirs = listdir_by_creation_date(ROOT)
      for delete_dir in dirs:
        delete_path = os.path.join(ROOT, delete_dir)

        if any(name.endswith(".lock") for name in os.listdir(delete_path)):
          continue

        try:
          cloudlog.info("deleting %s" % delete_path)
          shutil.rmtree(delete_path)
          break
        except OSError:
          cloudlog.exception("issue deleting %s" % delete_path)
      exit_event.wait(.1)
    else:
      exit_event.wait(30)


def main(gctx=None):
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
