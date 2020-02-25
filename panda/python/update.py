#!/usr/bin/env python3
import os
import time

def ensure_st_up_to_date():
  from panda import Panda, PandaDFU, BASEDIR

  with open(os.path.join(BASEDIR, "VERSION")) as f:
    repo_version = f.read()

  repo_version += "-EON" if os.path.isfile('/EON') else "-DEV"

  panda = None
  panda_dfu = None

  while 1:
    # break on normal mode Panda
    panda_list = Panda.list()
    if len(panda_list) > 0:
      panda = Panda(panda_list[0])
      break

    # flash on DFU mode Panda
    panda_dfu = PandaDFU.list()
    if len(panda_dfu) > 0:
      panda_dfu = PandaDFU(panda_dfu[0])
      panda_dfu.recover()

    print("waiting for board...")
    time.sleep(1)

  if panda.bootstub or not panda.get_version().startswith(repo_version):
    panda.flash()

  if panda.bootstub:
    panda.recover()

  assert(not panda.bootstub)
  version = str(panda.get_version())
  print("%s should be %s" % (version, repo_version))
  assert(version.startswith(repo_version))

if __name__ == "__main__":
  ensure_st_up_to_date()

