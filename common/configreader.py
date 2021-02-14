#!/bin/env python3
"""
This is a demo script which is supposed to be called by any module to retrieve
a particular path from a config.ini file which defines a set of relevant paths.

Not huge care has been taken to make it work properly since it's just a demo
(albeit not a working one).
"""

import os
import sys
import configparser 


_THISPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConfigReader:

  def __init__(self):
    # SEARCHPATH is the set of paths where the config.ini file may be found
    SEARCHPATH = []
    if   sys.platform == "linux":
      SEARCHPATH = [ _THISPATH,
               "/etc/openpilot",
              f"{os.getenv('XDG_CONFIG_HOME')}/openpilot",
              f"{os.getenv('HOME')}/.openpilot"]
    elif sys.platform == "win32":
      SEARCHPATH = [f"{os.getenv('APPDATA')}\\Programs\\openpilot"]
    elif sys.platform == "darwin":
      # no idea, but this is just a demo so I'll pass
      pass

    self.cp = configparser.ConfigParser()

    loaded = False
    for path in SEARCHPATH:
      contents = self.cp.read(f"{path}/config.ini")
      if len(contents) > 0:
        loaded = True

    if not loaded:
      raise Exception("no config file found")

    self.basedir = self.get_path("BASEDIR")

    if self.basedir == "":
      self.basedir = _THISPATH
      self.cp["path"]["BASEDIR"] = self.basedir

    return

  def get_path(self, path_name):

    if path_name not in self.cp["path"]:
      raise Exception(f"{path_name} is not defined")

    prefix = ""
    if path_name != "BASEDIR":
      prefix = self.basedir

    return os.path.join(prefix, self.cp["path"][path_name])

  @staticmethod
  def static_get_path(path_name):
    tmp_cr = ConfigReader()
    return tmp_cr.get_path(path_name)


if __name__ == "__main__":
  # Simple test
  cr = ConfigReader()
  print(cr.get_path("BASEDIR"))
  print(cr.get_path("TOOLS_FRAMEREADER_VIDINDEX_DIR"))
