#!/usr/bin/env python3
import sys
import os

installed_version = f'{sys.version_info.major}.{sys.version_info.minor}'
required_version = os.environ.get('REQUIRED_PYTHON_VERSION', "3.11").replace('"','')
if not sys.version_info >= tuple(map(int, required_version.split('.'))):
  print(f'Your python version "{installed_version}" is too old. You need python version at least "{required_version}" to continue.')
  quit(1)
