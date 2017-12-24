import os
from panda import build_st

def test_build_legacy():
  build_st("obj/comma.bin", "Makefile.legacy")

def test_build_bootstub_legacy():
  build_st("obj/bootstub.comma.bin", "Makefile.legacy")

def test_build_panda():
  build_st("obj/panda.bin")

def test_build_bootstub_panda():
  build_st("obj/bootstub.panda.bin")

