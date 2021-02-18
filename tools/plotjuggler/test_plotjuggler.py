#!/usr/bin/env python3
import unittest
import os


class TestPlotJuggler(unittest.TestCase):

  def test_install(self):
    exit_code = os.system("./install.sh")
    self.assertEqual(exit_code, 0)

  def test_run(self):
    exit_code = os.system('./juggle.py "0982d79ebb0de295|2021-01-17--17-13-08"')
    self.assertEqual(exit_code, 0)
