#!/usr/bin/env python3
import os
import random
import unittest

from selfdrive.test.openpilotci import get_url
from selfdrive.test.test_car_models import routes

class TestPlotJuggler(unittest.TestCase):

  def test_install(self):
    exit_code = os.system("./install.sh")
    self.assertEqual(exit_code, 0)

  def test_run(self):

    test_url = get_url(random.choice(routes.keys()), 0, log_type="rlog")

    exit_code = os.system(f'./juggle.py "{test_url}"')
    self.assertEqual(exit_code, 0)

if __name__ == "__main__":
  unittest.main()
