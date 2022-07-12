#!/usr/bin/env python3
import os
import pickle

from common.basedir import BASEDIR
from selfdrive.car.docs import get_all_car_info

# TODO: take path argument, or ideally make a jinja template somehow?
with open(os.path.join(BASEDIR, '../openpilot_cache/old_car_info'), 'wb') as f:
  pickle.dump(get_all_car_info(), f)

print('Dumping to {}'.format(os.path.join(BASEDIR, '../openpilot_cache/old_car_info')))
