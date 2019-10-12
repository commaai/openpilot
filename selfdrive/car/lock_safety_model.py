#!/usr/bin/env python3
import sys
from cereal import car
from common.params import Params

# This script locks the safety model to a given value.
# When the safety model is locked, boardd will preset panda to the locked safety model

# run example:
# ./lock_safety_model.py gm

if __name__ == "__main__":

  params = Params()

  if len(sys.argv) < 2:
    params.delete("SafetyModelLock")
    print("Clear locked safety model")

  else:
    safety_model = getattr(car.CarParams.SafetyModel, sys.argv[1])
    if type(safety_model) != int:
      raise Exception("Invalid safety model: " + sys.argv[1])
    if safety_model == car.CarParams.SafetyModel.allOutput:
      raise Exception("Locking the safety model to allOutput is not allowed")
    params.put("SafetyModelLock", str(safety_model))
    print("Locked safety model: " + sys.argv[1])
