#!/usr/bin/python3

import tensorflow as tf
import os
import sys
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from pathlib import Path


name = sys.argv[1].split('.keras')[0]
model = load_model(sys.argv[1])
model.save_weights(f"{name}.weights.keras")
model_json = model.to_json()
with open(f"{name}.model.keras", "w") as json_file:
    json_file.write(model_json)
json_file.close()
