#!/usr/bin/env python3
import numpy as np
import random

from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
from extra.training import train, evaluate
from extra.models.transformer import Transformer

# dataset idea from https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py
def make_dataset():
  ds = []
  for i in range(100):
    for j in range(100):
      s = i+j
      ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
  random.shuffle(ds)
  ds = np.array(ds).astype(np.float32)
  ds_X = ds[:, 0:6]
  ds_Y = np.copy(ds[:, 1:])
  ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
  ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]
  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test

if __name__ == "__main__":
  model = Transformer(10, 6, 2, 128, 4, 32)
  X_train, Y_train, X_test, Y_test = make_dataset()
  lr = 0.003
  for i in range(10):
    optim = Adam(get_parameters(model), lr=lr)
    train(model, X_train, Y_train, optim, 50, BS=64, allow_jit=True)
    acc, Y_test_preds = evaluate(model, X_test, Y_test, num_classes=10, return_predict=True)
    lr /= 1.2
    print(f'reducing lr to {lr:.4f}')
  if acc > 0.998:
    wrong=0
    for k in range(len(Y_test_preds)):
      if (Y_test_preds[k] != Y_test[k]).any():
        wrong+=1
        a,b,c,x = X_test[k,:2].astype(np.int32), X_test[k,2:4].astype(np.int32), Y_test[k,-3:].astype(np.int32), Y_test_preds[k,-3:].astype(np.int32)
        print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')
    print(f'Wrong predictions: {wrong}, acc = {acc:.4f}')
