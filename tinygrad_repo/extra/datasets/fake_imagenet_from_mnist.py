#!/usr/bin/env python3
import pathlib, json
from tinygrad.helpers import trange
from extra.datasets import fetch_mnist
from PIL import Image
import numpy as np
from multiprocessing import Pool

X_train, Y_train, X_test, Y_test = fetch_mnist()

def act(arg):
  (basedir, i, train) = arg
  if train:
    img = np.uint8(X_train[i]).reshape(28, 28)
    nm = f"train/{Y_train[i]}/{i}.jpg"
  else:
    img = np.uint8(X_test[i]).reshape(28, 28)
    nm = f"val/{Y_test[i]}/{i}.jpg"
  Image.fromarray(img).resize((224, 224)).convert('RGB').save(basedir / nm)

def create_fake_mnist_imagenet(basedir:pathlib.Path):
  print(f"creating mock MNIST dataset at {basedir}")
  basedir.mkdir(exist_ok=True)

  with (basedir / "imagenet_class_index.json").open('w') as f:
    f.write(json.dumps({str(i):[str(i), str(i)] for i in range(10)}))

  for i in range(10):
    (basedir / f"train/{i}").mkdir(parents=True, exist_ok=True)
    (basedir / f"val/{i}").mkdir(parents=True, exist_ok=True)

  def gen(train):
    for idx in trange(X_train.shape[0] if train else X_test.shape[0]):
      yield (basedir, idx, train)

  with Pool(64) as p:
    for _ in p.imap_unordered(act, gen(True)): pass
    for _ in p.imap_unordered(act, gen(False)): pass

if __name__ == "__main__":
  create_fake_mnist_imagenet(pathlib.Path("./mnist"))