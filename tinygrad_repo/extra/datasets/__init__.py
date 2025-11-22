import os, gzip, tarfile, pickle
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch

def fetch_mnist(tensors=False):
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"   # http://yann.lecun.com/exdb/mnist/ lacks https
  X_train = parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  if tensors: return Tensor(X_train).reshape(-1, 1, 28, 28), Tensor(Y_train), Tensor(X_test).reshape(-1, 1, 28, 28), Tensor(Y_test)
  else: return X_train, Y_train, X_test, Y_test

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

def fetch_cifar():
  X_train = Tensor.empty(50000, 3*32*32, device=f'disk:/tmp/cifar_train_x', dtype=dtypes.uint8)
  Y_train = Tensor.empty(50000, device=f'disk:/tmp/cifar_train_y', dtype=dtypes.int64)
  X_test = Tensor.empty(10000, 3*32*32, device=f'disk:/tmp/cifar_test_x', dtype=dtypes.uint8)
  Y_test = Tensor.empty(10000, device=f'disk:/tmp/cifar_test_y', dtype=dtypes.int64)

  if not os.path.isfile("/tmp/cifar_extracted"):
    def _load_disk_tensor(X, Y, db_list):
      idx = 0
      for db in db_list:
        x, y = db[b'data'], np.array(db[b'labels'])
        assert x.shape[0] == y.shape[0]
        X[idx:idx+x.shape[0]].assign(x)
        Y[idx:idx+x.shape[0]].assign(y)
        idx += x.shape[0]
      assert idx == X.shape[0] and X.shape[0] == Y.shape[0]

    print("downloading and extracting CIFAR...")
    fn = fetch('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    tt = tarfile.open(fn, mode='r:gz')
    _load_disk_tensor(X_train, Y_train, [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)])
    _load_disk_tensor(X_test, Y_test, [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")])
    open("/tmp/cifar_extracted", "wb").close()

  return X_train, Y_train, X_test, Y_test
