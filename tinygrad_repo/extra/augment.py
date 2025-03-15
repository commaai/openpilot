import numpy as np
from PIL import Image
from pathlib import Path
import sys
cwd = Path.cwd()
sys.path.append(cwd.as_posix())
sys.path.append((cwd / 'test').as_posix())
from extra.datasets import fetch_mnist
from tqdm import trange

def augment_img(X, rotate=10, px=3):
  Xaug = np.zeros_like(X)
  for i in trange(len(X)):
    im = Image.fromarray(X[i])
    im = im.rotate(np.random.randint(-rotate,rotate), resample=Image.BICUBIC)
    w, h = X.shape[1:]
    #upper left, lower left, lower right, upper right
    quad = np.random.randint(-px,px,size=(8)) + np.array([0,0,0,h,w,h,w,0])
    im = im.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
    Xaug[i] = im
  return Xaug

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
  X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
  X = np.vstack([X_train[:1]]*10+[X_train[1:2]]*10)
  fig, a = plt.subplots(2,len(X))
  Xaug = augment_img(X)
  for i in range(len(X)):
    a[0][i].imshow(X[i], cmap='gray')
    a[1][i].imshow(Xaug[i],cmap='gray')
    a[0][i].axis('off')
    a[1][i].axis('off')
  plt.show()

  #create some nice gifs for doc?!
  for i in range(10):
    im = Image.fromarray(X_train[7353+i])
    im_aug = [Image.fromarray(x) for x in augment_img(np.array([X_train[7353+i]]*100))]
    im.save(f"aug{i}.gif", save_all=True, append_images=im_aug, duration=100, loop=0)
