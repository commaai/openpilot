# for imagenet download prepare.sh and run it
import glob, random, json, math
import numpy as np
from PIL import Image
import functools, pathlib
from tinygrad.helpers import diskcache, getenv

@functools.lru_cache(None)
def get_imagenet_categories():
  ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
  return {v[0]: int(k) for k,v in ci.items()}

if getenv("MNISTMOCK"):
  BASEDIR = pathlib.Path(__file__).parent / "mnist"

  @functools.lru_cache(None)
  def get_train_files():
    if not BASEDIR.exists():
      from extra.datasets.fake_imagenet_from_mnist import create_fake_mnist_imagenet
      create_fake_mnist_imagenet(BASEDIR)

    if not (files:=glob.glob(p:=str(BASEDIR / "train/*/*"))): raise FileNotFoundError(f"No training files in {p}")
    return files
else:
  BASEDIR = pathlib.Path(__file__).parent / "imagenet"

  @diskcache
  def get_train_files():
    if not (files:=glob.glob(p:=str(BASEDIR / "train/*/*"))): raise FileNotFoundError(f"No training files in {p}")
    return files

@functools.lru_cache(None)
def get_val_files():
  if not (files:=glob.glob(p:=str(BASEDIR / "val/*/*"))): raise FileNotFoundError(f"No validation files in {p}")
  return files

def image_resize(img, size, interpolation):
  w, h = img.size
  w_new = int((w / h) * size) if w > h else size
  h_new = int((h / w) * size) if h > w else size
  return img.resize([w_new, h_new], interpolation)

def rand_flip(img):
  if random.random() < 0.5:
    img = np.flip(img, axis=1).copy()
  return img

def center_crop(img):
  rescale = min(img.size) / 256
  crop_left = (img.width - 224 * rescale) / 2.0
  crop_top = (img.height - 224 * rescale) / 2.0
  img = img.resize((224, 224), Image.BILINEAR, box=(crop_left, crop_top, crop_left + 224 * rescale, crop_top + 224 * rescale))
  return img

# we don't use supplied imagenet bounding boxes, so scale min is just min_object_covered
# https://github.com/tensorflow/tensorflow/blob/e193d8ea7776ef5c6f5d769b6fb9c070213e737a/tensorflow/core/kernels/image/sample_distorted_bounding_box_op.cc
def random_resized_crop(img, size, scale=(0.10, 1.0), ratio=(3/4, 4/3)):
  w, h = img.size
  area = w * h

  # Crop
  random_solution_found = False
  for _ in range(100):
    aspect_ratio = random.uniform(ratio[0], ratio[1])
    max_scale = min(min(w * aspect_ratio / h, h / aspect_ratio / w), scale[1])
    target_area = area * random.uniform(scale[0], max_scale)

    w_new = int(round(math.sqrt(target_area * aspect_ratio)))
    h_new = int(round(math.sqrt(target_area / aspect_ratio)))

    if 0 < w_new <= w and 0 < h_new <= h:
      crop_left = random.randint(0, w - w_new)
      crop_top = random.randint(0, h - h_new)

      img = img.crop((crop_left, crop_top, crop_left + w_new, crop_top + h_new))
      random_solution_found = True
      break

  if not random_solution_found:
    # Center crop
    img = center_crop(img)
  else:
    # Resize
    img = img.resize([size, size], Image.BILINEAR)

  return img

def preprocess_train(img):
  img = random_resized_crop(img, 224)
  img = rand_flip(np.array(img))
  return img
