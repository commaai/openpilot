import random
import functools
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import signal, ndimage
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch

BASEDIR = Path(__file__).parent / "kits19" / "data"
TRAIN_PREPROCESSED_DIR =  Path(__file__).parent / "kits19" / "preprocessed" / "train"
VAL_PREPROCESSED_DIR =  Path(__file__).parent / "kits19" / "preprocessed" / "val"

@functools.cache
def get_train_files():
  return sorted([x for x in BASEDIR.iterdir() if x.stem.startswith("case") and int(x.stem.split("_")[-1]) < 210 and x not in get_val_files()])

@functools.cache
def get_val_files():
  data = fetch("https://raw.githubusercontent.com/mlcommons/training/master/retired_benchmarks/unet3d/pytorch/evaluation_cases.txt").read_text()
  return sorted([x for x in BASEDIR.iterdir() if x.stem.split("_")[-1] in data.split("\n")])

def load_pair(file_path):
  image, label = nib.load(file_path / "imaging.nii.gz"), nib.load(file_path / "segmentation.nii.gz")
  image_spacings = image.header["pixdim"][1:4].tolist()
  image, label = image.get_fdata().astype(np.float32), label.get_fdata().astype(np.uint8)
  image, label = np.expand_dims(image, 0), np.expand_dims(label, 0)
  return image, label, image_spacings

def resample3d(image, label, image_spacings, target_spacing=(1.6, 1.2, 1.2)):
  if image_spacings != target_spacing:
    spc_arr, targ_arr, shp_arr = np.array(image_spacings), np.array(target_spacing), np.array(image.shape[1:])
    new_shape = (spc_arr / targ_arr * shp_arr).astype(int).tolist()
    image = F.interpolate(torch.from_numpy(np.expand_dims(image, axis=0)), size=new_shape, mode="trilinear", align_corners=True)
    label = F.interpolate(torch.from_numpy(np.expand_dims(label, axis=0)), size=new_shape, mode="nearest")
    image = np.squeeze(image.numpy(), axis=0)
    label = np.squeeze(label.numpy(), axis=0)
  return image, label

def normal_intensity(image, min_clip=-79.0, max_clip=304.0, mean=101.0, std=76.9):
  image = np.clip(image, min_clip, max_clip)
  image = (image - mean) / std
  return image

def pad_to_min_shape(image, label, roi_shape=(128, 128, 128)):
  current_shape = image.shape[1:]
  bounds = [max(0, roi_shape[i] - current_shape[i]) for i in range(3)]
  paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)]
  image = np.pad(image, paddings, mode="edge")
  label = np.pad(label, paddings, mode="edge")
  return image, label

def preprocess(file_path):
  image, label, image_spacings = load_pair(file_path)
  image, label = resample3d(image, label, image_spacings)
  image = normal_intensity(image.copy())
  image, label = pad_to_min_shape(image, label)
  return image, label

def preprocess_dataset(filenames, preprocessed_dir, val):
  if not preprocessed_dir.is_dir(): os.makedirs(preprocessed_dir)
  for fn in tqdm(filenames, desc=f"preprocessing {'validation' if val else 'training'}"):
    case = os.path.basename(fn)
    image, label = preprocess(fn)
    image, label = image.astype(np.float32), label.astype(np.uint8)
    np.save(preprocessed_dir / f"{case}_x.npy", image, allow_pickle=False)
    np.save(preprocessed_dir / f"{case}_y.npy", label, allow_pickle=False)

def iterate(files, preprocessed_dir=None, val=True, shuffle=False, bs=1):
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for i in range(0, len(files), bs):
    samples = []
    for i in order[i:i+bs]:
      if preprocessed_dir is not None:
        x_cached_path, y_cached_path = preprocessed_dir / f"{os.path.basename(files[i])}_x.npy", preprocessed_dir / f"{os.path.basename(files[i])}_y.npy"
        if x_cached_path.exists() and y_cached_path.exists():
          samples += [(np.load(x_cached_path), np.load(y_cached_path))]
      else: samples += [preprocess(files[i])]
    X, Y = [x[0] for x in samples], [x[1] for x in samples]
    if val:
      yield X[0][None], Y[0]
    else:
      X_preprocessed, Y_preprocessed = [], []
      for x, y in zip(X, Y):
        x, y = rand_balanced_crop(x, y)
        x, y = rand_flip(x, y)
        x, y = x.astype(np.float32), y.astype(np.uint8)
        x = random_brightness_augmentation(x)
        x = gaussian_noise(x)
        X_preprocessed.append(x)
        Y_preprocessed.append(y)
      yield np.stack(X_preprocessed, axis=0), np.stack(Y_preprocessed, axis=0)

def gaussian_kernel(n, std):
  gaussian_1d = signal.windows.gaussian(n, std)
  gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
  gaussian_3d = np.outer(gaussian_2d, gaussian_1d)
  gaussian_3d = gaussian_3d.reshape(n, n, n)
  gaussian_3d = np.cbrt(gaussian_3d)
  gaussian_3d /= gaussian_3d.max()
  return gaussian_3d

def pad_input(volume, roi_shape, strides, padding_mode="constant", padding_val=-2.2, dim=3):
  bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
  bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i] for i in range(dim)]
  paddings = [bounds[2]//2, bounds[2]-bounds[2]//2, bounds[1]//2, bounds[1]-bounds[1]//2, bounds[0]//2, bounds[0]-bounds[0]//2, 0, 0, 0, 0]
  return F.pad(torch.from_numpy(volume), paddings, mode=padding_mode, value=padding_val).numpy(), paddings

def sliding_window_inference(model, inputs, labels, roi_shape=(128, 128, 128), overlap=0.5, gpus=None):
  from tinygrad.engine.jit import TinyJit
  mdl_run = TinyJit(lambda x: model(x).realize())
  image_shape, dim = list(inputs.shape[2:]), len(inputs.shape[2:])
  strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
  bounds = [image_shape[i] % strides[i] for i in range(dim)]
  bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
  inputs = inputs[
    ...,
    bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2),
    bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2),
    bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2),
  ]
  labels = labels[
    ...,
    bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2),
    bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2),
    bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2),
  ]
  inputs, paddings = pad_input(inputs, roi_shape, strides)
  padded_shape = inputs.shape[2:]
  size = [(inputs.shape[2:][i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
  result = np.zeros((1, 3, *padded_shape), dtype=np.float32)
  norm_map = np.zeros((1, 3, *padded_shape), dtype=np.float32)
  norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0])
  norm_patch = np.expand_dims(norm_patch, axis=0)
  for i in range(0, strides[0] * size[0], strides[0]):
    for j in range(0, strides[1] * size[1], strides[1]):
      for k in range(0, strides[2] * size[2], strides[2]):
        out = mdl_run(Tensor(inputs[..., i:roi_shape[0]+i,j:roi_shape[1]+j, k:roi_shape[2]+k], device=gpus)).numpy()
        result[..., i:roi_shape[0]+i, j:roi_shape[1]+j, k:roi_shape[2]+k] += out * norm_patch
        norm_map[..., i:roi_shape[0]+i, j:roi_shape[1]+j, k:roi_shape[2]+k] += norm_patch
  result /= norm_map
  result = result[..., paddings[4]:image_shape[0]+paddings[4], paddings[2]:image_shape[1]+paddings[2], paddings[0]:image_shape[2]+paddings[0]]
  return result, labels

def rand_flip(image, label, axis=(1, 2, 3)):
  prob = 1 / len(axis)
  for ax in axis:
    if random.random() < prob:
      image = np.flip(image, axis=ax).copy()
      label = np.flip(label, axis=ax).copy()
  return image, label

def random_brightness_augmentation(image, low=0.7, high=1.3, prob=0.1):
  if random.random() < prob:
    factor = np.random.uniform(low=low, high=high, size=1)
    image = (image * (1 + factor)).astype(image.dtype)
  return image

def gaussian_noise(image, mean=0.0, std=0.1, prob=0.1):
  if random.random() < prob:
    scale = np.random.uniform(low=0.0, high=std)
    noise = np.random.normal(loc=mean, scale=scale, size=image.shape).astype(image.dtype)
    image += noise
  return image

def _rand_foreg_cropb(image, label, patch_size):
  def adjust(foreg_slice, label, idx):
    diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
    sign = -1 if diff < 0 else 1
    diff = abs(diff)
    ladj = 0 if diff == 0 else random.randrange(diff)
    hadj = diff - ladj
    low = max(0, foreg_slice[idx].start - sign * ladj)
    high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
    diff = patch_size[idx - 1] - (high - low)
    if diff > 0 and low == 0: high += diff
    elif diff > 0: low -= diff
    return low, high

  cl = np.random.choice(np.unique(label[label > 0]))
  foreg_slices = ndimage.find_objects(ndimage.label(label==cl)[0])
  foreg_slices = [x for x in foreg_slices if x is not None]
  slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
  slice_idx = np.argsort(slice_volumes)[-2:]
  foreg_slices = [foreg_slices[i] for i in slice_idx]
  if not foreg_slices: return _rand_crop(image, label)
  foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
  low_x, high_x = adjust(foreg_slice, label, 1)
  low_y, high_y = adjust(foreg_slice, label, 2)
  low_z, high_z = adjust(foreg_slice, label, 3)
  image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
  label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
  return image, label

def _rand_crop(image, label, patch_size):
  ranges = [s - p for s, p in zip(image.shape[1:], patch_size)]
  cord = [0 if x == 0 else random.randrange(x) for x in ranges]
  low_x, high_x = cord[0], cord[0] + patch_size[0]
  low_y, high_y = cord[1], cord[1] + patch_size[1]
  low_z, high_z = cord[2], cord[2] + patch_size[2]
  image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
  label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
  return image, label

def rand_balanced_crop(image, label, patch_size=(128, 128, 128), oversampling=0.4):
  if random.random() < oversampling:
    image, label = _rand_foreg_cropb(image, label, patch_size)
  else:
    image, label = _rand_crop(image, label, patch_size)
  return image, label

if __name__ == "__main__":
  for X, Y in iterate(get_val_files()):
    print(X.shape, Y.shape)
