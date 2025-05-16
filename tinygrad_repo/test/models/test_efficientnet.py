import ast
import pathlib
import unittest

import numpy as np
from PIL import Image

from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from extra.models.efficientnet import EfficientNet
from extra.models.vit import ViT
from extra.models.resnet import ResNet50

def _load_labels():
  labels_filename = pathlib.Path(__file__).parent / 'efficientnet/imagenet1000_clsidx_to_labels.txt'
  return ast.literal_eval(labels_filename.read_text())

_LABELS = _load_labels()

def preprocess(img, new=False):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0, x0 =(np.asarray(img.shape)[:2] - 224) // 2
  img = img[y0: y0 + 224, x0: x0 + 224]

  # low level preprocess
  if new:
    img = img.astype(np.float32)
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    img = img[None]
  else:
    img = np.moveaxis(img, [2, 0, 1], [0, 1, 2])
    img = img.astype(np.float32)[:3].reshape(1, 3, 224, 224)
    img /= 255.0
    img -= np.array([0.485, 0.456, 0.406]).reshape((1, -1, 1, 1))
    img /= np.array([0.229, 0.224, 0.225]).reshape((1, -1, 1, 1))
  return img


def _infer(model: EfficientNet, img, bs=1):
  old_training = Tensor.training
  Tensor.training = False
  img = preprocess(img)
  # run the net
  if bs > 1: img = img.repeat(bs, axis=0)
  out = model.forward(Tensor(img))
  Tensor.training = old_training
  return _LABELS[np.argmax(out.numpy()[0])]

chicken_img = Image.open(pathlib.Path(__file__).parent / 'efficientnet/Chicken.jpg')
car_img = Image.open(pathlib.Path(__file__).parent / 'efficientnet/car.jpg')

class TestEfficientNet(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = EfficientNet(number=getenv("NUM"))
    cls.model.load_from_pretrained()

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_chicken(self):
    label = _infer(self.model, chicken_img)
    self.assertEqual(label, "hen")

  def test_chicken_bigbatch(self):
    label = _infer(self.model, chicken_img, 2)
    self.assertEqual(label, "hen")

  def test_car(self):
    label = _infer(self.model, car_img)
    self.assertEqual(label, "sports car, sport car")

class TestViT(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = ViT()
    cls.model.load_from_pretrained()

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_chicken(self):
    label = _infer(self.model, chicken_img)
    self.assertEqual(label, "cock")

  def test_car(self):
    label = _infer(self.model, car_img)
    self.assertEqual(label, "racer, race car, racing car")

class TestResNet(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = ResNet50()
    cls.model.load_from_pretrained()

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_chicken(self):
    label = _infer(self.model, chicken_img)
    self.assertEqual(label, "hen")

  def test_car(self):
    label = _infer(self.model, car_img)
    self.assertEqual(label, "sports car, sport car")

if __name__ == '__main__':
  unittest.main()
