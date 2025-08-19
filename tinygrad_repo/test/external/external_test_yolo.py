import unittest
from pathlib import Path

import cv2

from examples.yolov3 import Darknet, infer, show_labels
from tinygrad.helpers import fetch

chicken_img = cv2.imread(str(Path(__file__).parent.parent / 'models/efficientnet/Chicken.jpg'))
car_img = cv2.imread(str(Path(__file__).parent.parent / 'models/efficientnet/car.jpg'))

class TestYOLO(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = Darknet(fetch("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg").read_bytes())
    print("Loading weights file (237MB). This might take a while…")
    cls.model.load_weights("https://pjreddie.com/media/files/yolov3.weights")

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_chicken(self):
    labels = show_labels(infer(self.model, chicken_img), confidence=0.56)
    self.assertEqual(labels, ["bird"])

  def test_car(self):
    labels = show_labels(infer(self.model, car_img))
    self.assertEqual(labels, ["car"])

if __name__ == '__main__':
  unittest.main()
