import unittest
from extra.models import resnet

class TestResnet(unittest.TestCase):
  def test_model_load(self):
    model = resnet.ResNet18()
    model.load_from_pretrained()

    model = resnet.ResNeXt50_32X4D()
    model.load_from_pretrained()

  def test_model_load_no_fc_layer(self):
    model = resnet.ResNet18(num_classes=None)
    model.load_from_pretrained()

    model = resnet.ResNeXt50_32X4D(num_classes=None)
    model.load_from_pretrained()


if __name__ == '__main__':
  unittest.main()