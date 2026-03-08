import unittest
from extra.models import resnet
from tinygrad import dtypes
from tinygrad.device import is_dtype_supported

# pretrained weights contain num_batches_tracked as int64
@unittest.skipUnless(is_dtype_supported(dtypes.int64), "need int64 support")
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