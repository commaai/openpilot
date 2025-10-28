#!/usr/bin/env python
import unittest
from tinygrad.device import Device, BufferSpec
from tinygrad.dtype import dtypes

@unittest.skipUnless(Device.DEFAULT == "QCOM", "QCOM device required to run")
class TestQcom(unittest.TestCase):
  def test_image_pitch(self):
    dev = Device["QCOM"]

    def __validate(imgdt, expected_pitch):
      img = dev.allocator.alloc(imgdt.shape[0] * imgdt.shape[1] * 16, options:=BufferSpec(image=imgdt))
      pitch = (img.descriptor[2] & 0x1fffff80) >> 7
      assert pitch == expected_pitch, f"Failed pitch for image: {imgdt}. Got 0x{pitch:X}, expected 0x{expected_pitch:X}"
      dev.allocator.free(img, imgdt.shape[0] * imgdt.shape[1] * 16, options)

    __validate(dtypes.imageh((1, 201)), 0x680)
    __validate(dtypes.imageh((16, 216)), 0x700)
    __validate(dtypes.imageh((16, 9)), 0x80)
    __validate(dtypes.imageh((48, 64)), 0x200)
    __validate(dtypes.imageh((32, 128)), 0x400)
    __validate(dtypes.imageh((96, 128)), 0x400)
    __validate(dtypes.imageh((64, 256)), 0x840)
    __validate(dtypes.imageh((64, 9)), 0x80)
    __validate(dtypes.imageh((192, 256)), 0x840)
    __validate(dtypes.imageh((64, 768)), 0x1840)
    __validate(dtypes.imageh((256, 49)), 0x1C0)
    __validate(dtypes.imageh((128, 9)), 0x80)
    __validate(dtypes.imageh((16, 1024)), 0x2080)
    __validate(dtypes.imageh((64, 512)), 0x1040)
    __validate(dtypes.imageh((16, 512)), 0x1080)
    __validate(dtypes.imageh((132, 64)), 0x200)
    __validate(dtypes.imageh((4, 512)), 0x1200)
    __validate(dtypes.imageh((8, 512)), 0x1100)
    __validate(dtypes.imageh((128, 128)), 0x400)
    __validate(dtypes.imageh((32, 512)), 0x1040)
    __validate(dtypes.imageh((26, 64)), 0x200)
    __validate(dtypes.imageh((32, 516)), 0x1040)
    __validate(dtypes.imageh((32, 1024)), 0x2040)
    __validate(dtypes.imageh((16, 2048)), 0x4080)
    __validate(dtypes.imageh((8, 2048)), 0x4100)
    __validate(dtypes.imageh((4, 4096)), 0x8200)

    __validate(dtypes.imagef((16, 49)), 0x380)
    __validate(dtypes.imagef((16, 1024)), 0x4080)
    __validate(dtypes.imagef((256, 64)), 0x400)
    __validate(dtypes.imagef((64, 512)), 0x2040)
    __validate(dtypes.imagef((16, 512)), 0x2080)
    __validate(dtypes.imagef((132, 64)), 0x400)
    __validate(dtypes.imagef((4, 512)), 0x2200)
    __validate(dtypes.imagef((4, 16)), 0x200)
    __validate(dtypes.imagef((2, 16)), 0x400)
    __validate(dtypes.imagef((8, 512)), 0x2100)
    __validate(dtypes.imagef((12, 64)), 0x400)
    __validate(dtypes.imagef((3, 32)), 0x400)
    __validate(dtypes.imagef((128, 128)), 0x840)
    __validate(dtypes.imagef((32, 512)), 0x2040)
    __validate(dtypes.imagef((8, 3072)), 0xC100)
    __validate(dtypes.imagef((4, 2048)), 0x8200)
    __validate(dtypes.imagef((4, 1024)), 0x4200)
    __validate(dtypes.imagef((4, 4096)), 0x10200)
    __validate(dtypes.imagef((10, 384)), 0x1900)
    __validate(dtypes.imagef((24, 64)), 0x400)
    __validate(dtypes.imagef((128, 12)), 0xC0)
    __validate(dtypes.imagef((10, 24)), 0x200)
    __validate(dtypes.imagef((1, 129)), 0x840)
    __validate(dtypes.imagef((1, 32)), 0x200)
    __validate(dtypes.imagef((1, 64)), 0x400)
    __validate(dtypes.imagef((1, 1239)), 0x4D80)
    __validate(dtypes.imagef((1, 1)), 0x40)

if __name__ == "__main__":
  unittest.main()
