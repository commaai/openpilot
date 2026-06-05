import unittest

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import fetch, round_up
from extra.hevc.hevc import parse_hevc_file_headers, nv_gpu
from extra.hevc.decode import hevc_decode

class TestHevc(unittest.TestCase):
  def test_hevc_parser(self):
    url = "https://github.com/haraschax/filedump/raw/09a497959f7fa6fd8dba501a25f2cdb3a41ecb12/comma_video.hevc"
    dat = fetch(url, headers={"Range": f"bytes=0-{512<<10}"}).read_bytes()

    opaque, frame_info, w, h, luma_w, luma_h, chroma_off = parse_hevc_file_headers(dat, device=Device.DEFAULT)

    def _test_common(frame, bts):
      self.assertEqual(frame0.pic_width_in_luma_samples, 1952)
      self.assertEqual(frame0.pic_height_in_luma_samples, 1216)
      self.assertEqual(frame0.chroma_format_idc, 1)
      self.assertEqual(frame0.bit_depth_luma, 8)
      self.assertEqual(frame0.bit_depth_chroma, 8)
      self.assertEqual(frame0.log2_min_luma_coding_block_size, 3)
      self.assertEqual(frame0.log2_max_luma_coding_block_size, 5)
      self.assertEqual(frame0.log2_min_transform_block_size, 2)
      self.assertEqual(frame0.log2_max_transform_block_size, 5)
      self.assertEqual(frame0.num_tile_columns, 3)
      self.assertEqual(frame0.num_tile_rows, 1)
      self.assertEqual(frame0.colMvBuffersize, 589)
      self.assertEqual(frame0.HevcSaoBufferOffset, 2888)
      self.assertEqual(frame0.HevcBsdCtrlOffset, 25992)
      self.assertEqual(frame0.v1.hevc_main10_444_ext.HevcFltAboveOffset, 26714)
      self.assertEqual(frame0.v1.hevc_main10_444_ext.HevcSaoAboveOffset, 36214)

      # tiles
      self.assertEqual(bytes(bts[0x200:0x210]), b'\x18\x00&\x00\x18\x00&\x00\r\x00&\x00\x00\x00\x00\x00')

    frame0 = nv_gpu.nvdec_hevc_pic_s.from_buffer(opaque[0].data())
    _test_common(frame0, opaque[0].data())
    self.assertEqual(frame0.stream_len, 148063)
    self.assertEqual(frame0.IDR_picture_flag, 1)
    self.assertEqual(frame0.RAP_picture_flag, 1)
    self.assertEqual(frame0.sw_hdr_skip_length, 0)
    self.assertEqual(frame0.num_ref_frames, 0)

    frame1 = nv_gpu.nvdec_hevc_pic_s.from_buffer(opaque[1].data())
    _test_common(frame1, opaque[1].data())
    self.assertEqual(frame1.stream_len, 57110)
    self.assertEqual(frame1.IDR_picture_flag, 0)
    self.assertEqual(frame1.RAP_picture_flag, 0)
    self.assertEqual(frame1.sw_hdr_skip_length, 9)
    self.assertEqual(frame1.num_ref_frames, 1)
    self.assertEqual(list(frame1.initreflistidxl0), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(list(frame1.initreflistidxl1), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(list(frame1.RefDiffPicOrderCnts), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    frame3 = nv_gpu.nvdec_hevc_pic_s.from_buffer(opaque[3].data())
    _test_common(frame3, opaque[3].data())
    self.assertEqual(frame3.stream_len, 47036)
    self.assertEqual(frame3.IDR_picture_flag, 0)
    self.assertEqual(frame3.RAP_picture_flag, 0)
    self.assertEqual(frame3.sw_hdr_skip_length, 9)
    self.assertEqual(frame3.num_ref_frames, 1)
    self.assertEqual(list(frame3.initreflistidxl0), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(list(frame3.initreflistidxl1), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(list(frame3.RefDiffPicOrderCnts), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  @unittest.skipUnless(Device.DEFAULT == "NV", "NV only")
  def test_hevc_decode(self):
    url = "https://github.com/haraschax/filedump/raw/09a497959f7fa6fd8dba501a25f2cdb3a41ecb12/comma_video.hevc"
    dat = fetch(url, headers={"Range": f"bytes=0-{512<<10}"}).read_bytes()

    opaque, frame_info, w, h, luma_w, luma_h, chroma_off = parse_hevc_file_headers(dat)
    frame_info = frame_info[:4]
    out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)

    hevc_tensor = Tensor(dat, device="NV")
    opaque_nv = opaque.to("NV").contiguous().realize()

    frames = list(hevc_decode(hevc_tensor, opaque_nv, frame_info, luma_h, luma_w))
    Device.default.synchronize()
    self.assertEqual(len(frames), 4)
    for f in frames:
      self.assertEqual(f.shape, out_image_size)
      self.assertEqual(f.dtype, dtypes.uint8)
      self.assertEqual(f.device, "NV")

if __name__ == "__main__":
  unittest.main()
