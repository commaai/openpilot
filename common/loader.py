import os
import struct
import bisect
import numpy as np
import _io

import capnp
from cereal import log as capnp_log

class RawData():
  def __init__(self, f):
    self.f = _io.FileIO(f, 'rb')
    self.lenn = struct.unpack("I", self.f.read(4))[0]
    self.count = os.path.getsize(f) / (self.lenn+4)

  def read(self, i):
    self.f.seek((self.lenn+4)*i + 4)
    return self.f.read(self.lenn)

def yuv420_to_rgb(raw, image_dim=None, swizzled=False):
  def expand(x):
    x = np.repeat(x, 2, axis=0)
    return np.repeat(x, 2, axis=1)

  if image_dim is None:
    image_dim = (raw.shape[1]*2, raw.shape[2]*2)
    swizzled = True

  if not swizzled:
    img_data = np.array(raw, copy=False, dtype=np.uint8)
    uv_len = (image_dim[0]/2)*(image_dim[1]/2)
    img_data_u = expand(img_data[image_dim[0]*image_dim[1]: \
                                 image_dim[0]*image_dim[1]+uv_len]. \
                                 reshape(image_dim[0]/2, image_dim[1]/2))

    img_data_v = expand(img_data[image_dim[0]*image_dim[1]+uv_len: \
                                 image_dim[0]*image_dim[1]+2*uv_len]. \
                                 reshape(image_dim[0]/2, image_dim[1]/2))
    img_data_y = img_data[0:image_dim[0]*image_dim[1]].reshape(image_dim)
  else:
    img_data_y = np.zeros(image_dim, dtype=np.uint8)
    img_data_y[0::2, 0::2] = raw[0]
    img_data_y[1::2, 0::2] = raw[1]
    img_data_y[0::2, 1::2] = raw[2]
    img_data_y[1::2, 1::2] = raw[3]
    img_data_u = expand(raw[4])
    img_data_v = expand(raw[5])

  yuv = np.stack((img_data_y, img_data_u, img_data_v)).swapaxes(0,2).swapaxes(0,1)
  yuv = yuv.astype(np.int16)

  # http://maxsharabayko.blogspot.com/2016/01/fast-yuv-to-rgb-conversion-in-python-3.html
  # according to ITU-R BT.709
  yuv[:,:, 0] = yuv[:,:, 0].clip(16, 235).astype(yuv.dtype) - 16
  yuv[:,:,1:] = yuv[:,:,1:].clip(16, 240).astype(yuv.dtype) - 128

  A = np.array([[1.164,  0.000,  1.793],
                [1.164, -0.213, -0.533],
                [1.164,  2.112,  0.000]])

  # our result
  img = np.dot(yuv, A.T).clip(0, 255).astype('uint8')
  return img


class YuvData():
  def __init__(self, f, dim=(160,320)):
    self.f = _io.FileIO(f, 'rb')
    self.image_dim = dim
    self.image_size = self.image_dim[0]/2 * self.image_dim[1]/2 * 6
    self.count = os.path.getsize(f) / self.image_size

  def read_frame(self, frame):
    self.f.seek(self.image_size*frame)
    raw = self.f.read(self.image_size)
    return raw

  def read_frames(self, range_start, range_len):
    self.f.seek(self.image_size*range_start)
    raw = self.f.read(self.image_size*range_len)
    return raw

  def read_frames_into(self, range_start, buf):
    self.f.seek(self.image_size*range_start)
    return self.f.readinto(buf)

  def read(self, frame):
    return yuv420_to_rgb(self.read_frame(frame), self.image_dim)

  def close(self):
    self.f.close()

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.close()



class OneReader():
  def __init__(self, base_path, goofy=False, segment_range=None):
    self.base_path = base_path

    route_name = os.path.basename(base_path)

    self.rcamera_size = (304, 560)

    if segment_range is None:
      parent_path = os.path.dirname(base_path)
      self.segment_nums = []
      for p in os.listdir(parent_path):
        if not p.startswith(route_name+"--"):
          continue
        self.segment_nums.append(int(p.rsplit("--", 1)[-1]))
      if not self.segment_nums:
        raise Exception("no route segments found")
      self.segment_nums.sort()
      self.segment_range = (self.segment_nums[0], self.segment_nums[-1])
    else:
      self.segment_range = segment_range
      self.segment_nums = range(segment_range[0], segment_range[1]+1)
      for i in self.segment_nums:
        if not os.path.exists(base_path+"--"+str(i)):
          raise Exception("missing segment in provided range")

    # goofy data is broken with discontinuous logs
    if goofy and (self.segment_range[0] != 0
        or self.segment_nums != range(self.segment_range[0], self.segment_range[1]+1)):
      raise Exception("goofy data needs all the segments for a route")

    self.cur_seg = None
    self.cur_seg_f = None

    # index the frames
    print("indexing frames {}...".format(self.segment_nums))

    self.rcamera_encode_map = {} # frame_id -> (segment num, segment id, frame_time)
    last_frame_id = -1

    if goofy:
      # goofy is goofy

      frame_size = self.rcamera_size[0]*self.rcamera_size[1]*3/2

      # find the encode id ranges for each segment by using the rcamera file size
      segment_encode_ids = []
      cur_encode_id = 0
      for n in self.segment_nums:
        camera_path = os.path.join(self.seg_path(n), "rcamera")
        if not os.path.exists(camera_path):
          # for goofy, missing camera files means a bad route
          raise Exception("Missing camera file {}".format(camera_path))
        camera_size = os.path.getsize(camera_path)
        assert (camera_size % frame_size) == 0

        num_frames = camera_size / frame_size
        segment_encode_ids.append(cur_encode_id)
        cur_encode_id += num_frames

      last_encode_id = -1
      # use the segment encode id map and frame events to build the frame index
      for n in self.segment_nums:
        log_path = os.path.join(self.seg_path(n), "rlog")
        if os.path.exists(log_path):
          with open(log_path, "rb") as f:
            for evt in capnp_log.Event.read_multiple(f):
              if evt.which() == 'frame':

                if evt.frame.frameId < last_frame_id:
                  # a non-increasing frame id is bad route (eg visiond was restarted)
                  raise Exception("non-increasing frame id")
                last_frame_id = evt.frame.frameId

                seg_i = bisect.bisect_right(segment_encode_ids, evt.frame.encodeId)-1
                assert seg_i >= 0
                seg_num = self.segment_nums[seg_i]
                seg_id = evt.frame.encodeId-segment_encode_ids[seg_i]
                frame_time = evt.logMonoTime / 1.0e9

                self.rcamera_encode_map[evt.frame.frameId] = (seg_num, seg_id,
                                                              frame_time)

                last_encode_id = evt.frame.encodeId

      if last_encode_id-cur_encode_id > 10:
        # too many missing frames is a bad route (eg route from before encoder rotating worked)
        raise Exception("goofy route is missing frames: {}, {}".format(
          last_encode_id, cur_encode_id))

    else:
      # for harry data, build the index from encodeIdx events
      for n in self.segment_nums:
        log_path = os.path.join(self.seg_path(n), "rlog")
        if os.path.exists(log_path):
          with open(log_path, "rb") as f:
            for evt in capnp_log.Event.read_multiple(f):
              if evt.which() == 'encodeIdx' and evt.encodeIdx.type == 'bigBoxLossless':
                frame_time = evt.logMonoTime / 1.0e9
                self.rcamera_encode_map[evt.encodeIdx.frameId] = (
                  evt.encodeIdx.segmentNum, evt.encodeIdx.segmentId,
                  frame_time)

    print("done")

    # read the first event to find the start time
    self.reset_to_seg(self.segment_range[0])
    for evt in self.events():
      if evt.which() != 'initData':
        self.start_mono = evt.logMonoTime
        break
    self.reset_to_seg(self.segment_range[0])


  def seg_path(self, num):
    return self.base_path+"--"+str(num)

  def reset_to_seg(self, seg):
    self.cur_seg = seg
    if self.cur_seg_f:
      self.cur_seg_f.close()
      self.cur_seg_f = None

  def seek_ts(self, ts):
    seek_seg = int(ts/60)
    if seek_seg < self.segment_range[0] or seek_seg > self.segment_range[1]:
      raise ValueError

    self.reset_to_seg(seek_seg)
    target_mono = self.start_mono + int(ts*1e9)
    for evt in self.events():
      if evt.logMonoTime >= target_mono:
        break

  def read_event(self):
    while True:
      if self.cur_seg > self.segment_range[1]:
        return None
      if self.cur_seg_f is None:
        log_path = os.path.join(self.seg_path(self.cur_seg), "rlog")
        if not os.path.exists(log_path):
          print("missing log file!", log_path)
          self.cur_seg += 1
          continue
        self.cur_seg_f = open(log_path, "rb")

      try:
        return capnp_log.Event.read(self.cur_seg_f)
      except capnp.lib.capnp.KjException as e:
        if 'EOF' in str(e): # dumb, but pycapnp does this too
          self.cur_seg_f.close()
          self.cur_seg_f = None
          self.cur_seg += 1
        else:
          raise

  def events(self):
    while True:
      r = self.read_event()
      if r is None:
        break
      yield r

  def read_frame(self, frame_id):
    encode_idx = self.rcamera_encode_map.get(frame_id)
    if encode_idx is None:
      return None

    seg_num, seg_id, _ = encode_idx
    camera_path = os.path.join(self.seg_path(seg_num), "rcamera")
    if not os.path.exists(camera_path):
      return None
    with YuvData(camera_path, self.rcamera_size) as data:
      return data.read_frame(seg_id)

  def close(self):
    if self.cur_seg_f is not None:
      self.cur_seg_f.close()

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.close()
