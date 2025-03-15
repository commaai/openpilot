import av

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    self.container = av.open(camera_id)
    assert self.container.streams.video, f"Can't open video stream for camera {camera_id}"
    self.video_stream = self.container.streams.video[0]
    self.W = self.video_stream.codec_context.width
    self.H = self.video_stream.codec_context.height

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    for frame in self.container.decode(self.video_stream):
      img = frame.to_rgb().to_ndarray()[:,:, ::-1] # convert to bgr24
      yuv = Camera.bgr2nv12(img)
      yield yuv.data.tobytes()
    self.container.close()
