from openpilot.system.loggerd.video_writer import MpegTsMuxer


def test_mpeg_ts_packets_are_fixed_size():
  output = []
  muxer = MpegTsMuxer(output.append, 20)
  muxer.write(b"\0\0\1\x65", codecconfig=False, keyframe=True)

  data = b"".join(output)
  assert len(data) % 188 == 0
  assert all(data[offset] == 0x47 for offset in range(0, len(data), 188))
