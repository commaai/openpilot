#!/usr/bin/env python
# flake8: noqa

import re
import binascii

from tools.lib.mkvparse import mkvparse
from tools.lib.mkvparse import mkvgen
from tools.lib.mkvparse.mkvgen import ben, ebml_element, ebml_encode_number

class MatroskaIndex(mkvparse.MatroskaHandler):
  # def __init__(self, banlist, nocluster_mode):
  #   pass
  def __init__(self):
    self.frameindex = []

  def tracks_available(self):
    _, self.config_record = self.tracks[1]['CodecPrivate']  # pylint: disable=no-member

  def frame(self, track_id, timestamp, pos, length, more_laced_frames, duration,
        keyframe, invisible, discardable):
    self.frameindex.append((pos, length, keyframe))



def mkvindex(f):
  handler = MatroskaIndex()
  mkvparse.mkvparse(f, handler)
  return handler.config_record, handler.frameindex


def simple_gen(of, config_record, w, h, framedata):
  mkvgen.write_ebml_header(of, "matroska", 2, 2)
  mkvgen.write_infinite_segment_header(of)

  of.write(ebml_element(0x1654AE6B, "" # Tracks
    + ebml_element(0xAE, "" # TrackEntry
      + ebml_element(0xD7, ben(1)) # TrackNumber
      + ebml_element(0x73C5, ben(1)) # TrackUID
      + ebml_element(0x83, ben(1)) # TrackType = video track
      + ebml_element(0x86, "V_MS/VFW/FOURCC") # CodecID
      + ebml_element(0xE0, "" # Video
        + ebml_element(0xB0, ben(w)) # PixelWidth
        + ebml_element(0xBA, ben(h)) # PixelHeight
        )
      + ebml_element(0x63A2, config_record) # CodecPrivate (ffv1 configuration record)
      )
    ))

  blocks = []
  for fd in framedata:
    blocks.append(
      ebml_element(0xA3, "" # SimpleBlock
        + ebml_encode_number(1) # track number
        + chr(0x00) + chr(0x00) # timecode, relative to Cluster timecode, sint16, in milliseconds
        + chr(0x80) # flags (keyframe)
        + fd
        )
      )

  of.write(ebml_element(0x1F43B675, "" # Cluster
    + ebml_element(0xE7, ben(0)) # TimeCode, uint, milliseconds
    # + ebml_element(0xA7, ben(0)) # Position, uint
    + ''.join(blocks)))
