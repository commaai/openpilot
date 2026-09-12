import dataclasses

from libdatachannel import Description


@dataclasses.dataclass
class StreamingMediaInfo:
  n_expected_camera_tracks: int
  expected_audio_track: bool
  incoming_audio_track: bool
  incoming_datachannel: bool


def parse_info_from_offer(sdp: str) -> StreamingMediaInfo:
  """
  helper function to parse info about outgoing and incoming streams from an offer sdp
  """
  desc = Description(sdp, Description.Type.Offer)
  n_video = 0
  expected_audio_track = False
  incoming_audio_track = False
  incoming_datachannel = desc.has_application()

  for i in range(desc.media_count()):
    media = desc.media(i)
    if media is None:
      continue
    direction = media.direction()
    if media.type() == "video" and direction in (Description.Direction.RecvOnly, Description.Direction.SendRecv):
      n_video += 1
    elif media.type() == "audio":
      if direction in (Description.Direction.RecvOnly, Description.Direction.SendRecv):
        expected_audio_track = True
      if direction in (Description.Direction.SendOnly, Description.Direction.SendRecv):
        incoming_audio_track = True

  return StreamingMediaInfo(n_video, expected_audio_track, incoming_audio_track, incoming_datachannel)
