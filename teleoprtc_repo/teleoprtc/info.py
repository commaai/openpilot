import dataclasses

import aiortc


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
  desc = aiortc.sdp.SessionDescription.parse(sdp)
  audio_tracks = [m for m in desc.media if m.kind == "audio"]
  video_tracks = [m for m in desc.media if m.kind == "video" and m.direction in ["recvonly", "sendrecv"]]
  application_tracks = [m for m in desc.media if m.kind == "application"]
  has_incoming_audio_track = next((t for t in audio_tracks if t.direction in ["sendonly", "sendrecv"]), None) is not None
  has_incoming_datachannel = len(application_tracks) > 0
  expects_outgoing_audio_track = next((t for t in audio_tracks if t.direction in ["recvonly", "sendrecv"]), None) is not None

  return StreamingMediaInfo(len(video_tracks), expects_outgoing_audio_track, has_incoming_audio_track, has_incoming_datachannel)
