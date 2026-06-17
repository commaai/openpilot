from dataclasses import dataclass, field

@dataclass
class StreamRequestBody:
  sdp: str
  init_camera: str
  enabled: bool
  bridge_services_in: list[str] = field(default_factory=list)
  bridge_services_out: list[str] = field(default_factory=list)
