import json
import os

from openpilot.cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.events import EVENTS, EVENT_NAME


DEBUG_EVENT_PATH = "/dev/shm/selfdrived_event"
MAX_DEBUG_EVENT_FILE_SIZE = 4096

EventName = log.OnroadEvent.EventName


def parse_debug_event(raw: str) -> int:
  command = json.loads(raw)
  if not isinstance(command, dict) or not isinstance(command.get("event"), str):
    raise ValueError("command must be a JSON object containing a string 'event' field")

  event_name = command["event"]
  event = EventName.schema.enumerants.get(event_name)
  if event is None or event not in EVENTS:
    raise ValueError(f"unknown event: {event_name}")
  return event


class DebugEventReader:
  def __init__(self, path: str = DEBUG_EVENT_PATH):
    self.path = path
    self._signature: tuple[int, int, int] | None = None
    self._event: int | None = None
    self._stat_error: str | None = None

  def read(self) -> int | None:
    try:
      stat = os.stat(self.path)
    except FileNotFoundError:
      self._signature = None
      self._event = None
      self._stat_error = None
      return None
    except OSError as e:
      error = str(e)
      if error != self._stat_error:
        cloudlog.warning(f"failed to stat debug event file {self.path}: {e}")
      self._signature = None
      self._event = None
      self._stat_error = error
      return None

    self._stat_error = None
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature == self._signature:
      return self._event

    self._signature = signature
    try:
      if stat.st_size > MAX_DEBUG_EVENT_FILE_SIZE:
        raise ValueError(f"command is larger than {MAX_DEBUG_EVENT_FILE_SIZE} bytes")
      with open(self.path) as f:
        self._event = parse_debug_event(f.read(MAX_DEBUG_EVENT_FILE_SIZE + 1))
      cloudlog.info(f"debug event injected: {EVENT_NAME[self._event]}")
    except (OSError, ValueError) as e:
      cloudlog.warning(f"invalid debug event file {self.path}: {e}")
      self._event = None

    return self._event
