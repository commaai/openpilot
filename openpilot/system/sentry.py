"""Install exception handler for process crash."""
import json
import os
import platform
import sys
import threading
import traceback
import uuid
from datetime import datetime, UTC
from enum import Enum
from urllib.parse import urlparse

import requests

from openpilot.common.params import Params
from openpilot.system.athena.registration import is_registered_device
from openpilot.common.hardware import HARDWARE, PC
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata, get_version


class SentryProject(Enum):
  # python project
  SELFDRIVE = "https://6f3c7076c1e14b2aa10f5dde6dda0cc4@o33823.ingest.sentry.io/77924"
  # native project
  SELFDRIVE_NATIVE = "https://3e4b586ed21a4479ad5d85083b639bc6@o33823.ingest.sentry.io/157615"


_endpoint: str | None = None
_release: str = ""
_environment: str = ""
_user: dict = {}
_tags: dict = {}
_local = threading.local()


def _set_tags(tags: dict) -> None:
  _local.tags = tags


def set_tag(key: str, value: str) -> None:
  tags = getattr(_local, "tags", None)
  if tags is not None:
    tags[key] = value
  else:
    _tags[key] = value


def _post_event(event: dict) -> None:
  if _endpoint is None:
    return
  payload = json.dumps(event).encode()
  envelope = json.dumps({"event_id": event["event_id"], "sent_at": datetime.now(tz=UTC).isoformat()}) + "\n"
  envelope += json.dumps({"type": "event", "length": len(payload)}) + "\n"
  requests.post(_endpoint, data=envelope.encode() + payload, timeout=10).raise_for_status()


def _frames_from_tb(tb) -> list:
  frames = []
  for f in traceback.extract_tb(tb):
    frames.append({
      "filename": os.path.relpath(f.filename),
      "function": f.name,
      "lineno": f.lineno,
      "context_line": f.line,
      "in_app": True,
    })
  frames.reverse()  # sentry expects oldest frame first
  return frames


def _base_event(project: SentryProject) -> dict:
  tags = dict(_tags)
  tags.update(getattr(_local, "tags", {}))
  return {
    "event_id": uuid.uuid4().hex,
    "timestamp": datetime.now(tz=UTC).isoformat(),
    "platform": "python" if project == SentryProject.SELFDRIVE else "native",
    "release": _release,
    "environment": _environment,
    "tags": {k: str(v) for k, v in tags.items()},
    "user": _user,
    "contexts": {"runtime": {"name": "CPython", "version": platform.python_version()}},
    "sdk": {"name": "openpilot.sentry", "version": get_version()},
  }


def report_tombstone(fn: str, message: str, contents: str) -> None:
  cloudlog.error({'tombstone': message})

  try:
    event = _base_event(SentryProject.SELFDRIVE_NATIVE)
    event["level"] = "error"
    event["message"] = message
    event["extra"] = {"tombstone_fn": fn, "tombstone": contents}
    _post_event(event)
  except Exception:
    cloudlog.exception("sentry tombstone exception")


def capture_exception(exc_info=None, *args, **kwargs) -> None:
  cloudlog.error("crash", exc_info=exc_info or True)

  try:
    if exc_info is None:
      exc_info = sys.exc_info()
    exc_type, exc_value, tb = exc_info
    if exc_type is None:
      return
    if _endpoint is None:
      init(SentryProject.SELFDRIVE)  # can fire in a forked daemon before init

    event = _base_event(SentryProject.SELFDRIVE)
    event["level"] = "error"
    event["exception"] = {
      "values": [{
        "type": exc_type.__name__,
        "value": str(exc_value),
        "stacktrace": {"frames": _frames_from_tb(tb)},
      }]
    }
    _post_event(event)
  except Exception:
    cloudlog.exception("sentry exception")


def init(project: SentryProject) -> bool:
  global _endpoint, _release, _environment, _user, _tags

  build_metadata = get_build_metadata()
  # forks like to mess with this, so double check
  comma_remote = build_metadata.openpilot.comma_remote and "commaai" in build_metadata.openpilot.git_origin
  if not comma_remote or not is_registered_device() or PC:
    return False

  dsn = urlparse(project.value)
  _endpoint = f"{dsn.scheme}://{dsn.hostname}/api{dsn.path}/envelope/?sentry_key={dsn.username}&sentry_version=7&sentry_client=openpilot%2f1.0"
  _release = get_version()
  _environment = "release" if build_metadata.tested_channel else "master"
  _user = {"id": Params().get("DongleId")}
  _tags = {
    "dirty": build_metadata.openpilot.is_dirty,
    "origin": build_metadata.openpilot.git_origin,
    "branch": build_metadata.channel,
    "commit": build_metadata.openpilot.git_commit,
    "device": HARDWARE.get_device_type(),
  }
  _local.tags = {}

  if project == SentryProject.SELFDRIVE:
    # report unhandled exceptions in spawned threads (replaces sentry's ThreadingIntegration)
    default_hook = threading.excepthook
    def _thread_excepthook(args) -> None:
      capture_exception((args.exc_type, args.exc_value, args.exc_traceback))
      default_hook(args)
    threading.excepthook = _thread_excepthook

  return True
