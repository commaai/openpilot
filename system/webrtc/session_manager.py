"""
In-process WebRTC session manager.

Runs a background asyncio event loop thread to host StreamSession lifecycles,
replacing the old webrtcd HTTP server on port 5001.
Sessions are only reachable through authenticated RPC (Athena/BLE).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from openpilot.system.webrtc.webrtcd import StreamSession

logger = logging.getLogger("webrtc.session_manager")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()
_sessions: dict[str, StreamSession] = {}


def _ensure_loop() -> asyncio.AbstractEventLoop:
  global _loop, _thread
  if _loop is not None and _loop.is_running():
    return _loop

  with _lock:
    if _loop is not None and _loop.is_running():
      return _loop

    _loop = asyncio.new_event_loop()

    def _run():
      asyncio.set_event_loop(_loop)
      _loop.run_forever()

    _thread = threading.Thread(target=_run, name="webrtc_loop", daemon=True)
    _thread.start()

  return _loop


async def _create_session_async(sdp: str, cameras: list[str], bridge_services_in: list[str], bridge_services_out: list[str]) -> dict[str, str]:
  session = StreamSession(sdp, cameras, bridge_services_in, bridge_services_out)
  answer = await session.get_answer()
  session.start()
  _sessions[session.identifier] = session
  return {"sdp": answer.sdp, "type": answer.type}


def create_session(sdp: str, cameras: list[str], bridge_services_in: list[str], bridge_services_out: list[str], timeout: float = 60) -> dict[str, str]:
  """Synchronous entry point — safe to call from any thread (athenad workers, BLE GLib loop, etc.)."""
  loop = _ensure_loop()
  future = asyncio.run_coroutine_threadsafe(
    _create_session_async(sdp, cameras, bridge_services_in, bridge_services_out),
    loop,
  )
  return future.result(timeout=timeout)


def _notify_all(payload: Any) -> None:
  msg = json.dumps(payload)
  for session in list(_sessions.values()):
    try:
      session.stream.get_messaging_channel().send(msg)
    except Exception:
      continue


def notify_all(payload: Any) -> None:
  """Push a JSON payload to all active WebRTC sessions' data channels."""
  loop = _ensure_loop()
  loop.call_soon_threadsafe(_notify_all, payload)
