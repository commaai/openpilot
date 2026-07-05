"""Decode CAN frames into jotpluggler-style signal paths."""

from __future__ import annotations

from opendbc.can.dbc import DBC
from opendbc.can.parser import MessageState

from openpilot.tools.rerun_bridge.extract import SeriesStore


def decode_can_messages(store: SeriesStore, dbc_name: str) -> None:
  if not dbc_name:
    return

  try:
    dbc = DBC(dbc_name)
  except FileNotFoundError:
    return

  states: dict[int, MessageState] = {}
  for address, msg in dbc.msgs.items():
    states[address] = MessageState(
      address=address,
      name=msg.name,
      size=msg.size,
      signals=list(msg.sigs.values()),
      ignore_alive=True,
    )

  for frame in store.can_frames:
    state = states.get(frame.address)
    if state is None:
      continue
    if not state.parse(int(frame.mono_time * 1e9), frame.data):
      continue
    base_path = f"/{frame.service}/{frame.bus}/{state.name}"
    for sig, val in zip(state.signals, state.vals, strict=True):
      store.append(f"{base_path}/{sig.name}", frame.mono_time, float(val))