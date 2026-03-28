from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BehaviorState(StrEnum):
  IDLE = "idle"
  ACQUIRE = "acquire"
  FROZEN = "frozen"
  ADVANCE = "advance"
  LOST = "lost"


@dataclass
class BehaviorInputs:
  enabled: bool
  target_visible: bool
  attending: bool
  obstacle_too_close: bool


def next_state(state: BehaviorState, inp: BehaviorInputs) -> BehaviorState:
  if not inp.enabled:
    return BehaviorState.IDLE

  if not inp.target_visible:
    return BehaviorState.LOST if state != BehaviorState.IDLE else BehaviorState.IDLE

  if state in (BehaviorState.IDLE, BehaviorState.LOST):
    return BehaviorState.ACQUIRE

  if inp.obstacle_too_close:
    return BehaviorState.FROZEN

  if inp.attending:
    return BehaviorState.FROZEN

  return BehaviorState.ADVANCE
