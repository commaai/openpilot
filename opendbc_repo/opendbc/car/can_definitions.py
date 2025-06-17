from collections.abc import Callable
from typing import NamedTuple, Protocol


class CanData(NamedTuple):
  address: int
  dat: bytes
  src: int


CanSendCallable = Callable[[list[CanData]], None]


class CanRecvCallable(Protocol):
  def __call__(self, wait_for_one: bool = False) -> list[list[CanData]]: ...
