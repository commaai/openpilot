from collections.abc import Callable
from dataclasses import dataclass

CanSendCallable = Callable[[list[tuple[int, bytes, int]]], None]


@dataclass
class CanData:
  address: int
  dat: bytes
  src: int
