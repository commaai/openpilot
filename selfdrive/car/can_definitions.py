from collections.abc import Callable
from dataclasses import dataclass

CanMsg = tuple[int, bytes, int]
CanSendCallable = Callable[[list[CanMsg]], None]


@dataclass
class CanData:
  address: int
  dat: bytes
  src: int
