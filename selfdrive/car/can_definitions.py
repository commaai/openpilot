from typing import NamedTuple


class CanData(NamedTuple):
  address: int
  dat: bytes
  src: int
