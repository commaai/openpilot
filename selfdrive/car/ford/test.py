AddrType = tuple[int, int | None]
EcuAddrSubAddr = tuple[int, int, int | None]


def test(a: bool) -> set[AddrType] | set[EcuAddrSubAddr]:
  _a1: set[AddrType] = {(1, 2), (3, 4)}
  _b1: set[EcuAddrSubAddr] = {(1, 2, 3), (3, 4, 5)}
  return _a1 if a else _b1


for i, x in test(True):
  print(i, x)
