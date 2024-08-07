from collections.abc import Callable

CanSendCallable = Callable[[list[tuple[int, bytes, int]]], None]
