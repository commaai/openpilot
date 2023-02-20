from typing import Any
import timeit

TIMER_ENABLED = True

if TIMER_ENABLED:
  def timer(name: str, f) -> Any:
    start = timeit.default_timer()
    ret = f()
    print(f"{name} took {round((timeit.default_timer() - start) * 10 ** 3, 3)}ms")
    return ret
else:
  def timer(name: str, f) -> Any:
    return f()