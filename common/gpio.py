from functools import lru_cache
from typing import Optional, List

def gpio_init(pin: int, output: bool) -> None:
  try:
    with open(f"/sys/class/gpio/gpio{pin}/direction", 'wb') as f:
      f.write(b"out" if output else b"in")
  except Exception as e:
    print(f"Failed to set gpio {pin} direction: {e}")

def gpio_set(pin: int, high: bool) -> None:
  try:
    with open(f"/sys/class/gpio/gpio{pin}/value", 'wb') as f:
      f.write(b"1" if high else b"0")
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")

def gpio_read(pin: int) -> Optional[bool]:
  val = None
  try:
    with open(f"/sys/class/gpio/gpio{pin}/value", 'rb') as f:
      val = bool(int(f.read().strip()))
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")

  return val

@lru_cache(maxsize=None)
def get_irq_action(irq: int) -> List[str]:
  try:
    with open(f"/sys/kernel/irq/{irq}/actions") as f:
      actions = f.read().strip().split(',')
      return actions
  except FileNotFoundError:
    return []

def get_irqs_for_action(action: str) -> List[str]:
  ret = []
  with open("/proc/interrupts") as f:
    for l in f.readlines():
      irq = l.split(':')[0].strip()
      if irq.isdigit() and action in get_irq_action(irq):
        ret.append(irq)
  return ret
