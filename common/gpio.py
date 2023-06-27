import glob
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

def get_irq_for_action(action: str) -> List[int]:
  ret = []
  for fn in glob.glob('/sys/kernel/irq/*/actions'):
    with open(fn) as f:
      actions = f.read().strip().split(',')
      if action in actions:
        irq = int(fn.split('/')[-2])
        ret.append(irq)
  return ret
