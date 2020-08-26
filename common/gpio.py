GPIO_HUB_RST_N = 30
GPIO_UBLOX_RST_N = 32
GPIO_UBLOX_SAFEBOOT_N = 33
GPIO_UBLOX_PWR_EN = 34
GPIO_STM_RST_N = 124
GPIO_STM_BOOT0 = 134


def gpio_init(pin, output):
  try:
    with open(f"/sys/class/gpio/gpio{pin}/direction", 'wb') as f:
      f.write(b"out" if output else b"in")
  except Exception as e:
    print(f"Failed to set gpio {pin} direction: {e}")


def gpio_set(pin, high):
  try:
    with open(f"/sys/class/gpio/gpio{pin}/value", 'wb') as f:
      f.write(b"1" if high else b"0")
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")
