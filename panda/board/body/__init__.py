# python helpers for the body panda
import struct

from panda import Panda

class PandaBody(Panda):

  MOTOR_LEFT: int = 1
  MOTOR_RIGHT: int = 2

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._rpm_left: int = 0
    self._rpm_right: int = 0

  @property
  def rpm_left(self) -> int:
    return self._rpm_left

  @rpm_left.setter
  def rpm_left(self, value: int) -> None:
    self._rpm_left = int(value)
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xb3, self._rpm_left, self._rpm_right, b'')

  @property
  def rpm_right(self) -> int:
    return self._rpm_right

  @rpm_right.setter
  def rpm_right(self, value: int) -> None:
    self._rpm_right = int(value)
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xb3, self._rpm_left, self._rpm_right, b'')

  # ****************** Motor Control *****************
  @staticmethod
  def _ensure_valid_motor(motor: int) -> None:
    if motor not in (PandaBody.MOTOR_LEFT, PandaBody.MOTOR_RIGHT):
      raise ValueError("motor must be MOTOR_LEFT or MOTOR_RIGHT")

  def motor_get_encoder_state(self, motor: int) -> tuple[int, float]:
    self._ensure_valid_motor(motor)
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xe2, motor, 0, 8)
    position, rpm_milli = struct.unpack("<ii", dat)
    return position, rpm_milli / 1000.0
