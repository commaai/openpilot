import fcntl

from serial.tools import list_ports
import scservo_sdk as scs

JOINTS = ("Shoulder pan", "Shoulder lift", "Elbow", "Wrist flex", "Wrist roll", "Gripper")
IDS = tuple(range(1, 7))


def ports():
  return [{"serial": p.serial_number, "port": p.device} for p in list_ports.comports()
          if (p.vid, p.pid) == (0x1A86, 0x55D3)]


def signed_position(value):
  return -(value & 0x7FFF) if value & 0x8000 else value


class Arm:
  def __init__(self, serial):
    self.serial = serial
    self.path = next(p["port"] for p in ports() if p["serial"] == serial)
    self.port = scs.PortHandler(self.path)
    self.packet = scs.PacketHandler(0)
    if not self.port.openPort():
      raise OSError(f"Could not open {self.path}")
    try:
      fcntl.flock(self.port.ser.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
      self.models = [self.read(i, 3, 2) for i in IDS]
      self.limits = [(self.read(i, 9, 2), self.read(i, 11, 2)) for i in IDS]
      self.modes = [self.read(i, 33, 1) for i in IDS]
      self.reader = scs.GroupSyncRead(self.port, self.packet, 56, 2)
      for i in IDS:
        self.reader.addParam(i)
    except Exception:
      self.close()
      raise

  def check(self, result, error=0):
    if result != scs.COMM_SUCCESS:
      raise OSError(f"{self.serial}: {self.packet.getTxRxResult(result)}")
    if error:
      raise OSError(f"{self.serial}: {self.packet.getRxPacketError(error)}")

  def read(self, motor, address, size):
    fn = self.packet.read1ByteTxRx if size == 1 else self.packet.read2ByteTxRx
    value, result, error = fn(self.port, motor, address)
    self.check(result, error)
    return value

  def positions(self):
    self.check(self.reader.txPacket())
    positions = []
    for i in IDS:
      data, result, error = self.packet.readRx(self.port, i, 2)
      self.check(result, error)
      positions.append(signed_position(data[0] | data[1] << 8))
    return positions

  def write(self, motor, address, value, size=1):
    fn = self.packet.write1ByteTxRx if size == 1 else self.packet.write2ByteTxRx
    result, error = fn(self.port, motor, address, value)
    self.check(result, error)

  def torque(self, enabled):
    for i in IDS:
      self.write(i, 40, int(enabled))

  def targets(self, positions):
    writer = scs.GroupSyncWrite(self.port, self.packet, 42, 2)
    for i, value in zip(IDS, positions, strict=True):
      value = int(round(value))
      if not 0 <= value <= 4095:
        raise ValueError(f"Invalid target for motor {i}: {value}")
      writer.addParam(i, [value & 255, value >> 8])
    self.check(writer.txPacket())

  def close(self):
    self.port.closePort()
