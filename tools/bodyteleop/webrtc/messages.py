import dataclasses


@dataclasses.dataclass
class PlayAlertMessage:
  alertType: str

@dataclasses.dataclass
class BatteryLevelMessage:
  batteryLevel: int

@dataclasses.dataclass
class JoystickMessage:
  x: int
  y: int

# one channel for everything
class Message:
  type: str
  message: object

  @staticmethod
  def from_dict(json_dict: dict):
    msg_type = json_dict["type"]
    msg_dict = json_dict["message"]
    if msg_type == "playAlert":
      return Message(msg_type, PlayAlertMessage(**msg_dict))
    elif msg_type == "batteryLevel":
      return Message(msg_type, BatteryLevelMessage(**msg_dict))
    elif msg_type == "joystick":
      return Message(msg_type, JoystickMessage(**msg_dict))
    else:
      raise Exception(f"Unknown message type {msg_type}")

  def to_dict(self) -> dict:
    msg_dict = dataclasses.asdict(self.message)
    return {"type": self.type, "message": msg_dict}
