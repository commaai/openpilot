from openpilot.common.conversions import Conversions as CV
from openpilot.common.enums import Star

def get_detail_sentence(
    make: str,
    model: str,
    longitudinal: str,
    min_enable_speed: float,
    min_steer_speed: float,
    steering_torque: Star,
    auto_resume: bool
  ) -> str:
  """
  Returns the detail sentence of a car.
  """
  sentence_builder = "openpilot upgrades your <strong>{car_model}</strong> with automated lane centering{alc} and adaptive cruise control{acc}."
  if min_steer_speed > min_enable_speed:
    alc = f" <strong>above {min_steer_speed * CV.MS_TO_MPH:.0f} mph</strong>," if min_steer_speed > 0 else " <strong>at all speeds</strong>,"
  else:
    alc = ""
  acc = ""
  if min_enable_speed > 0:
    acc = f" <strong>while driving above {min_enable_speed * CV.MS_TO_MPH:.0f} mph</strong>"
  elif auto_resume:
    acc = " <strong>that automatically resumes from a stop</strong>"
  if steering_torque != Star.FULL:
    sentence_builder += " This car may not be able to take tight turns on its own."
  exp_link = "<a href='https://blog.comma.ai/090release/#experimental-mode' target='_blank' class='link-light-new-regular-text'>Experimental mode</a>"
  if longitudinal == "openpilot":
    sentence_builder += f" Traffic light and stop sign handling is also available in {exp_link}."
  return sentence_builder.format(car_model=f"{make} {model}", alc=alc, acc=acc)
