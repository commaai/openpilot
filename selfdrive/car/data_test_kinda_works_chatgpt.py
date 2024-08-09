# import attr
from dataclasses import dataclass, field
from typing import Any, List
from enum import auto

auto_obj = object()


def apply_auto_defaults(cls):
  cls_annotations = cls.__annotations__
  for name, typ in cls_annotations.items():
    current_value = getattr(cls, name, None)
    if current_value is auto_obj:
      setattr(cls, name, field(default_factory=typ))
  return cls


def auto_factory():
  return auto_obj


@dataclass
@apply_auto_defaults
class CarControl:
  enabled: bool = auto_factory()
  pts: list[int] = auto_factory()
  logMonoTime: list[int] = field(default_factory=lambda: [1, 2, 3])


# This will now work with default values set by the decorator
car_control_instance = CarControl()
print(car_control_instance.enabled)  # Should print False
print(car_control_instance.pts)  # Should print []
