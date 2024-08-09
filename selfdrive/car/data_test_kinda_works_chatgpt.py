# import attr
from dataclasses import dataclass, field

auto_obj = object()


def auto_factory():
  return auto_obj


def apply_auto_factory(cls):
  cls_annotations = cls.__dict__.get('__annotations__', {})
  for name, typ in cls_annotations.items():
    current_value = getattr(cls, name, None)
    if current_value is auto_obj:
      setattr(cls, name, field(default_factory=typ))
  return cls


@dataclass
@apply_auto_factory
class CarControl:
  enabled: bool = auto_factory()
  pts: list[int] = auto_factory()
  logMonoTime: list[int] = field(default_factory=lambda: [1, 2, 3])


# This will now work with default values set by the decorator
car_control_instance = CarControl()
print(car_control_instance.enabled)  # Should print False
print(car_control_instance.pts)  # Should print []
