# import attr
from enum import Enum
from typing import get_origin, get_args, get_type_hints
from dataclasses import dataclass, field, is_dataclass

auto_obj = object()


def auto_field():
  return auto_obj


def apply_auto_fields(cls):
  cls_annotations = cls.__dict__.get('__annotations__', {})
  for name, typ in cls_annotations.items():
    current_value = getattr(cls, name, None)
    if current_value is auto_obj:
      origin_typ = get_origin(typ) or typ
      if isinstance(origin_typ, str):
        raise TypeError(f"Forward references are not supported for auto_field: '{origin_typ}'. Use a default_factory with lambda instead.")
      elif origin_typ in (int, float, str, bytes, list, tuple, set, dict, bool) or is_dataclass(origin_typ):
        setattr(cls, name, field(default_factory=origin_typ))
      elif origin_typ is None:
        setattr(cls, name, field(default=origin_typ))
      elif issubclass(origin_typ, Enum):  # first enum is the default
        setattr(cls, name, field(default=next(iter(origin_typ))))
      else:
        raise TypeError(f"Unsupported type for auto_field: {origin_typ}")
  return cls


@dataclass
@apply_auto_fields
class CarControl:
  enabled: bool = auto_field()
  pts: list[int] = auto_field()
  logMonoTime: list[int] = field(default_factory=lambda: [1, 2, 3])


# This will now work with default values set by the decorator
car_control_instance = CarControl()
print(car_control_instance.enabled)  # Should print False
print(car_control_instance.pts)  # Should print []
