import capnp
from functools import cache
from typing import Any, cast, get_type_hints

from cereal import car
from opendbc.car import structs

_FIELDS = '__dataclass_fields__'  # copy of dataclasses._FIELDS


def is_dataclass(obj):
  """Similar to dataclasses.is_dataclass without instance type check checking"""
  return hasattr(obj, _FIELDS)


def _asdictref_inner(obj) -> dict[str, Any] | Any:
  if is_dataclass(obj):
    ret = {}
    for field in getattr(obj, _FIELDS):  # similar to dataclasses.fields()
      ret[field] = _asdictref_inner(getattr(obj, field))
    return ret
  elif isinstance(obj, (tuple, list)):
    return type(obj)(_asdictref_inner(v) for v in obj)
  else:
    return obj


def asdictref(obj) -> dict[str, Any]:
  """
  Similar to dataclasses.asdict without recursive type checking and copy.deepcopy
  Note that the resulting dict will contain references to the original struct as a result
  """
  if not is_dataclass(obj):
    raise TypeError("asdictref() should be called on dataclass instances")

  return _asdictref_inner(obj)


def convert_to_capnp(struct: structs.CarParams | structs.CarState | structs.CarControl.Actuators) -> capnp.lib.capnp._DynamicStructBuilder:
  struct_dict = asdictref(struct)

  if isinstance(struct, structs.CarParams):
    del struct_dict['lateralTuning']
    struct_capnp = car.CarParams.new_message(**struct_dict)

    # this is the only union, special handling
    which = struct.lateralTuning.which()
    struct_capnp.lateralTuning.init(which)
    lateralTuning_dict = asdictref(getattr(struct.lateralTuning, which))
    setattr(struct_capnp.lateralTuning, which, lateralTuning_dict)
  elif isinstance(struct, structs.CarState):
    struct_capnp = car.CarState.new_message(**struct_dict)
  elif isinstance(struct, structs.CarControl.Actuators):
    struct_capnp = car.CarControl.Actuators.new_message(**struct_dict)
  else:
    raise ValueError(f"Unsupported struct type: {type(struct)}")

  return struct_capnp


@cache
def _get_fieldtypes(cls: Any) -> dict[str, Any]:
  return get_type_hints(cls)


def convert_carControl(struct: capnp.lib.capnp._DynamicStructReader) -> structs.CarControl:

  def initialize_dataclass(cls: Any, data: capnp.lib.capnp._DynamicStructReader) -> Any:
    fieldtypes = _get_fieldtypes(cls)
    return cls(**{
        f: initialize_dataclass(fieldtypes[f], getattr(data, f))
        if is_dataclass(fieldtypes[f]) else getattr(data, f)
        for f in fieldtypes if hasattr(data, f)
    })

  return cast(structs.CarControl, initialize_dataclass(structs.CarControl, struct))
