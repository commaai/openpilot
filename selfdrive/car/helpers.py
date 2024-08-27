import capnp
from typing import Any

from cereal import car
from opendbc.car import structs

_FIELDS = '__dataclass_fields__'  # copy of dataclasses._FIELDS


def is_dataclass(obj):
  """Similar to dataclasses.is_dataclass without instance type check checking"""
  return hasattr(obj, _FIELDS)


def _populate_capnp_fields(dataclass: Any, capnp_builder: capnp.lib.capnp._DynamicStructBuilder) -> None:
  union_fields = capnp_builder.schema.union_fields

  for field in getattr(dataclass, _FIELDS):
    value = getattr(dataclass, field)

    if is_dataclass(value):
      if field not in union_fields or field == dataclass.which:
        nested_builder = capnp_builder.init(field)
        _populate_capnp_fields(value, nested_builder)
    elif isinstance(value, (list, tuple)):
      if value:
        list_builder = capnp_builder.init(field, len(value))
        # Check if items in the list/tuple are dataclasses
        item_is_dataclass = is_dataclass(value[0])
        for i, item in enumerate(value):
          if item_is_dataclass:
            _populate_capnp_fields(item, list_builder[i])
          else:
            list_builder[i] = item
    elif field != 'which':
      setattr(capnp_builder, field, value)


def convert_to_capnp(struct: structs.CarParams | structs.CarState | structs.CarControl.Actuators) -> capnp.lib.capnp._DynamicStructBuilder:
  if isinstance(struct, structs.CarParams):
    struct_capnp = car.CarParams.new_message()
  elif isinstance(struct, structs.CarState):
    struct_capnp = car.CarState.new_message()
  elif isinstance(struct, structs.CarControl.Actuators):
    struct_capnp = car.CarControl.Actuators.new_message()
  else:
    raise ValueError(f"Unsupported struct type: {type(struct)}")

  _populate_capnp_fields(struct, struct_capnp)
  return struct_capnp


def convert_carControl(struct: capnp.lib.capnp._DynamicStructReader) -> structs.CarControl:
  # TODO: recursively handle any car struct as needed
  def remove_deprecated(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.endswith('DEPRECATED')}

  struct_dict = struct.to_dict()
  struct_dataclass = structs.CarControl(**remove_deprecated({k: v for k, v in struct_dict.items() if not isinstance(k, dict)}))

  struct_dataclass.actuators = structs.CarControl.Actuators(**remove_deprecated(struct_dict.get('actuators', {})))
  struct_dataclass.cruiseControl = structs.CarControl.CruiseControl(**remove_deprecated(struct_dict.get('cruiseControl', {})))
  struct_dataclass.hudControl = structs.CarControl.HUDControl(**remove_deprecated(struct_dict.get('hudControl', {})))

  return struct_dataclass
