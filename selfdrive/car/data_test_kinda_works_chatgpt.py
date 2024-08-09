# import attr
from dataclasses import dataclass, field
from typing import Any, List

class AutoDefault:
    """Marker class for fields that should automatically get a default."""
    def __init__(self, factory: Any = None):
        self.factory = factory

def auto_factory(factory=None):
    """Function to return AutoDefault instance."""
    return AutoDefault(factory=factory)

def apply_auto_defaults(cls):
    cls_annotations = cls.__annotations__
    for name, typ in cls_annotations.items():
        current_value = getattr(cls, name, None)
        if isinstance(current_value, AutoDefault):
            if current_value.factory is not None:
                setattr(cls, name, field(default_factory=current_value.factory))
            else:
                # Handle specific default types here or raise an error
                setattr(cls, name, field(default_factory=typ))
                # if typ == bool:
                #     setattr(cls, name, attr.field(default=False))
                # elif typ == list:
                #     setattr(cls, name, attr.field(factory=list))
                # elif typ == dict:
                #     setattr(cls, name, attr.field(factory=dict))
                # else:
                #     raise ValueError(f"No default or factory defined for type {typ}")
    return cls

@dataclass
@apply_auto_defaults
class CarControl:
    enabled: bool = auto_factory()  # Automatically get a default of False
    pts: list[int] = auto_factory()  # Automatically get an empty list as the default


# This will now work with default values set by the decorator
car_control_instance = CarControl(enabled=True,)
print(car_control_instance.enabled)  # Should print False
print(car_control_instance.pts)      # Should print []
