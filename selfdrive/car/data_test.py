# sentinel value to detect when to replace a field's default
from dataclasses import dataclass, field

auto = object()


def check_auto(name, bases, cls_dict):
  default = 1

  cls_annotations = cls_dict.get('__annotations__', {})
  print('cls_annotations2', cls_annotations)

  for name, val in cls_dict.items():
    print('test', name, val)
    if val == auto:
      # cls_dict[name] = default
      # cls_dict[name] = cls_annotations[name]
      cls_dict[name] = field(default_factory=cls_annotations[name])
      default += 1

  cls = type(name, bases, cls_dict)
  return cls


@dataclass(frozen=True)
class State(metaclass=check_auto):
  test: int = auto
  test2: str = auto
  test3: bool = auto


s = State()
# print(s)  # State(val_A=1, val_B=2, val_C=3)
# assert s.val_B == 2
#
# s = State(val_A=5)
# assert s.val_A == 5
# assert s.val_C == 3
