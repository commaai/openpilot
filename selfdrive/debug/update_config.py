#!/usr/bin/env python3
import ast
import importlib
import inspect
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum


class PlatformConfigFinder(ast.NodeVisitor):
  def __init__(self, platform):
    self.platform = platform

  def visit_Assign(self, node):
    for target in node.targets:
      if isinstance(target, ast.Name) and target.id == self.platform.name:
        self.target_node = node
        return


# def items_to_string(items, indent):


def stringify(value):
  if isinstance(value, str):
    return f'"{value}"'
  elif isinstance(value, list):
    # TODO: support indentation
    return f'[{", ".join(stringify(item) for item in value)}]'
  elif isinstance(value, Enum):
    return f'{value.__class__.__name__}.{value.name}'
  elif is_dataclass(value):
    return dataclass_to_string(value)
  elif isinstance(value, (int, float, bool)):
    return str(value)
  elif value is None:
    return 'None'
  else:
    raise ValueError(f'Unsupported type {type(value)}')


def dataclass_to_string(value, forcekw=False):
  def is_default_value(field):
    default_value = field.default if field.default_factory is MISSING else field.default_factory()
    return getattr(value, field.name) == default_value

  items = []
  usekw = forcekw
  for field in fields(value):
    if is_default_value(field):
      usekw = True
      continue
    items.append((f'{field.name}=' if usekw else '') + stringify(getattr(value, field.name)))

  return f'{value.__class__.__name__}({", ".join(items)})'


def update_field(updates, field, node, config, BrandFlags, debug):
  # TODO: support updating car_info
  if field.name == 'car_info':
    new_text = dataclass_to_string(config.car_info)
  elif field.name == 'specs':
    new_text = dataclass_to_string(config.specs, forcekw=True)
  elif field.name == 'flags':
    new_text = ' | '.join(f'{BrandFlags.__name__}.{flag}' for flag in BrandFlags(config.flags).name.split('|'))
  else:
    if debug: print(f'Warning: Updating {field.name} is not supported')
    return
  if ast.unparse(node) == new_text:
    if debug: print(f'{field.name} is already up to date')
    return
  updates.append(((node.lineno, (node.col_offset, node.end_col_offset)), new_text))


def update_config(platform, config, debug=True):
  car = type(platform)

  try:
    brand = platform.__module__.split('.')[-2]
    brand = brand[0].upper() + brand[1:]
    BrandFlags = importlib.import_module(car.__module__).__getattribute__(f'{brand}Flags')
  except AttributeError:
    BrandFlags = None

  values_file = inspect.getsourcefile(car)
  with open(values_file) as f:
    source = f.read()

  tree = ast.parse(source)
  finder = PlatformConfigFinder(platform)
  finder.visit(tree)

  platform_fields = fields(config)
  platform_fields_by_name = {field.name: field for field in platform_fields}

  updates = []
  target_node = finder.target_node
  for field, arg in zip(platform_fields, target_node.value.args, strict=False):
    update_field(updates, field, arg, config, BrandFlags, debug)
  for kwarg in target_node.value.keywords:
    update_field(updates, platform_fields_by_name[kwarg.arg], kwarg.value, config, BrandFlags, debug)

  source = source.split('\n')
  for (line, (start, end)), new_text in updates:
    source[line - 1] = source[line - 1][:start] + new_text + source[line - 1][end:]
  source = '\n'.join(source)

  with open(values_file, 'w') as f:
    f.write(source)

  if debug: print(f'Made {len(updates)} updates to {values_file}')
  return source


if __name__ == '__main__':
  from dataclasses import replace

  from openpilot.selfdrive.car.subaru.values import CAR as CAR_SUBARU, SubaruFlags

  original_config = CAR_SUBARU.CROSSTREK_HYBRID.config

  new_specs = replace(original_config.specs,
                      mass=0,
                      steerRatio=123,
                      centerToFrontRatio=1.0,
                      tireStiffnessFactor=0.66,
                      minEnableSpeed=99)
  new_config = replace(original_config,
                       specs=new_specs,
                       flags=SubaruFlags.HYBRID | SubaruFlags.STEER_RATE_LIMITED)

  update_config(CAR_SUBARU.CROSSTREK_HYBRID, new_config)
