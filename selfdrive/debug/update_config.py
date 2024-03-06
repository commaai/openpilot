#!/usr/bin/env python3
import ast
import importlib
import inspect
from dataclasses import MISSING, fields


class PlatformConfigFinder(ast.NodeVisitor):
  def __init__(self, platform):
    super().__init__()
    self.platform = platform
    self.target_node = None

  def visit_Assign(self, node):
    if self.target_node is not None:
      return
    for target in node.targets:
      if isinstance(target, ast.Name) and target.id == self.platform.name:
        self.target_node = node
        return


def dataclass_to_string(value):
  def is_default_value(field):
    default_value = field.default if field.default_factory is MISSING else field.default_factory()
    return getattr(value, field.name) == default_value

  s = ', '.join(f'{field.name}={getattr(value, field.name)!r}'
                for field in fields(value) if not is_default_value(field))
  return f'{value.__class__.__name__}({s})'


def update_field(updates, field, node, config, BrandFlags):
  # TODO: support updating car_info
  if field.name == 'specs':
    new_text = dataclass_to_string(config.specs)
  elif field.name == 'flags':
    new_text = ' | '.join(f'{BrandFlags.__name__}.{flag}' for flag in BrandFlags(config.flags).name.split('|'))
  else:
    print(f'Warning: Updating {field.name} is not supported')
    return
  updates.append(((node.lineno, (node.col_offset, node.end_col_offset)), new_text))


def update_config(platform, config):
  car = type(platform)
  source_file = inspect.getsourcefile(car)

  brand = platform.__module__.split('.')[-2]
  brand = brand[0].upper() + brand[1:]
  BrandFlags = importlib.import_module(car.__module__).__getattribute__(f'{brand}Flags')

  with open(source_file, 'r') as f:
    source = f.read()

  tree = ast.parse(source)
  finder = PlatformConfigFinder(platform)
  finder.visit(tree)

  target_node = finder.target_node

  platform_fields = fields(config)
  platform_fields_by_name = {field.name: field for field in platform_fields}

  updates = []
  for field, arg in zip(platform_fields, target_node.value.args, strict=False):
    update_field(updates, field, arg, config, BrandFlags)
  for kwarg in target_node.value.keywords:
    update_field(updates, platform_fields_by_name[kwarg.arg], kwarg.value, config, BrandFlags)

  source = source.split('\n')
  for (line, (start, end)), new_text in updates:
    source[line - 1] = source[line - 1][:start] + new_text + source[line - 1][end:]
  source = '\n'.join(source)

  with open(source_file, 'w') as f:
    f.write(source)

  return source


if __name__ == '__main__':
  from dataclasses import replace

  from openpilot.selfdrive.car.subaru.values import CAR as CAR_SUBARU, SubaruFlags

  original_config = CAR_SUBARU.CROSSTREK_HYBRID.config

  new_specs = replace(original_config.specs, mass=0, steerRatio=123)
  new_config = replace(original_config,
                       specs=new_specs,
                       flags=SubaruFlags.HYBRID | SubaruFlags.STEER_RATE_LIMITED)

  update_config(CAR_SUBARU.CROSSTREK_HYBRID, new_config)
