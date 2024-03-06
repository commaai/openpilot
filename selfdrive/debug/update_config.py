import ast
import inspect
from dataclasses import MISSING, fields


class PlatformConfigFinder(ast.NodeVisitor):
  def __init__(self, platform):
    super().__init__()
    self.platform = platform
    self.target_node = None

  def visit_Assign(self, node):
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


def replace_colspan(source_lines, line, start, end, new_text):
  source_lines[line] = source_lines[line][:start] + new_text + source_lines[line][end:]


def update_field(source, field, node, config):
  # TODO: support updating car_info, flags
  if field.name == 'platform_str':
    new_text = config.platform_str
  elif field.name == 'specs':
    new_text = dataclass_to_string(config.specs)
  else:
    print(f"Warning: Updating {field.name} is not supported")
    return source
  return replace_colspan(source, node.lineno, node.col_offset, node.end_col_offset, new_text)


def update_config(platform, config):
  car = type(platform)
  source, starting_line = inspect.getsourcelines(car)
  source = ''.join(source)
  source_file = inspect.getsourcefile(car)

  tree = ast.parse(source)
  finder = PlatformConfigFinder(platform)
  finder.visit(tree)

  target_node = finder.target_node

  platform_fields = fields(config)
  platform_fields_by_name = {field.name: field for field in platform_fields}

  for field, arg in zip(platform_fields, target_node.value.args, strict=False):
    source = update_field(source, field, arg, config)
  for kwarg in target_node.value.keywords:
    source = update_field(source, platform_fields_by_name[kwarg.arg], kwarg.value, config)

  with open(source_file, 'w') as f:
    values = f.read()
    values = replace_colspan(
      values,
      (starting_line + target_node.lineno - 1, target_node.col_offset),
      (starting_line + target_node.end_lineno - 1, target_node.end_col_offset),
      source,
    )
    f.write(values)

  return source
