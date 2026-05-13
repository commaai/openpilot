import sys
import capnp
from pathlib import Path

NO_DISCRIMINANT = 65535
SCALAR_KINDS = {
  "bool": "Bool",
  "int8": "Int",
  "int16": "Int",
  "int32": "Int",
  "int64": "Int",
  "uint8": "UInt",
  "uint16": "UInt",
  "uint32": "UInt",
  "uint64": "UInt",
  "float32": "Float",
  "float64": "Float",
  "enum": "Enum",
}
NESTED_TYPE_KINDS = {"struct", "list"}
IGNORED_TYPE_KINDS = {"void", "text", "data", "interface", "anyPointer"}


def cxx_string(value):
  return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def accessor(prefix, name):
  return prefix + name[:1].upper() + name[1:]


def field_type(field):
  if field.proto.which() == "group":
    return "struct"
  return field.proto.slot.type.which()


def field_type_proto(field):
  return field.proto.slot.type if field.proto.which() == "slot" else None


def scalar_kind(type_proto):
  if type_proto is None:
    return None
  return SCALAR_KINDS.get(type_proto.which())


def enum_names(schema):
  if schema is None:
    return []
  names_by_ordinal = schema.enumerants
  if not names_by_ordinal:
    return []
  max_ordinal = max(names_by_ordinal.values())
  out = [""] * (max_ordinal + 1)
  for name, ordinal in names_by_ordinal.items():
    out[ordinal] = name
  return out


class Generator:
  def __init__(self, event_schema):
    self.event_schema = event_schema
    self.fixed_paths = []
    self.tmp_index = 0
    self.lines = []
    self.emits_memo = {}

  def tmp(self, prefix):
    self.tmp_index += 1
    return f"{prefix}_{self.tmp_index}"

  def add_fixed_path(self, path):
    slot = len(self.fixed_paths)
    self.fixed_paths.append(path)
    return slot

  def emit(self, indent, text=""):
    self.lines.append(" " * indent + text)

  def scalar_double_expr(self, value_expr, kind):
    if kind == "Bool":
      return f"({value_expr} ? 1.0 : 0.0)"
    if kind == "Enum":
      return f"static_cast<double>(static_cast<uint16_t>({value_expr}))"
    return f"static_cast<double>({value_expr})"

  def emit_enum_capture(self, indent, path_expr, names):
    if not names:
      return
    names_expr = "{" + ", ".join(cxx_string(name) for name in names) + "}"
    self.emit(indent, f"capture_static_enum_info({path_expr}, {names_expr}, series);")

  def emit_node(self, indent, type_kind, type_proto, schema, expr, path, path_expr, dynamic_path):
    if not self.node_emits(type_kind, type_proto, schema):
      return
    kind = scalar_kind(type_proto)
    if kind is not None:
      double_expr = self.scalar_double_expr(expr, kind)
      if dynamic_path:
        if kind == "Enum":
          self.emit_enum_capture(indent, path_expr, enum_names(schema))
        self.emit(indent, f"append_dynamic_scalar_point({path_expr}, tm, {double_expr}, series);")
      else:
        slot = self.add_fixed_path(path)
        if kind == "Enum":
          self.emit_enum_capture(indent, cxx_string(path), enum_names(schema))
        self.emit(indent, f"append_fixed_scalar_point(&series->fixed_series[{slot}], tm, {double_expr});")
      return

    if type_kind == "struct":
      self.emit_struct(indent, schema, expr, path, path_expr, dynamic_path)
      return

    if type_kind == "list":
      self.emit_list(indent, type_proto, schema, expr, path, path_expr, dynamic_path)

  def emit_field(self, indent, struct_schema, reader_expr, field_name, base_path, base_path_expr, dynamic_path):
    field = struct_schema.fields[field_name]
    proto = field.proto
    type_kind = field_type(field)
    type_proto = field_type_proto(field)
    kind = scalar_kind(type_proto)
    value_schema = field.schema if kind == "Enum" or type_kind in NESTED_TYPE_KINDS else None
    if not self.node_emits(type_kind, type_proto, value_schema):
      return

    field_path = f"{base_path}/{field_name}"
    field_path_expr = None
    if dynamic_path:
      field_path_var = self.tmp("path")
      self.emit(indent, f"const std::string {field_path_var} = {base_path_expr} + {cxx_string('/' + field_name)};")
      field_path_expr = field_path_var

    get_call = f"{reader_expr}.{accessor('get', field_name)}()"
    has_call = f"{reader_expr}.{accessor('has', field_name)}()"
    conditions = []
    if proto.discriminantValue != NO_DISCRIMINANT:
      conditions.append(f"{reader_expr}.which() == static_cast<decltype({reader_expr}.which())>({proto.discriminantValue})")
    if proto.which() == "slot" and type_kind in NESTED_TYPE_KINDS:
      conditions.append(has_call)

    if conditions:
      self.emit(indent, f"if ({' && '.join(conditions)}) {{")
      indent += 2

    value_var = self.tmp("value")
    self.emit(indent, f"const auto {value_var} = {get_call};")
    self.emit_node(indent, type_kind, type_proto, value_schema, value_var, field_path, field_path_expr, dynamic_path)

    if conditions:
      indent -= 2
      self.emit(indent, "}")

  def emit_struct(self, indent, schema, reader_expr, path, path_expr, dynamic_path):
    if schema is None:
      return
    for field_name in schema.fieldnames:
      self.emit_field(indent, schema, reader_expr, field_name, path, path_expr, dynamic_path)

  def emit_list(self, indent, type_proto, schema, list_expr, path, path_expr, dynamic_path):
    elem_type = type_proto.list.elementType
    elem_kind = elem_type.which()
    if elem_kind in IGNORED_TYPE_KINDS:
      return

    base_path_var = path_expr
    if base_path_var is None:
      base_path_var = self.tmp("base_path")
      self.emit(indent, f"const std::string {base_path_var} = {cxx_string(path)};")

    elem_scalar = scalar_kind(elem_type)
    if elem_scalar is not None:
      self.emit(indent, f"if ({list_expr}.size() <= 16) {{")
      index_var = self.tmp("i")
      self.emit(indent + 2, f"for (uint {index_var} = 0; {index_var} < {list_expr}.size(); ++{index_var}) {{")
      item_series = self.tmp("item_series")
      self.emit(indent + 4, f"RouteSeries *{item_series} = ensure_list_scalar_series({base_path_var}, {index_var}, series);")
      if elem_scalar == "Enum":
        self.emit_enum_capture(indent + 4, f"{item_series}->path", enum_names(schema.elementType))
      self.emit(indent + 4, f"append_fixed_scalar_point({item_series}, tm, {self.scalar_double_expr(f'{list_expr}[{index_var}]', elem_scalar)});")
      self.emit(indent + 2, "}")
      self.emit(indent, "}")
      return

    if elem_kind in {"struct", "list"}:
      index_var = self.tmp("i")
      self.emit(indent, f"for (uint {index_var} = 0; {index_var} < {list_expr}.size(); ++{index_var}) {{")
      item_path = self.tmp("item_path")
      self.emit(indent + 2, f"const std::string {item_path} = {base_path_var} + \"/\" + std::to_string({index_var});")
      item = self.tmp("item")
      self.emit(indent + 2, f"const auto {item} = {list_expr}[{index_var}];")
      if elem_kind == "struct":
        self.emit_struct(indent + 2, schema.elementType, item, path, item_path, True)
      else:
        self.emit_list(indent + 2, elem_type, schema.elementType, item, path, item_path, True)
      self.emit(indent, "}")

  def node_emits(self, type_kind, type_proto, schema, seen=frozenset()):
    if scalar_kind(type_proto) is not None:
      return True
    if type_kind == "struct":
      if schema is None:
        return False
      schema_id = int(schema.node.id)
      if schema_id in seen:
        return False
      if schema_id in self.emits_memo:
        return self.emits_memo[schema_id]
      next_seen = seen | {schema_id}
      for field_name in schema.fieldnames:
        field = schema.fields[field_name]
        ft = field_type(field)
        ftp = field_type_proto(field)
        fkind = scalar_kind(ftp)
        if ft in IGNORED_TYPE_KINDS:
          continue
        fschema = field.schema if fkind == "Enum" or ft in NESTED_TYPE_KINDS else None
        if self.node_emits(ft, ftp, fschema, next_seen):
          self.emits_memo[schema_id] = True
          return True
      self.emits_memo[schema_id] = False
      return False
    if type_kind == "list":
      if type_proto is None or schema is None:
        return False
      elem_type = type_proto.list.elementType
      elem_kind = elem_type.which()
      if elem_kind in IGNORED_TYPE_KINDS:
        return False
      if scalar_kind(elem_type) is not None:
        return True
      if elem_kind == "struct":
        return self.node_emits("struct", None, schema.elementType, seen)
      if elem_kind == "list":
        return self.node_emits("list", elem_type, schema.elementType, seen)
    return False

  def emit_can_special(self, indent, service_name):
    service_kind = "CanServiceKind::Can" if service_name == "can" else "CanServiceKind::Sendcan"
    self.emit(indent, f"const CanServiceKind can_service = {service_kind};")
    self.emit(indent, f"for (const auto &msg : event.{accessor('get', service_name)}()) {{")
    self.emit(indent + 2, "append_can_frame(can_service, static_cast<uint8_t>(msg.getSrc()), msg.getAddress(), msg.getDeprecated().getBusTime(), msg.getDat(), tm, series);")  # noqa: E501
    self.emit(indent + 2, "if (skip_raw_can) {")
    self.emit(indent + 4, "const auto dat = msg.getDat();")
    self.emit(indent + 4, f"decode_can_frame(can_dbc, {cxx_string(service_name)}, static_cast<uint8_t>(msg.getSrc()), msg.getAddress(), dat.begin(), dat.size(), tm, series);")  # noqa: E501
    self.emit(indent + 2, "}")
    self.emit(indent, "}")
    self.emit(indent, "if (skip_raw_can) {")
    self.emit(indent + 2, "return true;")
    self.emit(indent, "}")

  def emit_event_case(self, field_name):
    field = self.event_schema.fields[field_name]
    proto = field.proto
    type_kind = field_type(field)
    type_proto = field_type_proto(field)
    kind = scalar_kind(type_proto)
    schema = field.schema if kind == "Enum" or type_kind in NESTED_TYPE_KINDS else None
    self.emit(4, f"case static_cast<cereal::Event::Which>({proto.discriminantValue}): {{")
    valid_slot = self.add_fixed_path(f"/{field_name}/valid")
    mono_slot = self.add_fixed_path(f"/{field_name}/logMonoTime")
    seconds_slot = self.add_fixed_path(f"/{field_name}/t")
    self.emit(6, f"append_fixed_scalar_point(&series->fixed_series[{valid_slot}], tm, event.getValid() ? 1.0 : 0.0);")
    self.emit(6, f"append_fixed_scalar_point(&series->fixed_series[{mono_slot}], tm, static_cast<double>(event.getLogMonoTime()));")
    self.emit(6, f"append_fixed_scalar_point(&series->fixed_series[{seconds_slot}], tm, tm);")
    if field_name in {"can", "sendcan"}:
      self.emit_can_special(6, field_name)
    if self.node_emits(type_kind, type_proto, schema):
      payload = self.tmp("payload")
      self.emit(6, f"const auto {payload} = event.{accessor('get', field_name)}();")
      self.emit_node(6, type_kind, type_proto, schema, payload, f"/{field_name}", None, False)
    self.emit(6, "return true;")
    self.emit(4, "}")

  def generate(self):
    self.lines = []
    self.emit(0, "// Generated by tools/jotpluggler/generate_event_extractors.py; do not edit.")
    self.emit(0, "")
    self.emit(0, "const std::vector<std::string> &static_event_fixed_paths() {")
    self.emit(2, "static const std::vector<std::string> paths = {")
    path_insert_at = len(self.lines)
    self.emit(2, "};")
    self.emit(2, "return paths;")
    self.emit(0, "}")
    self.emit(0, "")
    self.emit(0, "void capture_static_enum_info(const std::string &path, std::initializer_list<std::string_view> names, SeriesAccumulator *series) {")
    self.emit(2, "if (series->enum_info.find(path) != series->enum_info.end()) {")
    self.emit(4, "return;")
    self.emit(2, "}")
    self.emit(2, "EnumInfo info;")
    self.emit(2, "info.names.reserve(names.size());")
    self.emit(2, "for (std::string_view name : names) {")
    self.emit(4, "info.names.emplace_back(name);")
    self.emit(2, "}")
    self.emit(2, "if (!info.names.empty()) {")
    self.emit(4, "series->enum_info.emplace(path, std::move(info));")
    self.emit(2, "}")
    self.emit(0, "}")
    self.emit(0, "")
    self.emit(0, "bool append_event_static_reader(cereal::Event::Which which, const cereal::Event::Reader &event, const dbc::Database *can_dbc, bool skip_raw_can, double time_offset, SeriesAccumulator *series) {")  # noqa: E501
    self.emit(2, "const double tm = static_cast<double>(event.getLogMonoTime()) / 1.0e9 - time_offset;")
    self.emit(2, "switch (which) {")
    for field_name in self.event_schema.union_fields:
      self.emit_event_case(field_name)
    self.emit(4, "default:")
    self.emit(6, "return false;")
    self.emit(2, "}")
    self.emit(0, "}")

    path_lines = ["    " + cxx_string(path) + "," for path in self.fixed_paths]
    self.lines[path_insert_at:path_insert_at] = path_lines
    return "\n".join(self.lines) + "\n"


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <repo-root> <output>", file=sys.stderr)
    sys.exit(2)

  repo_root = Path(sys.argv[1]).resolve()
  output = Path(sys.argv[2])
  capnp.remove_import_hook()
  log = capnp.load(str(repo_root / "cereal" / "log.capnp"))
  generated = Generator(log.Event.schema).generate()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(generated)
