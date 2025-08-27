import numpy as np
import threading
import multiprocessing
import bisect
from collections import defaultdict
from typing import Any
import tqdm
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.logreader import _LogFileReader, LogReader


def flatten_dict(d: dict, sep: str = "/", prefix: str = None) -> dict:
  result = {}
  stack = [(d, prefix)]

  while stack:
    obj, current_prefix = stack.pop()

    if isinstance(obj, dict):
      for key, val in obj.items():
        new_prefix = key if current_prefix is None else f"{current_prefix}{sep}{key}"
        if isinstance(val, (dict, list)):
          stack.append((val, new_prefix))
        else:
          result[new_prefix] = val
    elif isinstance(obj, list):
      for i, item in enumerate(obj):
        new_prefix = f"{current_prefix}{sep}{i}"
        if isinstance(item, (dict, list)):
          stack.append((item, new_prefix))
        else:
          result[new_prefix] = item
    else:
      if current_prefix is not None:
        result[current_prefix] = obj
  return result


def extract_field_types(schema, prefix, field_types_dict):
  stack = [(schema, prefix)]

  while stack:
    current_schema, current_prefix = stack.pop()

    for field in current_schema.fields_list:
      field_name = field.proto.name
      field_path = f"{current_prefix}/{field_name}"
      field_proto = field.proto
      field_which = field_proto.which()

      field_type = field_proto.slot.type.which() if field_which == 'slot' else field_which
      field_types_dict[field_path] = field_type

      if field_which == 'slot':
        slot_type = field_proto.slot.type
        type_which = slot_type.which()

        if type_which == 'list':
          element_type = slot_type.list.elementType.which()
          list_path = f"{field_path}/*"
          field_types_dict[list_path] = element_type

          if element_type == 'struct':
            stack.append((field.schema.elementType, list_path))

        elif type_which == 'struct':
          stack.append((field.schema, field_path))

      elif field_which == 'group':
        stack.append((field.schema, field_path))


def _convert_to_optimal_dtype(values_list, capnp_type):
  if not values_list:
    return np.array([])

  dtype_mapping = {
    'bool': np.bool_, 'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
    'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
    'float32': np.float32, 'float64': np.float64, 'text': object, 'data': object,
    'enum': object, 'anyPointer': object,
  }

  target_dtype = dtype_mapping.get(capnp_type)
  return np.array(values_list, dtype=target_dtype) if target_dtype else np.array(values_list)


def _match_field_type(field_path, field_types):
  if field_path in field_types:
    return field_types[field_path]

  path_parts = field_path.split('/')
  template_parts = [p if not p.isdigit() else '*' for p in path_parts]
  template_path = '/'.join(template_parts)
  return field_types.get(template_path)


def msgs_to_time_series(msgs):
  """Extract scalar fields and return (time_series_data, start_time, end_time)."""
  collected_data = defaultdict(lambda: {'timestamps': [], 'columns': defaultdict(list), 'sparse_fields': set()})
  field_types = {}
  extracted_schemas = set()
  min_time = max_time = None

  for msg in msgs:
    typ = msg.which()
    timestamp = msg.logMonoTime * 1e-9
    if typ != 'initData':
      if min_time is None:
        min_time = timestamp
      max_time = timestamp

    sub_msg = getattr(msg, typ)
    if not hasattr(sub_msg, 'to_dict') or typ in ('qcomGnss', 'ubloxGnss'):
      continue

    if hasattr(sub_msg, 'schema') and typ not in extracted_schemas:
      extract_field_types(sub_msg.schema, typ, field_types)
      extracted_schemas.add(typ)

    msg_dict = sub_msg.to_dict(verbose=True)
    flat_dict = flatten_dict(msg_dict)
    flat_dict['_valid'] = msg.valid

    type_data = collected_data[typ]
    columns, sparse_fields = type_data['columns'], type_data['sparse_fields']
    known_fields = set(columns.keys())
    missing_fields = known_fields - flat_dict.keys()

    for field, value in flat_dict.items():
      if field not in known_fields and type_data['timestamps']:
        sparse_fields.add(field)
      columns[field].append(value)
      if value is None:
        sparse_fields.add(field)

    for field in missing_fields:
      columns[field].append(None)
      sparse_fields.add(field)

    type_data['timestamps'].append(timestamp)

  final_result = {}
  for typ, data in collected_data.items():
    if not data['timestamps']:
      continue

    typ_result = {'t': np.array(data['timestamps'], dtype=np.float64)}
    sparse_fields = data['sparse_fields']

    for field_name, values in data['columns'].items():
      if len(values) < len(data['timestamps']):
        values = [None] * (len(data['timestamps']) - len(values)) + values
        sparse_fields.add(field_name)

      if field_name in sparse_fields:
        typ_result[field_name] = np.array(values, dtype=object)
      else:
        capnp_type = _match_field_type(f"{typ}/{field_name}", field_types)
        typ_result[field_name] = _convert_to_optimal_dtype(values, capnp_type)

    final_result[typ] = typ_result

  return final_result, min_time or 0.0, max_time or 0.0


def _process_segment(segment_identifier: str) -> tuple[dict[str, Any], float, float]:
  try:
    lr = _LogFileReader(segment_identifier, sort_by_time=True)
    return msgs_to_time_series(lr)
  except Exception as e:
    cloudlog.warning(f"Warning: Failed to process segment {segment_identifier}: {e}")
    return {}, 0.0, 0.0


class DataManager:
  def __init__(self):
    self._segments = []
    self._segment_starts = []
    self._start_time = 0.0
    self._duration = 0.0
    self._paths = set()
    self._observers = []
    self.loading = False
    self._lock = threading.RLock()

  def load_route(self, route: str) -> None:
    if self.loading:
      return
    self._reset()
    threading.Thread(target=self._load_async, args=(route,), daemon=True).start()

  def get_timeseries(self, path: str):
    with self._lock:
      msg_type, field = path.split('/', 1)
      times, values = [], []

      for segment in self._segments:
        if msg_type in segment and field in segment[msg_type]:
          times.append(segment[msg_type]['t'])
          values.append(segment[msg_type][field])

      if not times:
        return None

      combined_times = np.concatenate(times) - self._start_time
      if len(values) > 1 and any(arr.dtype != values[0].dtype for arr in values):
        values = [arr.astype(object) for arr in values]

      return combined_times, np.concatenate(values)

  def get_value_at(self, path: str, time: float):
    with self._lock:
      absolute_time = self._start_time + time
      message_type, field = path.split('/', 1)
      current_index = bisect.bisect_right(self._segment_starts, absolute_time) - 1
      for index in (current_index, current_index - 1):
        if not 0 <= index < len(self._segments):
          continue
        segment = self._segments[index].get(message_type)
        if not segment or field not in segment:
          continue
        times = segment['t']
        if len(times) == 0 or (index != current_index and absolute_time - times[-1] > 1):
          continue
        position = np.searchsorted(times, absolute_time, 'right') - 1
        if position >= 0 and absolute_time - times[position] <= 1:
          return segment[field][position]
      return None

  def get_all_paths(self):
    with self._lock:
      return sorted(self._paths)

  def get_duration(self):
    with self._lock:
      return self._duration

  def is_plottable(self, path: str):
    data = self.get_timeseries(path)
    if data is None:
      return False
    _, values = data
    return np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)

  def add_observer(self, callback):
    with self._lock:
      self._observers.append(callback)

  def _reset(self):
    with self._lock:
      self.loading = True
      self._segments.clear()
      self._segment_starts.clear()
      self._paths.clear()
      self._start_time = self._duration = 0.0

  def _load_async(self, route: str):
    try:
      lr = LogReader(route, sort_by_time=True)
      if not lr.logreader_identifiers:
        cloudlog.warning(f"Warning: No log segments found for route: {route}")
        return

      with multiprocessing.Pool() as pool, tqdm.tqdm(total=len(lr.logreader_identifiers), desc="Processing Segments") as pbar:
        for segment_result, start_time, end_time in pool.imap(_process_segment, lr.logreader_identifiers):
          pbar.update(1)
          if segment_result:
            self._add_segment(segment_result, start_time, end_time)
    except Exception as e:
      cloudlog.exception(f"Error loading route {route}:")
    finally:
      self._finalize_loading()

  def _add_segment(self, segment_data: dict, start_time: float, end_time: float):
    with self._lock:
      self._segments.append(segment_data)
      self._segment_starts.append(start_time)

      if len(self._segments) == 1:
        self._start_time = start_time
      self._duration = end_time - self._start_time

      for msg_type, data in segment_data.items():
        for field in data.keys():
          if field != 't':
            self._paths.add(f"{msg_type}/{field}")

      observers = self._observers.copy()

    for callback in observers:
      callback({'segment_added': True, 'duration': self._duration})

  def _finalize_loading(self):
    with self._lock:
      self.loading = False
      observers = self._observers.copy()
      duration = self._duration

    for callback in observers:
      callback({'loading_complete': True, 'duration': duration})
