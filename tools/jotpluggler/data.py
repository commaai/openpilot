import numpy as np
import threading
import multiprocessing
import bisect
from collections import defaultdict
from tqdm import tqdm
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.logreader import _LogFileReader, LogReader


def flatten_dict(d: dict, sep: str = "/", prefix: str = None) -> dict:
  result = {}
  stack: list[tuple] = [(d, prefix)]

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
  dtype_mapping = {
    'bool': np.bool_, 'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
    'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
    'float32': np.float32, 'float64': np.float64, 'text': object, 'data': object,
    'enum': object, 'anyPointer': object,
  }

  target_dtype = dtype_mapping.get(capnp_type, object)
  return np.array(values_list, dtype=target_dtype)


def _match_field_type(field_path, field_types):
  if field_path in field_types:
    return field_types[field_path]

  path_parts = field_path.split('/')
  template_parts = [p if not p.isdigit() else '*' for p in path_parts]
  template_path = '/'.join(template_parts)
  return field_types.get(template_path)


def _get_field_times_values(segment, field_name):
  if field_name not in segment:
    return None, None

  field_data = segment[field_name]
  segment_times = segment['t']

  if field_data['sparse']:
    if len(field_data['t_index']) == 0:
      return None, None
    return segment_times[field_data['t_index']], field_data['values']
  else:
    return segment_times, field_data['values']


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
    if not hasattr(sub_msg, 'to_dict'):
      continue

    if hasattr(sub_msg, 'schema') and typ not in extracted_schemas:
      extract_field_types(sub_msg.schema, typ, field_types)
      extracted_schemas.add(typ)

    try:
      msg_dict = sub_msg.to_dict(verbose=True)
    except Exception as e:
      cloudlog.warning(f"Failed to convert sub_msg.to_dict() for message of type: {typ}: {e}")
      continue

    flat_dict = flatten_dict(msg_dict)
    flat_dict['_valid'] = msg.valid
    field_types[f"{typ}/_valid"] = 'bool'

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

      capnp_type = _match_field_type(f"{typ}/{field_name}", field_types)

      if field_name in sparse_fields:  # extract non-None values and their indices
        non_none_indices = []
        non_none_values = []
        for i, value in enumerate(values):
          if value is not None:
            non_none_indices.append(i)
            non_none_values.append(value)

        if non_none_values: # check if indices > uint16 max, currently would require a 1000+ Hz signal since indices are within segments
          assert max(non_none_indices) <= 65535, f"Sparse field {typ}/{field_name} has timestamp indices exceeding uint16 max. Max: {max(non_none_indices)}"

        typ_result[field_name] = {
          'values': _convert_to_optimal_dtype(non_none_values, capnp_type),
          'sparse': True,
          't_index': np.array(non_none_indices, dtype=np.uint16),
        }
      else:  # dense representation
        typ_result[field_name] = {'values': _convert_to_optimal_dtype(values, capnp_type), 'sparse': False}

    final_result[typ] = typ_result

  return final_result, min_time or 0.0, max_time or 0.0


def _process_segment(segment_identifier: str):
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
    self._loading = False
    self._lock = threading.RLock()

  def load_route(self, route: str) -> None:
    if self._loading:
      return
    self._reset()
    threading.Thread(target=self._load_async, args=(route,), daemon=True).start()

  def get_timeseries(self, path: str):
    with self._lock:
      msg_type, field = path.split('/', 1)
      times, values = [], []

      for segment in self._segments:
        if msg_type in segment:
          field_times, field_values = _get_field_times_values(segment[msg_type], field)
          if field_times is not None:
            times.append(field_times)
            values.append(field_values)

      if not times:
        return np.array([]), np.array([])

      combined_times = np.concatenate(times) - self._start_time

      if len(values) > 1:
        first_dtype = values[0].dtype
        if all(arr.dtype == first_dtype for arr in values):  # check if all arrays have compatible dtypes
          combined_values = np.concatenate(values)
        else:
          combined_values = np.concatenate([arr.astype(object) for arr in values])
      else:
        combined_values = values[0] if values else np.array([])

      return combined_times, combined_values

  def get_value_at(self, path: str, time: float):
    with self._lock:
      MAX_LOOKBACK = 5.0  # seconds
      absolute_time = self._start_time + time
      message_type, field = path.split('/', 1)
      current_index = bisect.bisect_right(self._segment_starts, absolute_time) - 1
      for index in (current_index, current_index - 1):
        if not 0 <= index < len(self._segments):
          continue
        segment = self._segments[index].get(message_type)
        if not segment:
          continue
        times, values = _get_field_times_values(segment, field)
        if times is None or len(times) == 0 or (index != current_index and absolute_time - times[-1] > MAX_LOOKBACK):
          continue
        position = np.searchsorted(times, absolute_time, 'right') - 1
        if position >= 0 and absolute_time - times[position] <= MAX_LOOKBACK:
          return values[position]
      return None

  def get_all_paths(self):
    with self._lock:
      return sorted(self._paths)

  def get_duration(self):
    with self._lock:
      return self._duration

  def is_plottable(self, path: str):
    _, values = self.get_timeseries(path)
    if len(values) == 0:
      return False
    return np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)

  def add_observer(self, callback):
    with self._lock:
      self._observers.append(callback)

  def remove_observer(self, callback):
    with self._lock:
      if callback in self._observers:
        self._observers.remove(callback)

  def _reset(self):
    with self._lock:
      self._loading = True
      self._segments.clear()
      self._segment_starts.clear()
      self._paths.clear()
      self._start_time = self._duration = 0.0
      observers = self._observers.copy()

    for callback in observers:
      callback({'reset': True})

  def _load_async(self, route: str):
    try:
      lr = LogReader(route, sort_by_time=True)
      if not lr.logreader_identifiers:
        cloudlog.warning(f"Warning: No log segments found for route: {route}")
        return

      num_processes = max(1, multiprocessing.cpu_count() // 2)
      with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(lr.logreader_identifiers), desc="Processing Segments") as pbar:
        for segment_result, start_time, end_time in pool.imap(_process_segment, lr.logreader_identifiers):
          pbar.update(1)
          if segment_result:
            self._add_segment(segment_result, start_time, end_time)
    except Exception:
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
        for field_name in data.keys():
          if field_name != 't':
            self._paths.add(f"{msg_type}/{field_name}")

      observers = self._observers.copy()

    for callback in observers:
      callback({'segment_added': True, 'duration': self._duration, 'segment_count': len(self._segments)})

  def _finalize_loading(self):
    with self._lock:
      self._loading = False
      observers = self._observers.copy()
      duration = self._duration

    for callback in observers:
      callback({'loading_complete': True, 'duration': duration})
