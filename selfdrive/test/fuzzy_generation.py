import hypothesis.strategies as st
import random

class FuzzyGenerator:
  def __init__(self, real_floats):
    self.real_floats=real_floats

  def generate_native_type(self, field):
    def floats(**kwargs):
      allow_nan = not self.real_floats
      allow_infinity = not self.real_floats
      return st.floats(**kwargs, allow_nan=allow_nan, allow_infinity=allow_infinity)

    if field == 'bool':
      return st.booleans()
    elif field == 'int8':
      return st.integers(min_value=-2**7, max_value=2**7-1)
    elif field == 'int16':
      return st.integers(min_value=-2**15, max_value=2**15-1)
    elif field == 'int32':
      return st.integers(min_value=-2**31, max_value=2**31-1)
    elif field == 'int64':
      return st.integers(min_value=-2**63, max_value=2**63-1)
    elif field == 'uint8':
      return st.integers(min_value=0, max_value=2**8-1)
    elif field == 'uint16':
      return st.integers(min_value=0, max_value=2**16-1)
    elif field == 'uint32':
      return st.integers(min_value=0, max_value=2**32-1)
    elif field == 'uint64':
      return st.integers(min_value=0, max_value=2**64-1)
    elif field == 'float32':
      return floats(width=32)
    elif field == 'float64':
      return floats(width=64)
    elif field == 'text':
      return st.text(max_size=1000)
    elif field == 'data':
      return st.text(max_size=1000)
    elif field == 'anyPointer':
      return st.text()
    else:
      raise NotImplementedError(f'Invalid type : {field}')

  def generate_field(self, field):
    def rec(field_type):
      if field_type.which() == 'struct':
        return self.generate_struct(field.schema.elementType if base_type == 'list' else field.schema)
      elif field_type.which() == 'list':
        return st.lists(rec(field_type.list.elementType))
      elif field_type.which() == 'enum':
        schema = field.schema.elementType if base_type == 'list' else field.schema
        return st.sampled_from(list(schema.enumerants.keys()))
      else:
        return self.generate_native_type(field_type.which())

    if 'slot' in field.proto.to_dict():
      base_type = field.proto.slot.type.which()
      return rec(field.proto.slot.type)
    else:
      return self.generate_struct(field.schema)

  def generate_struct(self, schema):
    full_fill = list(schema.non_union_fields) if schema.non_union_fields else []
    single_fill = [random.choice(schema.union_fields)] if schema.union_fields else []
    return st.fixed_dictionaries(dict((field, self.generate_field(schema.fields[field])) for field in full_fill + single_fill))

def get_random_msg(struct, real_floats=False):
  return FuzzyGenerator(real_floats=real_floats).generate_struct(struct.schema)
