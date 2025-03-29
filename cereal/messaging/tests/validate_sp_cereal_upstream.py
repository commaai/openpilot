#!/usr/bin/env python3
import argparse
import sys
from typing import Any, List, Tuple

DEBUG = False


def print_debug(string: str) -> None:
  if DEBUG:
    print(string)


def create_schema_instance(struct: Any, prop: Tuple[str, Any]) -> Any:
  """
  Create a new instance of a schema type, handling different field types.

  Args:
      struct: The Cap'n Proto schema structure
      prop: A tuple containing the field name and field metadata

  Returns:
      A new initialized schema instance
  """
  struct_instance = struct.new_message()
  field_name, field_metadata = prop

  try:
    field_type = field_metadata.proto.slot.type.which()

    # Initialize different types of fields
    if field_type in ('list', 'text', 'data'):
      struct_instance.init(field_name, 1)
      print_debug(f"Initialized list/text/data field: {field_name}")
    elif field_type in ('struct', 'object'):
      struct_instance.init(field_name)
      print_debug(f"Initialized struct/object field: {field_name}")

    return struct_instance

  except Exception as e:
    print(f"Error creating instance for {field_name}: {e}")
    return None


def get_schema_fields(schema_struct: Any) -> List[Tuple[str, Any]]:
  """
  Retrieve all fields from a given schema structure.

  Args:
      schema_struct: The Cap'n Proto schema structure

  Returns:
      A list of field names and their metadata
  """
  try:
    # Get all fields from the schema
    schema_fields = list(schema_struct.schema.fields.items())

    print_debug("Discovered schema fields:")
    for field_name, field_metadata in schema_fields:
      print_debug(f"- {field_name}")

    return schema_fields

  except Exception as e:
    print(f"Error retrieving schema fields: {e}")
    return []


def generate_schema_instances(schema_struct: Any) -> List[Any]:
  """
  Generate instances for all fields in a given schema.

  Args:
      schema_struct: The Cap'n Proto schema structure

  Returns:
      A list of schema instances
  """
  schema_fields = get_schema_fields(schema_struct)
  instances = []

  for field_prop in schema_fields:
    try:
      instance = create_schema_instance(schema_struct, field_prop)
      if instance is not None:
        instances.append(instance)
    except Exception as e:
      print(f"Skipping field due to error: {e}")

  print(f"Generated {len(instances)} schema instances")
  return instances


def persist_instances(instances: List[Any], filename: str) -> None:
  """
  Write schema instances to a binary file.

  Args:
      instances: List of schema instances
      filename: Output file path
  """
  try:
    with open(filename, 'wb') as f:
      for instance in instances:
        f.write(instance.to_bytes())

    print(f"Successfully wrote {len(instances)} instances to {filename}")

  except Exception as e:
    print(f"Error persisting instances: {e}")
    sys.exit(1)


def read_instances(filename: str, schema_type: Any) -> List[Any]:
  """
  Read schema instances from a binary file.

  Args:
      filename: Input file path
      schema_type: The schema type to use for reading

  Returns:
      A list of read schema instances
  """
  try:
    with open(filename, 'rb') as f:
      data = f.read()

    instances = list(schema_type.read_multiple_bytes(data))

    print(f"Read {len(instances)} instances from {filename}")
    return instances

  except Exception as e:
    print(f"Error reading instances: {e}")
    sys.exit(1)


def compare_schemas(original_instances: List[Any], read_instances: List[Any]) -> bool:
  """
  Compare original and read-back instances to detect potential breaking changes.

  Args:
      original_instances: List of originally generated instances
      read_instances: List of instances read back from file

  Returns:
      Boolean indicating whether schemas appear compatible
  """
  if len(original_instances) != len(read_instances):
    print("❌ Schema Compatibility Warning: Instance count mismatch")
    return False

  compatible = True
  for struct in read_instances:
    try:
      getattr(struct, struct.which())  # Attempting to access the field to validate readability
    except Exception as e:
      print(f"❌ Structural change detected: {struct.which()} is not readable.\nFull error: {e}")
      compatible = False

  return compatible


def main():
  """
  CLI entry point for schema compatibility testing.
  """
  # Setup argument parser
  parser = argparse.ArgumentParser(
    description='Cap\'n Proto Schema Compatibility Testing Tool',
    epilog='Test schema compatibility by generating and reading back instances.'
  )

  # Add mutually exclusive group for generation or reading mode
  mode_group = parser.add_mutually_exclusive_group(required=True)
  mode_group.add_argument('-g', '--generate', action='store_true',
                          help='Generate schema instances')
  mode_group.add_argument('-r', '--read', action='store_true',
                          help='Read and validate schema instances')

  # Common arguments
  parser.add_argument('-f', '--file',
                      default='schema_instances.bin',
                      help='Output/input binary file (default: schema_instances.bin)')

  # Parse arguments
  args = parser.parse_args()

  # Import the schema dynamically 
  try:
    from cereal import log
    schema_type = log.Event
  except ImportError:
    print("Error: Unable to import schema. Ensure 'cereal' is installed.")
    sys.exit(1)

  # Execute based on mode
  if args.generate:
    print("🔧 Generating Schema Instances")
    instances = generate_schema_instances(schema_type)
    persist_instances(instances, args.file)
    print("✅ Instance generation complete")

  elif args.read:
    print("🔍 Reading and Validating Schema Instances")
    generated_instances = generate_schema_instances(schema_type)
    read_back_instances = read_instances(args.file, schema_type)

    # Compare schemas
    if compare_schemas(generated_instances, read_back_instances):
      print("✅ Schema Compatibility: No breaking changes detected")
      sys.exit(0)
    else:
      print("❌ Potential Schema Breaking Changes Detected")
      sys.exit(1)


if __name__ == "__main__":
  main()
