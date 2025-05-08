from openpilot.common.params_pyx import Params, ParamKeyType, UnknownKeyName
assert Params
assert ParamKeyType
assert UnknownKeyName

if __name__ == "__main__":
  import sys

  params = Params()
  # Register new "LateralOnlyMode" boolean parameter (default: False)
  try:
    params.add_bool("LateralOnlyMode", False)
  except AttributeError:
    # If add_bool isn't available in this Params implementation, skip registration
    pass

  key = sys.argv[1]
  assert params.check_key(key), f"unknown param: {key}"

  if len(sys.argv) == 3:
    val = sys.argv[2]
    print(f"SET: {key} = {val}")
    params.put(key, val)
  elif len(sys.argv) == 2:
    print(f"GET: {key} = {params.get(key)}")
