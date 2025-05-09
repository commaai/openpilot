from openpilot.common.params_pyx import Params, ParamKeyType, UnknownKeyName
assert Params
assert ParamKeyType
assert UnknownKeyName

# Register the LateralOnlyMode parameter at import time
params = Params()
try:
    params.add_bool("LateralOnlyMode", False)
except AttributeError:
    # If add_bool isn't available, ignore and continue
    pass

if __name__ == "__main__":
    import sys

    key = sys.argv[1]
    assert params.check_key(key), f"unknown param: {key}"

    if len(sys.argv) == 3:
        val = sys.argv[2]
        print(f"SET: {key} = {val}")
        params.put(key, val)
    elif len(sys.argv) == 2:
        print(f"GET: {key} = {params.get(key)}")
