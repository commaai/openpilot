from common.params_pyx import Params, ParamKeyType, UnknownKeyName, put_nonblocking # pylint: disable=no-name-in-module, import-error
assert Params
assert ParamKeyType
assert UnknownKeyName
assert put_nonblocking

if __name__ == "__main__":
  import sys
  from common.params_pyx import keys # pylint: disable=no-name-in-module, import-error

  params = Params()
  if len(sys.argv) == 3:
    name = sys.argv[1]
    val = sys.argv[2]
    assert name.encode("utf-8") in keys.keys(), f"unknown param: {name}"
    print(f"SET: {name} = {val}")
    params.put(name, val)
  elif len(sys.argv) == 2:
    name = sys.argv[1]
    assert name.encode("utf-8") in keys.keys(), f"unknown param: {name}"
    print(f"GET: {name} = {params.get(name)}")
  else:
    for k in keys.keys():
      print(f"GET: {k} = {params.get(k)}")
