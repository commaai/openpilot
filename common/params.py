from common.params_pyx import Params, UnknownKeyName, put_nonblocking # pylint: disable=no-name-in-module, import-error
assert Params
assert UnknownKeyName
assert put_nonblocking

if __name__ == "__main__":
  import sys
  params = Params()
  if len(sys.argv) == 3:
    name = sys.argv[1]
    val = sys.argv[2]
    print(f"SET: {name} = {val}")
    params.put(name, val)
  elif len(sys.argv) == 2:
    name = sys.argv[1]
    print(f"GET: {name} = {params.get(name)}")
  else:
    from common.params_pyx import keys # pylint: disable=no-name-in-module, import-error
    for k in keys.keys():
      print(f"GET: {k} = {params.get(k)}")
