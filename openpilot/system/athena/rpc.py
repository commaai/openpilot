import inspect
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

# a minimal implementation of json-rpc 2.0 https://www.jsonrpc.org/specification

JSONRPC_VERSION = "2.0"

# JSON-RPC 2.0 reserved / application error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SERVER_ERROR = -32000

JsonDict = dict[str, Any]
MethodMap = Mapping[str, Callable[..., Any]]
Validator = Callable[[Any], bool]


# *** param validators ***

def boolean(v: Any) -> bool:
  return isinstance(v, bool)

def integer(v: Any) -> bool:
  return isinstance(v, int) and not isinstance(v, bool)

def number(v: Any) -> bool:
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def regex(pattern: str) -> Validator:
  p = re.compile(pattern, re.ASCII)
  return lambda v: isinstance(v, str) and p.fullmatch(v) is not None

def one_of(*choices: Any) -> Validator:
  return lambda v: any(type(v) is type(c) and v == c for c in choices)

def optional(validator: Validator) -> Validator:
  return lambda v: v is None or validator(v)

def either(*validators: Validator) -> Validator:
  return lambda v: any(validator(v) for validator in validators)

def list_of(validator: Validator) -> Validator:
  return lambda v: isinstance(v, list) and all(validator(x) for x in v)

def dict_of(key: Validator, value: Validator) -> Validator:
  return lambda v: isinstance(v, dict) and all(key(k) and value(x) for k, x in v.items())

def struct(**fields: Validator) -> Validator:
  return lambda v: isinstance(v, dict) and all(k in fields and fields[k](x) for k, x in v.items())


class Dispatcher(dict[str, Callable[..., Any]]):
  """Every param of every method must have a validator, all input is rejected by default."""

  def __setitem__(self, name: str, fn: Callable[..., Any]) -> None:
    raise TypeError("register methods with add_method")

  def add_method(self, f: Callable[..., Any] | None = None, *, name: str | None = None, **validators: Validator):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
      method = name or fn.__name__
      sig = inspect.signature(fn)
      params = {k for k, p in sig.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
      if len(params) != len(sig.parameters) or params != set(validators):
        raise TypeError(f"{method}: params {sorted(sig.parameters)} need exactly one validator each, got {sorted(validators)}")

      def validated(*args, **kwargs):
        for k, v in sig.bind(*args, **kwargs).arguments.items():
          if not validators[k](v):
            raise TypeError(f"invalid param: {k}")
        return fn(*args, **kwargs)

      super(Dispatcher, self).__setitem__(method, validated)
      return fn
    return decorator(f) if f is not None else decorator


dispatcher = Dispatcher()


def dumps_call(method: str, params: Any = None, request_id: Any = None) -> str:
  msg: JsonDict = {"jsonrpc": JSONRPC_VERSION, "method": method, "id": request_id}
  if params is not None:
    msg["params"] = params
  return json.dumps(msg)


def dumps_result(request_id: Any, result: Any) -> str:
  return json.dumps({"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result})


def dumps_error(request_id: Any, message: str, code: int = SERVER_ERROR) -> str:
  return json.dumps({
    "jsonrpc": JSONRPC_VERSION,
    "id": request_id,
    "error": {"code": code, "message": message},
  })


def loads(raw: str | bytes) -> JsonDict:
  if isinstance(raw, bytes):
    raw = raw.decode()
  data = json.loads(raw)
  if not isinstance(data, dict):
    raise ValueError("message must be a JSON object")
  return data


def is_call(msg: JsonDict) -> bool:
  return "method" in msg


def is_response(msg: JsonDict) -> bool:
  return "id" in msg and ("result" in msg or "error" in msg)


def error_message(err: Any) -> str:
  """Normalize JSON-RPC object errors and plain-string errors."""
  if isinstance(err, str):
    return err
  if isinstance(err, dict):
    data = err.get("data")
    if isinstance(data, dict) and data.get("message"):
      return str(data["message"])
    if err.get("message") is not None:
      return str(err["message"])
  return str(err)


def _invoke(fn: Callable[..., Any], params: Any) -> Any:
  if params is None:
    return fn()
  if isinstance(params, dict):
    return fn(**params)
  if isinstance(params, (list, tuple)):
    return fn(*params)
  raise TypeError("params must be a list, object, or omitted")


def handle(raw: str | bytes | JsonDict, methods: MethodMap | None = None) -> str:
  methods = dispatcher if methods is None else methods

  try:
    msg = raw if isinstance(raw, dict) else loads(raw)
  except (TypeError, ValueError, UnicodeDecodeError):
    return dumps_error(None, "parse error", PARSE_ERROR)

  if not is_call(msg):
    raise ValueError("not a call")

  req_id = msg.get("id")
  name = msg.get("method")
  if not isinstance(name, str):
    return dumps_error(req_id, "invalid request", INVALID_REQUEST)

  try:
    fn = methods[name]
  except KeyError:
    return dumps_error(req_id, f"method not found: {name}", METHOD_NOT_FOUND)

  try:
    return dumps_result(req_id, _invoke(fn, msg.get("params")))
  except TypeError as e:
    return dumps_error(req_id, str(e), INVALID_PARAMS)
  except Exception as e:
    return dumps_error(req_id, str(e), SERVER_ERROR)
