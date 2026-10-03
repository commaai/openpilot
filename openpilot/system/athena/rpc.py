import json
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


class Dispatcher(dict[str, Callable[..., Any]]):
  def add_method(self, f: Callable[..., Any] | None = None, *, name: str | None = None):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
      self[name or fn.__name__] = fn
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
