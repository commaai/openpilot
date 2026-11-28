"""UIR: the text form of a UOp graph, one line per UOp: `%id = op src src key=value`, like studentgrad.

  %0 = param 4 slot=0 dtype=f32
  %1 = range 4 slot=0
  %2 = index %0 %1
  %3 = store %2 1.0
  %4 = end %3 %1
  %5 = sink %4 name=test

srcs are %ids or literals: a CONST (1, 2.5, -inf, true, invalid) or an all-CONST STACK ([1 2]) has no line. args are key=value after the srcs.
a value is a literal, a bare word, a 'string', a [list], a %id, or a `python object`. parse_uir(render_uir(x)) is x, since UOps are
hash-consed, when every python object's repr evaluates back to it. runtime objects aren't rendered: a BUFFER's Buffer and a CALL's grad_fxn.
parse_uir also reads studentgrad's hand-written UIR: %names, nested (op ...), [stacks] of expressions and {scoped blocks}.
"""
import re, math, ast, importlib
from collections import deque
from enum import Enum
from dataclasses import fields
from typing import Any, Callable
from tinygrad import dtype
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid
from tinygrad.uop import Ops, GroupOp, ops
from tinygrad.uop.ops import UOp, KernelInfo, CallInfo, NativeInfo, ProgramInfo, ParamArg, AxisType, StageOpts
from tinygrad.helpers import colored, ansistrip, Target

# ***** render *****

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               Ops.RANGE: "#c8a0e0", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#CEF9B7", Ops.STACK: "#D8F9E4",
               Ops.WMMA: "#efefc0", Ops.UNSHARD: "#f6ccff",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.BUFFER: "#B0BDFF", Ops.EXTERN: "#7FA8D8", Ops.GETADDR: "#9DB1F0", Ops.COPY: "#ff90c0", Ops.NATIVE: "#bf71b6",
               Ops.CALL: "#00B7C8", Ops.PARAM: "#14686F", Ops.SOURCE: "#c0c0c0", Ops.BINARY: "#404040",
               Ops.LINEAR: "#7DF4FF", Ops.ALLOC: "#C07788", Ops.PROGRAM: "#a0a0a0",
               Ops.ALLREDUCE: "#ff40a0", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0",
               Ops.STAGE: "#FFC14D", Ops.REWRITE_ERROR: "#1a1b26", Ops.AFTER: "#8A7866", Ops.END: "#524C46", Ops.BACKEDGE: "#464752"}

# keys whose value is written by its name: dtype=f32 addrspace=REG axis=LOOP op=add target=CPU
_named_keys = {"dtype", "addrspace", "axis", "op", "target"}
_keywords = {"true", "false", "inf", "nan", "invalid"}
# ops whose arg is one value with a name, and ops whose arg is a dataclass written as key=value per field
_arg_key = {Ops.CAST:"dtype", Ops.BITCAST:"dtype", Ops.COPY:"device", Ops.GETADDR:"device", Ops.SPECIAL:"name", Ops.UNSHARD:"axes",
            Ops.MSELECT:"index"}
_skip_fields = {"buffer", "grad_fxn"}   # runtime objects

_struct_args: dict[Ops, type] = {Ops.PARAM: ParamArg, Ops.REFPARAM: ParamArg, Ops.BUFFER: ParamArg, Ops.ALLOC: ParamArg, Ops.EXTERN: ParamArg,
                                 Ops.SINK: KernelInfo, Ops.CALL: CallInfo, Ops.NATIVE: NativeInfo, Ops.STAGE: StageOpts, Ops.PROGRAM: ProgramInfo}

def _inline(u:UOp) -> bool: return u.op is Ops.CONST or (u.op is Ops.STACK and all(s.op is Ops.CONST for s in u.src))

def _const_str(x:UOp) -> str:
  """CONSTs are weak or bool, so literals are bare. typed values are CASTs of these"""
  return str(x.val).lower()

def _val(v, named:bool, ref:Callable[[UOp], str|None]) -> str:
  """literals and one-word strings are bare, tuples are [a b], a UOp with a line is its %id, any other python object is `module:repr`"""
  if named and isinstance(v, (Enum, DType)): return v.name.lower() if isinstance(v, Ops) else v.name
  if named and isinstance(v, Target): return _val(repr(v), False, ref)
  if isinstance(v, bool): return str(v).lower()
  if type(v) in (int, float): return repr(v)   # exact types, a ConstFloat or an enum is a python object
  if isinstance(v, tuple): return "[" + " ".join(_val(x, False, ref) for x in v) + "]"
  if isinstance(v, UOp) and (r:=ref(v)) is not None: return r
  if isinstance(v, str): return v if re.fullmatch(r"[A-Za-z_][\w:.,+\-]*", v) and v not in _keywords else repr(v)
  # a python object's repr is evaluated with the names in the module of its type (rdna3 and rdna4 instructions have the same names)
  mod = "" if type(v).__module__ in ("builtins", "tinygrad.dtype", "tinygrad.uop.ops") else type(v).__module__ + ":"
  return "`" + (mod + repr(v)).replace("\n", " ").replace("`", "\\x60") + "`"   # newlines and backticks are only in string literals

def _fields_str(a, ref:Callable[[UOp], str|None], first:bool) -> str:
  ret = []
  for i,f in enumerate(fields(a)):
    v = getattr(a, f.name)
    # a default is left out, except the first field if first. compared by repr, since == of a UOp builds a UOp
    if f.name in _skip_fields or (repr(v) == repr(f.default) and not (first and i == 0)): continue
    ret.append(f"{f.name}={_val(v, f.name in _named_keys, ref)}")
  return " ".join(ret)

def _render_arg(x:UOp, ref:Callable[[UOp], str|None]=lambda _: None) -> str:
  """the args, written after the srcs"""
  if (a:=x.arg) is None: return ""
  if x.op is Ops.CONST: return _const_str(x)
  if x.op is Ops.RANGE:   # like UOp.range the axis defaults to LOOP. flatten_range merges slots: slot=[1 2]
    slot = a[1] if len(a) == 2 else a[1:]
    return f"slot={_val(slot, False, ref)}" + ("" if a[0] is AxisType.LOOP else f" axis={a[0].name}")
  if x.op is Ops.REDUCE: return f"op={a[0].name.lower()}" + (f" pop={a[1]}" if a[1] else "")
  if x.op is Ops.ALLREDUCE: return f"op={a[0].name.lower()} device={_val(a[1], False, ref)}"
  if type(a) is _struct_args.get(x.op):
    # the first field is always written, so a default arg isn't None. a CALL's arg is always a CallInfo
    return _fields_str(a, ref, first=x.op is not Ops.CALL)
  key = _arg_key.get(x.op, "arg")
  return f"{key}={_val(a, key in _named_keys, ref)}"

def render_uir(root:UOp|list[UOp], color:bool=True) -> str:
  """one line per UOp: %id = op src src key=value"""
  nodes = [u for u in (root.toposort(enter_calls=True) if isinstance(root, UOp) else root) if not _inline(u)]
  table = {u:i for i,u in enumerate(nodes)}
  def src_str(u:UOp) -> str:
    if not _inline(u): return f"%{table[u]}"
    return _const_str(u) if u.op is Ops.CONST else "[" + " ".join(src_str(s) for s in u.src) + "]"
  lines = []
  for i,u in enumerate(nodes):
    # a UOp in an arg is its %id if it has an earlier line
    arg = _render_arg(u, lambda v: f"%{table[v]}" if table.get(v, i) < i else None)
    op_str = colored(u.op.name.lower(), uops_colors.get(u.op)) if color else u.op.name.lower()
    lines.append(" ".join([f"%{i} = {op_str}", *map(src_str, u.src)] + ([arg] if arg else [])))
  # an inline root has no line of its own, so it's a literal value
  if isinstance(root, UOp) and _inline(root): lines.append(f"%{len(nodes)} = {src_str(root)}")
  return "\n".join(lines)

# ***** parse *****

# a token is a comment, an assignment '%name =', a key=, a %ref, a bracket, a quoted string, a `python object`, or a bare word
_tokens = re.compile(r"""#[^\n]*|%\w+\s*=|\w+=|%\w+|[\[\](){}]|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`[^`]*`|[^\s\[\](){}#]+""", re.ASCII)
_ops = {o.name.lower(): o for o in Ops}
_dtypes = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
_named_parse: dict[str, Callable[[str], Any]] = {"dtype": lambda v: _dtypes[v], "addrspace": lambda v: AddrSpace[v],
                                                  "axis": lambda v: AxisType[v], "op": lambda v: _ops[v]}

def _lit(t:str) -> Any:
  """literals: true false 0x2a -3 1.5 1e+20 -inf nan. anything else is a bare word"""
  if t in ("true", "false"): return t == "true"
  if re.fullmatch(r"0x[0-9a-fA-F]+", t): return int(t, 16)
  if re.fullmatch(r"-?[0-9]+", t): return int(t)
  if re.fullmatch(r"-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?|-?inf|nan", t): return float(t)
  return t

def _eval(expr:str) -> Any:
  """a `python object`: a repr, evaluated with the names in tinygrad.dtype, tinygrad.uop.ops and the module before a ':'. NOTE: this is eval"""
  ns = {**vars(dtype), **vars(ops), "inf": math.inf, "nan": math.nan}
  if (m:=re.match(r"([\w.]+):(?!=)", expr)):
    ns.update(vars(importlib.import_module(m[1])))
    expr = expr[m.end():]
  return eval(expr, ns)  # pylint: disable=eval-used

_tuple_keys = {Ops.RANGE: {"slot", "axis"}, Ops.REDUCE: {"op", "pop"}, Ops.ALLREDUCE: {"op", "device"}}

def _build(name:str, srcs:tuple[UOp, ...], kv:dict[str, Any]) -> UOp:
  """make the UOp for `name srcs key=value`. any op's arg can be a lone arg="""
  if (op:=_ops.get(name)) is None: raise ValueError(f"unknown op {name!r}")
  struct = _struct_args.get(op)
  keys = _tuple_keys.get(op) or ({f.name for f in fields(struct)} if struct else {_arg_key.get(op, "arg")})
  if set(kv) - keys and set(kv) != {"arg"}: raise ValueError(f"{name} has no {sorted(set(kv) - keys)}, its args are {sorted(keys)}")
  if "target" in kv: kv["target"] = Target.parse(kv["target"])   # a Target is written as its string
  arg: Any = None
  if set(kv) == {"arg"}: arg = kv["arg"]
  elif op is Ops.RANGE:
    slot = kv["slot"]
    arg = (kv.get("axis", AxisType.LOOP), *(slot if isinstance(slot, tuple) else (slot,)))
  elif op is Ops.REDUCE: arg = (kv["op"], kv.get("pop", 0))
  elif op is Ops.ALLREDUCE: arg = (kv["op"], kv["device"])
  elif struct is not None and (kv or op is Ops.CALL): arg = struct(**kv)   # a CALL always has a CallInfo
  elif kv: arg = kv[_arg_key[op]]
  return UOp(op, srcs, arg)

def parse_uir(text:str) -> UOp:
  """parse UIR text into a UOp graph: render_uir's output, or hand-written UIR like studentgrad's.
  a program is `%name = value` assignments and its root is the last one. a value is `op srcs key=value` (only right after the `=`),
  `(op srcs key=value)`, `[values]` for a STACK, `{assignments}` for the last of a scoped block, a %name, or a literal.
  NOTE: a `python object` is eval'd, only parse UIR you trust"""
  toks = deque(t for t in _tokens.findall(ansistrip(text)) if not t.startswith("#"))   # op names may carry ANSI color from render_uir
  def is_assign(t:str) -> bool: return t[0] == "%" and t[-1] == "="
  def at_end() -> bool: return not toks or toks[0] in (")", "]", "}") or is_assign(toks[0])
  def pop(what:str) -> str:
    if not toks: raise ValueError(f"expected {what}, got end of input")
    return toks.popleft()
  def ref(names:dict[str, UOp|None], t:str) -> UOp:
    if (u:=names.get(t)) is None: raise ValueError(f"{t} is not defined")
    return u
  def attr(names:dict[str, UOp|None], k:str) -> Any:
    t = pop(f"a value for {k}=")
    if t in ("(", ")", "]", "{", "}"): raise ValueError(f"expected a value for {k}=, got {t!r}")
    if t[0] in "`'\"" and (len(t) < 2 or t[-1] != t[0]): raise ValueError(f"unterminated {t[0]} in {k}=")
    if t == "[":
      ret = []
      while toks and toks[0] != "]": ret.append(attr(names, k))
      pop("']'")
      return tuple(ret)
    if t[0] == "`": return _eval(t[1:-1])
    if t[0] in "'\"": return ast.literal_eval(t)
    return ref(names, t) if t[0] == "%" else _named_parse.get(k, _lit)(t)
  def expr(names:dict[str, UOp|None], name:str, close:str|None) -> UOp:
    srcs: list[UOp] = []
    while not at_end() and not toks[0].endswith("="): srcs.append(value(names))
    kv: dict[str, Any] = {}
    while not at_end():
      k = pop("key=")
      if not k.endswith("="): raise ValueError(f"srcs must come before key=value args in {name}, got {k!r}")
      if k[:-1] in kv: raise ValueError(f"{name} has {k} twice")
      kv[k[:-1]] = attr(names, k[:-1])
    if close is not None and pop(repr(close)) != close: raise ValueError(f"expected {close!r} to end {name}")
    return _build(name, tuple(srcs), kv)
  def value(names:dict[str, UOp|None], bare=False) -> UOp:
    t = pop("a value")
    if t == "{": return scope(dict(names), "}")
    if t == "(": return expr(names, pop("an op"), ")")
    if t == "[":
      srcs = []
      while toks and toks[0] != "]": srcs.append(value(names))
      pop("']'")
      return UOp(Ops.STACK, tuple(srcs))
    if t[0] == "%": return ref(names, t)
    if t == "invalid": return UOp(Ops.CONST, arg=Invalid)
    if isinstance(v:=_lit(t), str):
      if bare and re.fullmatch(r"[a-z_0-9]+", t): return expr(names, t, None)   # an assignment's value can be a bare op
      raise ValueError(f"{t!r} is not a %name or a literal" + (f", write ({t} ...)" if t in _ops else ""))
    return UOp.const(v)
  def scope(names:dict[str, UOp|None], close:str|None) -> UOp:
    root: UOp|None = None
    while toks and toks[0] != "}":
      t = pop("'%name ='")
      if not is_assign(t): raise ValueError(f"expected '%name =', got {t!r}")
      name = t[:-1].rstrip()
      if name in names: raise ValueError(f"{name} is already defined")
      names[name] = None   # defined, so its value can't define it again, but not usable until its value is parsed
      names[name] = root = value(names, bare=True)
    if (toks.popleft() if toks else None) != close: raise ValueError(f"expected {close or 'end of input'!r}")
    if root is None: raise ValueError("expected an assignment")
    return root
  return scope({}, None)
