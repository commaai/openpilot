#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3, difflib, warnings, itertools, functools, base64, codecs
from dataclasses import replace
from typing import Callable, Any

ASSERT_DIFF = int((flag:="[pr]") in os.getenv("COMMIT_MESSAGE", flag) or flag in os.getenv("PR_TITLE", flag))
if not int(os.getenv("ASSERT_PROCESS_REPLAY", "1")): ASSERT_DIFF = 0

try:
  from tinygrad.schedule.rangeify import get_rangeify_map
  from tinygrad.renderer import Renderer, ProgramSpec
  from tinygrad.engine.realize import get_program
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.codegen.opt import Opt
  from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm, BEAM
except ImportError as e:
  print(repr(e))
  exit(int(ASSERT_DIFF))

# *** process replay settings

# internal
PAGE_SIZE = getenv("PAGE_SIZE", 100)
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
TABLE_NAME = f"process_replay_{VERSION}"
os.environ["CAPTURE_PROCESS_REPLAY"] = "0"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format="%(message)s")
MAX_LINES = 500
def trunc_log(x):
  if len(lines:=(x if isinstance(x, str) else repr(x)).splitlines()) > MAX_LINES:
    lines = lines[:MAX_LINES]+[f"WARN: truncated string with {len(lines)} lines"]
  logging.info("\n".join(lines))

# user config
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
# uncomment this to disable by default
#SKIP_PROCESS_REPLAY = not ASSERT_DIFF and not ((k:="[p]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", ""))
if REF == "master": SKIP_PROCESS_REPLAY = True
class ProcessReplayWarning(Warning): pass

# *** replay the function and convert return values to string

def replay_get_rangeify_map(ret:dict[UOp, UOp], big_sink:UOp) -> tuple[str, str, tuple[Any, ...]]:
  UOp.unique_num = itertools.count(max([u.arg for u in big_sink.toposort() if u.op is Ops.UNIQUE], default=0)+1)
  new_sink = big_sink.substitute(get_rangeify_map(big_sink))
  def to_str(ret:UOp) -> str:
    asts = [repr(u.arg.ast) for u in ret.toposort() if u.op is Ops.KERNEL]
    return "\n".join([f"{len(asts)} kernels", *asts])
  return to_str(new_sink), to_str(big_sink.substitute(ret)), (big_sink,)

def replay_get_program(p:ProgramSpec, ast:UOp, renderer:Renderer, opts:list[Opt]|None=None) -> tuple[str, str, tuple[Any, ...]]:
  # the ast.arg is non None if we are inside of search.py
  sink_arg = ast.arg or KernelInfo(opts_to_apply=tuple(opts) if opts is not None else p.applied_opts if BEAM>=1 else None)
  input_ast = ast if ast.op is Ops.PROGRAM else ast.replace(arg=replace(sink_arg, name=p.name))
  p2 = get_program(input_ast, renderer=renderer)
  def to_str(ret:ProgramSpec) -> str:
    # PYTHON renderer pickles UOps, first unpickle and decode here
    if p.device.startswith("PYTHON"): return "\n".join([str(x) for x in pickle.loads(base64.b64decode(ret.src))])
    return ret.src
  # properly color the name arg
  ast_repr = codecs.decode(str(input_ast), "unicode_escape")
  return to_str(p2), to_str(p), (ast_repr, renderer)

replayers: dict[str, Callable[..., tuple[str, str, tuple[Any, ...]]]] = {}
replayers["get_program"] = replay_get_program
# disable this for speed, does it ever find things?
#replayers["get_rangeify_map"] = replay_get_rangeify_map

# *** run replayers on captured rows and print diffs

def diff(offset:int, fxns:dict[str, Callable[..., tuple|None]]) -> None:
  if ASSERT_DIFF: warnings.filterwarnings("error", category=ProcessReplayWarning)
  if early_stop.is_set(): return None
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    if changed > MAX_DIFF_PCT:
      warnings.warn(f"detected changes in over {MAX_DIFF_PCT}%. skipping further diff generation.", ProcessReplayWarning)
      early_stop.set()
      break
    name, loc = "", ""
    try:
      name, args, kwargs, ctx_vals, loc, ret = pickle.loads(row[0])
      ctx_vars = {k:v.value for k,v in ctx_vals.items() if k != "DEBUG" and (var:=ContextVar._cache.get(k)) is not None and var.value != v.value}
      if (replayer:=fxns.get(name)) is None: continue
      with Context(**ctx_vars):
        if (ret:=replayer(ret, *args, **kwargs)) is None: continue
        good, compare, metadata = ret
      if good != compare:
        for m in metadata: trunc_log(m)
        logging.info(loc)
        for line in difflib.unified_diff(good.splitlines(), compare.splitlines()):
          logging.info(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
        if ctx_vars: logging.info(ctx_vars)
        warnings.warn("PROCESS REPLAY DETECTED CHANGE", ProcessReplayWarning)
    except Exception as e:
      changed += 1
      warnings.warn(f"{name=} {loc=} {e=}", ProcessReplayWarning)
  cur.close()

# *** generic runner to map rows of a table to a function in parallel

def _pmap(fxns:dict[str, Callable]) -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    raise RuntimeError(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
  finally:
    cur.close()

  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count()) as pool:
    bar = tqdm(total=row_count)
    for _ in pool.imap_unordered(functools.partial(diff, fxns=fxns), range(0, row_count, s:=min(PAGE_SIZE, row_count))): bar.update(s)
    pool.close()
    pool.join()
    pool.terminate()

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  logging.info(f"running process replay with {ASSERT_DIFF=}")
  try: _pmap(replayers)
  except Exception as e:
    logging.info(f"process replay err: {e}")
    exit(int(ASSERT_DIFF))
