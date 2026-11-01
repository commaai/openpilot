#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3, difflib, warnings, functools, base64, codecs
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Callable, Any

ASSERT_DIFF = int((flag:="[PR]") in os.getenv("COMMIT_MESSAGE", flag) or flag in os.getenv("PR_TITLE", flag))
if not int(os.getenv("ASSERT_PROCESS_REPLAY", "1")): ASSERT_DIFF = 0

try:
  from tinygrad.renderer import Renderer
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm
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
if REF == "update_benchmark": SKIP_PROCESS_REPLAY = "[pr]" not in os.getenv("COMMIT_MESSAGE", "").lower()
class ProcessReplayWarning(Warning): pass

# *** replay the function and convert return values to string

def replay_to_program(p:UOp, ast:UOp, renderer:Renderer) -> tuple[str, str, tuple[Any, ...]]:
  if ast.op is Ops.PROGRAM: input_ast = ast
  else:
    sink_arg = ast.arg
    if sink_arg.beam: sink_arg = replace(sink_arg, opts_to_apply=p.src[0].arg.applied_opts)
    input_ast = ast.replace(arg=replace(sink_arg, name=p.src[0].arg.name))
  p2 = to_program(input_ast, renderer=renderer)
  device = renderer.target.device
  def to_str(ret:UOp) -> str:
    src = next(x.arg for x in ret.src if x.op is Ops.SOURCE)
    # PYTHON renderer pickles UOps, first unpickle and decode here
    # a nested PROGRAM prints its LINEAR, its SOURCE/BINARY are pickles of that LINEAR whose bytes depend on object identity
    if device.startswith("PYTHON"):
      uops = pickle.loads(base64.b64decode(src))
      pickles = {u:u.replace(arg=None) for u in UOp.sink(*uops).toposort(enter_calls=True) if u.op in (Ops.SOURCE, Ops.BINARY)}
      return "\n".join([str(x.substitute(pickles, enter_calls=True)) for x in uops])
    return src
  # properly color the name arg
  ast_repr = codecs.decode(str(input_ast), "unicode_escape")
  return to_str(p2), to_str(p), (ast_repr, renderer)

replayers: dict[str, Callable[..., tuple[str, str, tuple[Any, ...]]]] = {}
replayers["do_to_program"] = replay_to_program

# *** run replayers on captured rows and print diffs

def diff(offset:int, fxns:dict[str, Callable[..., tuple|None]], table_name=TABLE_NAME) -> None:
  if ASSERT_DIFF: warnings.filterwarnings("error", category=ProcessReplayWarning)
  if early_stop.is_set(): return None
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{table_name}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    if changed > MAX_DIFF_PCT:
      warnings.warn(f"detected changes in over {MAX_DIFF_PCT}%. skipping further diff generation.", ProcessReplayWarning)
      early_stop.set()
      break
    name, loc = "", ""
    try:
      name, args, kwargs, ctx_vals, ret = pickle.loads(row[0])
      loc = ctx_vals["PROCESS_REPLAY_LOC"].value
      ctx_vars = {k:v.value for k,v in ctx_vals.items() if k not in ("DEBUG", "CAPTURE_PROCESS_REPLAY")
                  and (var:=ContextVar._cache.get(k)) is not None and var.value != v.value}
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

def _pmap(fxns:dict[str, Callable], table_name:str=TABLE_NAME) -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{table_name}'").fetchone()[0]
  except sqlite3.OperationalError:
    raise RuntimeError(f"{table_name} isn't accessible in master, did DB_VERSION change?")
  finally:
    cur.close()

  with ProcessPoolExecutor(multiprocessing.cpu_count(), mp_context=multiprocessing.get_context("spawn")) as pool:
    with tqdm(total=row_count) as bar:
      for _ in pool.map(functools.partial(diff, fxns=fxns, table_name=table_name), range(0, row_count, s:=min(PAGE_SIZE, row_count))):
        bar.update(min(s, row_count-bar.n))

# *** main loop

def main(fxns:dict[str, Callable], table_name:str=TABLE_NAME) -> None:
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  logging.info(f"running process replay with {ASSERT_DIFF=}")
  try: _pmap(fxns, table_name)
  except Exception as e:
    logging.info(f"process replay err: {e}")
    exit(int(ASSERT_DIFF))

if __name__ == "__main__":
  main(replayers)
