#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3, difflib, functools, warnings
from typing import Callable, cast
from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm, dedup
from tinygrad.engine.grouper import get_becomes_map
from tinygrad.codegen.kernel import Kernel, Opt
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, Ops

# *** process replay settings

# internal
PAGE_SIZE = getenv("PAGE_SIZE", 100)
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
TABLE_NAME = f"process_replay_{VERSION}"
os.environ["RUN_PROCESS_REPLAY"] = "0"
os.environ["CAPTURE_PROCESS_REPLAY"] = "0"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format="%(message)s")
MAX_LINES = 500
def trunc_log(x):
  if len(lines:=repr(x).splitlines()) > MAX_LINES: lines = lines[:MAX_LINES]+[f"WARN: truncated string with {len(lines)} lines"]
  logging.info("\n".join(lines))

# user config
ASSERT_DIFF = int((flag:="[pr]") in os.getenv("COMMIT_MESSAGE", flag) or flag in os.getenv("PR_TITLE", flag))
if not getenv("ASSERT_PROCESS_REPLAY", 1): ASSERT_DIFF = 0
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
if REF == "master": SKIP_PROCESS_REPLAY = True
class ProcessReplayWarning(Warning): pass

# *** recreators

def recreate_sched(big_sink:UOp) -> list[UOp]:
  becomes_map = get_becomes_map(big_sink)
  sched_sink = big_sink.substitute(becomes_map)
  return dedup(u.arg.ast for u in sched_sink.toposort() if u.op is Ops.KERNEL)

def recreate_kernel(ast:UOp, opts:Renderer, applied_opts:list[Opt], name:str, _) -> str:
  k = Kernel(ast, opts=opts)
  for opt in applied_opts: k.apply_opt(opt)
  # NOTE: replay with the captured renderer, not the one in master
  return k.opts.render(cast(list,k.to_program(name).uops))

# *** diff a "good" recreation against the generated version

def diff(offset:int, name:str, fxn:Callable) -> None:
  # TODO: add this assert back for schedule
  if ASSERT_DIFF and name != "schedule": warnings.filterwarnings("error", category=ProcessReplayWarning)
  if early_stop.is_set(): return None
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{name}_{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    if changed > MAX_DIFF_PCT:
      warnings.warn(f"detected changes in over {MAX_DIFF_PCT}% of {name}s. skipping further diff generation.")
      early_stop.set()
      break
    # try unpickle
    try: args = pickle.loads(row[0])
    except Exception as e:
      changed += 1
      warnings.warn(f"FAILED TO UNPICKLE OBJECTS {e}", ProcessReplayWarning)
      continue
    # try recreate
    try:
      ctx_vars = {k:v.value for k,v in args[-2].items() if k != "DEBUG" and (var:=ContextVar._cache.get(k)) is not None and var.value != v.value}
      with Context(**ctx_vars): good = fxn(*args[:-2])
      if good is None: continue
    except Exception as e:
      changed += 1
      if ctx_vars: logging.info(ctx_vars)
      for x in args[:-2]: trunc_log(x)
      warnings.warn(f"FAILED TO RECREATE KERNEL {e}", ProcessReplayWarning)
      continue
    # diff kernels
    try: assert str(args[-1]) == str(good)
    except AssertionError:
      changed += 1
      if ctx_vars: logging.info(ctx_vars)
      for x in args[:-2]: trunc_log(x)
      changes = list(difflib.unified_diff(str(good).splitlines(), str(args[-1]).splitlines()))
      logging.info("\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in changes))
      warnings.warn("PROCESS REPLAY DETECTED CHANGE", ProcessReplayWarning)
  conn.commit()
  cur.close()

# *** generic runner for executing fxn across all rows of a table in parallel

def _pmap(name:str, fxn:Callable, maxtasksperchild:int=16) -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{name}_{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    warnings.warn(f"{name}_{TABLE_NAME} isn't accessible in master, did DB_VERSION change?", ProcessReplayWarning)
    return None
  conn.commit()
  cur.close()
  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=maxtasksperchild) as pool:
    inputs = list(range(0, row_count, PAGE_SIZE))
    list(tqdm(pool.imap_unordered(functools.partial(diff, name=name, fxn=fxn), inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  print(f"running process replay with {ASSERT_DIFF=}")
  for name,fxn in [("schedule", recreate_sched), ("kernel", recreate_kernel)]:
    logging.info(f"***** {name} diff")
    try: _pmap(name, fxn)
    except Exception as e:
      if ASSERT_DIFF: raise e
      logging.error(f"{name} diff err {e}")
