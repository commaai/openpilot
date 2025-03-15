#!/usr/bin/env python3
# compare kernels created by HEAD against master
from collections import defaultdict
import os, multiprocessing, logging, pickle, sqlite3, difflib, functools, warnings
from typing import Callable, List, Set, Tuple, Union, cast
from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm
from tinygrad.engine.schedule import ScheduleContext, schedule_uop
from tinygrad.codegen.kernel import Kernel, Opt
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp
from test.helpers import print_diff

# *** process replay settings

# internal
PAGE_SIZE = getenv("PAGE_SIZE", 100)
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
TABLE_NAME = f"process_replay_{VERSION}"
os.environ["RUN_PROCESS_REPLAY"] = "0"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format="%(message)s")

# user config
ASSERT_DIFF = int((flag:="[pr]") in os.getenv("COMMIT_MESSAGE", flag) or flag in os.getenv("PR_TITLE", flag))
if not getenv("ASSERT_PROCESS_REPLAY", 1): ASSERT_DIFF = 0
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
if REF == "master": SKIP_PROCESS_REPLAY = True
class ProcessReplayWarning(Warning): pass

# *** recreators

def recreate_sched(ast:UOp, assigns:Set[UOp]) -> UOp:
  # NOTE: process replay isn't meant to actually schedule anything
  return schedule_uop(ast, ScheduleContext(assigns=assigns, tensor_uops=defaultdict(list))).ast
def recreate_kernel(ast:UOp, opts:Renderer, applied_opts:List[Opt], name:str, _) -> str:
  k = Kernel(ast, opts=opts)
  for opt in applied_opts: k.apply_opt(opt)
  # NOTE: replay with the captured renderer, not the one in master
  return k.opts.render(name, cast(List,k.to_program().uops))

# *** diff a "good" recreation against the generated version

def diff(offset:int, name:str, fxn:Callable) -> Union[Tuple[int, int], bool]:
  if early_stop.is_set(): return True
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{name}_{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  additions, deletions, changed = 0, 0, 0
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
      with Context(**{k:v for k,v in args[-2].items() if k in ContextVar._cache and k != "DEBUG"}): good = fxn(*args[:-2])
      if good is None: continue
    except Exception as e:
      changed += 1
      warnings.warn(f"FAILED TO RECREATE KERNEL {e}", ProcessReplayWarning)
      for x in args[:-1]: logging.info(x)
      continue
    # diff kernels
    try: assert args[-1] == good
    except AssertionError:
      changed += 1
      logging.info("PROCESS REPLAY DETECTED CHANGE")
      for x in args[:-1]: logging.info(x)
      print_diff(good, args[-1])
      changes = list(difflib.unified_diff(str(good).splitlines(), str(args[-1]).splitlines()))
      additions += len([x for x in changes if x.startswith("+")])
      deletions += len([x for x in changes if x.startswith("-")])
      if ASSERT_DIFF: return additions, deletions
  conn.commit()
  cur.close()
  return additions, deletions

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
    ret: List[Union[bool, Tuple[int, int]]] = list(tqdm(pool.imap_unordered(functools.partial(diff, name=name, fxn=fxn), inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()
    changed = [bool(x[0] or x[1]) if isinstance(x, tuple) else x for x in ret]
    insertion, deletions = [x[0] for x in ret if isinstance(x, tuple)], [x[1] for x in ret if isinstance(x, tuple)]
    logging.info(f"{sum(changed)} kernels changed")
    if sum(insertion) != 0: logging.info(colored(f"{sum(insertion)} insertions(+)", "green"))
    if sum(deletions) != 0: logging.info(colored(f"{sum(deletions)} deletions(-)", "red"))
    if any(changed): warnings.warn("process replay detected changes", ProcessReplayWarning)

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  if ASSERT_DIFF: warnings.filterwarnings("error", category=ProcessReplayWarning)
  for name,fxn in [("schedule", recreate_sched), ("kernel", recreate_kernel)]:
    logging.info(f"***** {name} diff")
    try: _pmap(name, fxn)
    except Exception as e:
      if ASSERT_DIFF: raise e
      logging.error(f"{name} diff err {e}")
