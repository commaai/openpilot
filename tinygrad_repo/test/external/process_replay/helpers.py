from dataclasses import dataclass
import traceback, subprocess
from typing import Dict, Optional, Tuple
from tinygrad.helpers import ContextVar, getenv

@dataclass(frozen=True)
class ProcessReplayContext:
  loc: str
  head_sha: str
  run_id: Optional[int]
def get_process_replay_ctx() -> Tuple[ProcessReplayContext, Dict]:
  stack = filter(lambda x: "tinygrad" in x.filename and not any(n in x.filename for n in ["engine/schedule.py", "engine/realize.py", \
      "codegen/kernel.py", "unittest"]), traceback.extract_stack()[:-1])
  loc = "\n".join(traceback.format_list(stack))
  try: head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
  except Exception: head_sha = ""
  return ProcessReplayContext(loc, head_sha, getenv("GITHUB_RUN_ID") or None), {k:v.value for k,v in ContextVar._cache.items()}
