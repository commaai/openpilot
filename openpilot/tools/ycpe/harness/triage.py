"""Which file a finding points at (from its text and the classes each module creates), and what changed since the last run."""
import re
from functools import cache
from pathlib import Path

YCPE = Path(__file__).resolve().parent.parent
SOURCES = [p for p in YCPE.rglob("*.js") if not {"harness"} & set(p.relative_to(YCPE).parts)]
CLASS_RE = re.compile(r"""class(?:Name)?\s*[:=]\s*[`"']([^`"']+)|classList\.(?:add|toggle)\(\s*["']([\w-]+)""")
LOCAL_FILE = re.compile(r"(?:localhost:\d+)?/((?:[\w-]+/)*[\w.-]+\.js)\b")


@cache
def class_files():
  """class name -> the module that creates it, for classes only one module uses"""
  seen = {}
  for p in SOURCES:
    for a, b in CLASS_RE.findall(p.read_text()):
      for c in (a or b).split():
        if re.fullmatch(r"[\w-]+", c):
          seen.setdefault(c, set()).add(str(p.relative_to(YCPE)))
  return {c: next(iter(f)) for c, f in seen.items() if len(f) == 1}


def owner_file(f):
  """the source file a finding points at: a local script in its text, the module that creates its innermost class, or a guess by host"""
  text = f"{f['key']} {f['detail']}"
  local = re.sub(r"https?://(?!localhost)\S+", "", text)
  if f["rule"] in ("long-frame", "console-error", "console-warning", "page-error") and (m := LOCAL_FILE.search(local)):
    return m.group(1)
  if f.get("selector"):
    for part in reversed(f["selector"].split(" > ")):
      for c in re.findall(r"\.([\w-]+)", part):
        if c in class_files():
          return class_files()[c]
  if "api.commadotai.com" in text:
    return "api.js"
  if re.search(r"cdn\.jsdelivr|api\.mapbox", text):
    return "index.html"
  return None


def annotate(found):
  for f in found:
    f["file"] = owner_file(f)


def diff(found, prev, ran):
  """new, fixed and still-open findings by id; prev is the previous run's findings.json (or []).
  Only findings from scenarios that ran can be fixed, so a -k run doesn't fix the rest."""
  before = {f["id"]: f for f in prev}
  now = {f["id"]: f for f in found}
  return {"new": [f for i, f in now.items() if i not in before],
          "fixed": [f for i, f in before.items() if i not in now and ran & set(f.get("scenarios", ran))],
          "still": [f for i, f in now.items() if i in before]}
