# extra/weekly_commits_table.py
import os, subprocess, datetime as dt

NAMES = ["chenyu","George Hotz","nimlgen","qazal","wozeparrot","Christopher Milan"]
REPO  = os.environ.get("REPO_PATH",".")
today = dt.date.today()
days  = [(today - dt.timedelta(i)).strftime("%Y-%m-%d") for i in range(6,-1,-1)]
seen  = {d:{n:False for n in NAMES} for d in days}

cmd = ["git","-C",REPO,"log","--use-mailmap","--since=7 days ago","--no-merges",
       "--date=short","--pretty=%ad%x09%aN%x09%ae"]
out = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
for line in out:
  try: d, name, email = line.split("\t")
  except: continue
  if d in seen:
    low = (name+" "+email).lower()
    for n in NAMES:
      if n.lower() in low: seen[d][n] = True

# --- width-aware padding so emoji align ---
try:
  from wcwidth import wcswidth as _wcswidth
  vlen = lambda s: _wcswidth(s)
except Exception:
  vlen = lambda s: sum(2 if ch in "✅❌" else 1 for ch in s)
pad  = lambda s,w: s + " " * max(0, w - vlen(s))

w_date = 10
w_cols = [max(3, vlen(n)) for n in NAMES]

header = " | ".join([pad("date", w_date)] + [pad(n, w_cols[i]) for i,n in enumerate(NAMES)])
rule   = "-+-".join(["-"*w_date] + ["-"*w for w in w_cols])

rows=[]
for d in days:
  cells = ["✅" if seen[d][n] else "❌" for n in NAMES]
  rows.append(" | ".join([pad(d, w_date)] + [pad(c, w_cols[i]) for i,c in enumerate(cells)]))

print("** Commits by day (last 7) **")
print("```")
print("\n".join([header, rule] + rows))
print("```")
