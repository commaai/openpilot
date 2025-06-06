#!/usr/bin/env python3
import os, sys
import token
import tokenize
import itertools
from tabulate import tabulate

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]

def is_docstring(t):
  return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')

def is_js_token(s): return len(s) and not s.startswith('//')

def gen_stats(base_path="."):
  table = []
  for path, _, files in os.walk(os.path.join(base_path, "tinygrad")):
    for name in files:
      if not (name.endswith(".py") or name.endswith(".js")): continue
      if any(s in path.replace('\\', '/') for s in ['tinygrad/runtime/autogen', 'tinygrad/viz/assets']): continue
      filepath = os.path.join(path, name)
      relfilepath = os.path.relpath(filepath, base_path).replace('\\', '/')
      if name.endswith(".js"):
        with open(filepath) as file_: lines = [line.strip() for line in file_.readlines()]
        token_count, line_count = sum(len(line.split()) for line in lines if is_js_token(line)), sum(1 for line in lines if is_js_token(line))
      else:
        with tokenize.open(filepath) as file_:
          tokens = [t for t in tokenize.generate_tokens(file_.readline) if t.type in TOKEN_WHITELIST and not is_docstring(t)]
          token_count, line_count = len(tokens), len(set([x for t in tokens for x in range(t.start[0], t.end[0]+1)]))
      if line_count > 0: table.append([relfilepath, line_count, token_count/line_count])
  return table

def gen_diff(table_old, table_new):
  table = []
  files_new = set([x[0] for x in table_new])
  files_old = set([x[0] for x in table_old])
  added, deleted, unchanged = files_new - files_old, files_old - files_new, files_new & files_old
  if added:
    for file in added:
      file_stat = [stats for stats in table_new if file in stats]
      table.append([file_stat[0][0], file_stat[0][1], file_stat[0][1]-0, file_stat[0][2], file_stat[0][2]-0])
  if deleted:
    for file in deleted:
      file_stat = [stats for stats in table_old if file in stats]
      table.append([file_stat[0][0], 0, 0 - file_stat[0][1], 0, 0-file_stat[0][2]])
  if unchanged:
    for file in unchanged:
      file_stat_old = [stats for stats in table_old if file in stats]
      file_stat_new = [stats for stats in table_new if file in stats]
      if file_stat_new[0][1]-file_stat_old[0][1] != 0 or file_stat_new[0][2]-file_stat_old[0][2] != 0:
        table.append([file_stat_new[0][0], file_stat_new[0][1], file_stat_new[0][1]-file_stat_old[0][1], file_stat_new[0][2],
                      file_stat_new[0][2]-file_stat_old[0][2]])
  return table

def display_diff(diff): return "+"+str(diff) if diff > 0 else str(diff)

if __name__ == "__main__":
  if len(sys.argv) == 3:
    headers = ["Name", "Lines", "Diff", "Tokens/Line", "Diff"]
    table = gen_diff(gen_stats(sys.argv[1]), gen_stats(sys.argv[2]))
  elif len(sys.argv) == 2:
    headers = ["Name", "Lines", "Tokens/Line"]
    table = gen_stats(sys.argv[1])
  else:
    headers = ["Name", "Lines", "Tokens/Line"]
    table = gen_stats(".")

  if table:
    if len(sys.argv) == 3:
      print("### Changes")
      print("```")
      print(tabulate([headers] + sorted(table, key=lambda x: -x[1]), headers="firstrow", intfmt=(..., "d", "+d"),
                     floatfmt=(..., ..., ..., ".1f", "+.1f"))+"\n")
      print(f"\ntotal lines changes: {display_diff(sum([x[2] for x in table]))}")
      print("```")
    else:
      print(tabulate([headers] + sorted(table, key=lambda x: -x[1]), headers="firstrow", floatfmt=".1f")+"\n")
      groups = sorted([('/'.join(x[0].rsplit("/", 1)[0].split("/")[0:2]), x[1], x[2]) for x in table])
      for dir_name, group in itertools.groupby(groups, key=lambda x:x[0]):
        print(f"{dir_name:30s} : {sum([x[1] for x in group]):6d}")
      total_lines = sum([x[1] for x in table])
      print(f"\ntotal line count: {total_lines}")
      max_line_count = int(os.getenv("MAX_LINE_COUNT", "-1"))
      assert max_line_count == -1 or total_lines <= max_line_count, f"OVER {max_line_count} LINES"
