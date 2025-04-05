files = ["./tinygrad/viz/js/index.js", "./tinygrad/viz/js/worker.js"]
for fp in files:
  with open(fp) as f: content = f.read()
  cnt = 0
  for i,line in enumerate(content.splitlines()):
    if not (line:=line.strip()) or line.startswith("//"): continue
    #print(f"{i} {line}")
    cnt += 1
  print(f"{fp.split('/')[-1]} - {cnt} lines")
