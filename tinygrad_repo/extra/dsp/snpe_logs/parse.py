di = open("dlc_info_2").read().split("\n")
layers = {}
for l in di:
  if not l.startswith("| "): continue
  if l.startswith("|     |"): continue
  ll = [x.strip() for x in l.split("|")]
  if ll[1] == "Id": continue
  layers[int(ll[1])] = (ll[2], ll[6])
hp = open("high_perf_2").read().split("Layer Times:")[1].strip().split("\n")[2:]
sl = 1
tms = 0
for l in hp:
  kk, tm, _ = l.split(" ", 2)
  tm = int(tm)
  lnum = int(kk.strip(":"))
  if int(tm) != 0:
    print(f"{sl:2d} {tm:4d} us {layers[lnum]}")
    tms += tm
    sl += 1
print(f"total time, {tms/1000:.2f} ms")

