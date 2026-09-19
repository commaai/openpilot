import sys, pickle, decimal, json
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent
from tinygrad.helpers import tqdm, temp, ProfileEvent, ProfileRangeEvent, TracingKey

devices:dict[str, tuple[decimal.Decimal, int]] = {}
def prep_ts(device:str, ts:decimal.Decimal): return int(decimal.Decimal(ts) + devices[device][0])
def dev_to_pid(device:str): return {"pid": devices[device][1], "tid": 0}
def dev_ev_to_perfetto_json(ev:ProfileDeviceEvent):
  devices[ev.device] = (ev.tdiff, len(devices))
  return [{"name": "process_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "args": {"name": ev.device}},
          {"name": "thread_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "tid": 0, "args": {"name": ev.device}}]
def range_ev_to_perfetto_json(ev:ProfileRangeEvent):
  name = ev.name.display_name if isinstance(ev.name, TracingKey) else ev.name
  return [{"name": name, "ph": "X", "ts": prep_ts(ev.device, ev.st), "dur": float(ev.en-ev.st), **dev_to_pid(ev.device)}]
def graph_ev_to_perfetto_json(ev:ProfileGraphEvent, reccnt):
  ret = []
  for i,e in enumerate(ev.ents):
    st, en = ev.sigs[e.st_id], ev.sigs[e.en_id]
    name = e.name.display_name if isinstance(e.name, TracingKey) else e.name
    ret += [{"name": name, "ph": "X", "ts": prep_ts(e.device, st), "dur": float(en-st), **dev_to_pid(e.device)}]
    for dep in ev.deps[i]:
      d = ev.ents[dep]
      ret += [{"ph": "s", **dev_to_pid(d.device), "id": reccnt+len(ret), "ts": prep_ts(d.device, ev.sigs[d.en_id]), "bp": "e"}]
      ret += [{"ph": "f", **dev_to_pid(e.device), "id": reccnt+len(ret)-1, "ts": prep_ts(e.device, st), "bp": "e"}]
  return ret
def to_perfetto(profile:list[ProfileEvent]):
  # Start json with devices.
  profile += [ProfileDeviceEvent("TINY")]

  prof_json = [x for ev in profile if isinstance(ev, ProfileDeviceEvent) for x in dev_ev_to_perfetto_json(ev)]
  for ev in tqdm(profile, desc="preparing profile"):
    if isinstance(ev, ProfileRangeEvent): prof_json += range_ev_to_perfetto_json(ev)
    elif isinstance(ev, ProfileGraphEvent): prof_json += graph_ev_to_perfetto_json(ev, reccnt=len(prof_json))
  return {"traceEvents": prof_json}

if __name__ == "__main__":
  fp = sys.argv[1]
  with open(fp, "rb") as f: profile = pickle.load(f)
  ret = to_perfetto(profile)
  with open(fp:=temp("perfetto.json", append_user=True), "w") as f: json.dump(ret, f)
  print(f"Saved perfetto output to {fp}. You can use upload this to the perfetto UI or Chrome devtools.")
