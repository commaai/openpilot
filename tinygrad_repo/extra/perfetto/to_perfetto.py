import sys, pickle, decimal, json
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent
from tinygrad.helpers import tqdm, temp, ProfileEvent, ProfileRangeEvent, TracingKey

devices:dict[str, tuple[decimal.Decimal, decimal.Decimal, int]] = {}
def prep_ts(device:str, ts:decimal.Decimal, is_copy): return int(decimal.Decimal(ts) + devices[device][is_copy])
def dev_to_pid(device:str, is_copy=False): return {"pid": devices[device][2], "tid": int(is_copy)}
def dev_ev_to_perfetto_json(ev:ProfileDeviceEvent):
  devices[ev.device] = (ev.comp_tdiff, ev.copy_tdiff if ev.copy_tdiff is not None else ev.comp_tdiff, len(devices))
  return [{"name": "process_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "args": {"name": ev.device}},
          {"name": "thread_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "tid": 0, "args": {"name": "COMPUTE"}},
          {"name": "thread_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "tid": 1, "args": {"name": "COPY"}}]
def range_ev_to_perfetto_json(ev:ProfileRangeEvent):
  name = ev.name.display_name if isinstance(ev.name, TracingKey) else ev.name
  return [{"name": name, "ph": "X", "ts": prep_ts(ev.device, ev.st, ev.is_copy), "dur": float(ev.en-ev.st), **dev_to_pid(ev.device, ev.is_copy)}]
def graph_ev_to_perfetto_json(ev:ProfileGraphEvent, reccnt):
  ret = []
  for i,e in enumerate(ev.ents):
    st, en = ev.sigs[e.st_id], ev.sigs[e.en_id]
    name = e.name.display_name if isinstance(e.name, TracingKey) else e.name
    ret += [{"name": name, "ph": "X", "ts": prep_ts(e.device, st, e.is_copy), "dur": float(en-st), **dev_to_pid(e.device, e.is_copy)}]
    for dep in ev.deps[i]:
      d = ev.ents[dep]
      ret += [{"ph": "s", **dev_to_pid(d.device, d.is_copy), "id": reccnt+len(ret), "ts": prep_ts(d.device, ev.sigs[d.en_id], d.is_copy), "bp": "e"}]
      ret += [{"ph": "f", **dev_to_pid(e.device, e.is_copy), "id": reccnt+len(ret)-1, "ts": prep_ts(e.device, st, e.is_copy), "bp": "e"}]
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
