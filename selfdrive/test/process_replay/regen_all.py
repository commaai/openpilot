#!/usr/bin/env python3
from selfdrive.test.process_replay.regen import regen_and_save
from selfdrive.test.process_replay.test_processes import original_segments as segments

if __name__ == "__main__":
  new_segments = []
  for segment in segments:
    route = segment[1].rsplit('--', 1)[0]
    sidx = int(segment[1].rsplit('--', 1)[1])
    relr = regen_and_save(route, sidx, upload=True, use_route_meta=False)

    print("\n\n", "*"*30, "\n\n")
    print("New route:", relr, "\n")
    relr = relr.replace('/', '|')
    new_segments.append(f'("{segment[0]}", "{relr}"), ')
  print()
  print()
  print()
  print('COPY THIS INTO test_processes.py')
  for seg in new_segments:
    print(seg)
