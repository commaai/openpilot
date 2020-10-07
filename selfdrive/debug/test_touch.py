#!/usr/bin/env python3
import os
import evdev
import threading

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

print(f"Found {len(devices)} input devices")
for d in devices:
  print(""*4, d.path, d.name, d.phys)

print("starting read loops")

def read(d):
  for event in d.read_loop():
    print(event)

for d in devices:
  threading.Thread(target=read,args=(d,)).start()


