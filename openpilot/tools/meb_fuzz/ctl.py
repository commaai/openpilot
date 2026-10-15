#!/usr/bin/env python3
# DO NOT MERGE: edit the MEB hold fuzz harness on the device over ssh.
#
#   ctl.py <ip> program programs/e1_aborted_stop.json [--hold-source standstill] [--starting stock]
#   ctl.py <ip> live pid 0.3          drive directly while engaged (state, accel)
#   ctl.py <ip> live off              back to the program
#   ctl.py <ip> alert "line 1" "line 2"   override the on-screen text ("" "" to clear)
#   ctl.py <ip> set key value         set any top-level control key, value parsed as JSON
#   ctl.py <ip> off | on              disable / enable the harness (off = plain #3861 behavior)
#   ctl.py <ip> show | status | results [N] | reset
#
# Writes are atomic (write .tmp then mv), the harness picks them up within 0.5 s.
import argparse
import json
import subprocess
import sys

FUZZ_DIR = '/data/meb_fuzz'


def ssh(addr, cmd, data=None, user='comma'):
  r = subprocess.run(['ssh', '-o', 'ConnectTimeout=5', f'{user}@{addr}', cmd], input=data, capture_output=True, text=True)
  if r.returncode != 0:
    sys.exit(f'ssh failed: {r.stderr.strip()}')
  return r.stdout


def read_control(addr):
  out = ssh(addr, f'cat {FUZZ_DIR}/control.json 2>/dev/null || echo {{}}')
  return json.loads(out or '{}')


def write_control(addr, control):
  ssh(addr, f'mkdir -p {FUZZ_DIR} && cat > {FUZZ_DIR}/control.json.tmp && mv {FUZZ_DIR}/control.json.tmp {FUZZ_DIR}/control.json',
      data=json.dumps(control, indent=1))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('addr')
  p.add_argument('cmd')
  p.add_argument('args', nargs='*')
  p.add_argument('--hold-source', choices=('hms', 'standstill'))
  p.add_argument('--starting', choices=('master', 'stock', 'hold_first', 'hold_first2'))
  a = p.parse_args()

  if a.cmd in ('show', 'status'):
    path = 'control.json' if a.cmd == 'show' else 'status.json'
    print(ssh(a.addr, f'cat {FUZZ_DIR}/{path}'))
    return
  if a.cmd == 'results':
    n = int(a.args[0]) if a.args else 5
    print(ssh(a.addr, f'tail -n {n} {FUZZ_DIR}/results.jsonl'))
    return
  if a.cmd == 'reset':
    ssh(a.addr, f'rm -f {FUZZ_DIR}/state.json')
    return

  control = read_control(a.addr)
  if a.cmd == 'program':
    with open(a.args[0]) as f:
      control['program'] = json.load(f)
    control['enabled'] = True
    control['live'] = None
  elif a.cmd == 'live':
    control['live'] = None if a.args[0] == 'off' else {'state': a.args[0], 'accel': float(a.args[1])}
  elif a.cmd == 'alert':
    control['alert'] = a.args if any(a.args) else None
  elif a.cmd == 'set':
    control[a.args[0]] = json.loads(a.args[1])
  elif a.cmd in ('on', 'off'):
    control['enabled'] = a.cmd == 'on'
  else:
    sys.exit(f'unknown command {a.cmd}')
  if a.hold_source:
    control['hold_source'] = a.hold_source
  if a.starting:
    control['starting'] = a.starting
  write_control(a.addr, control)
  print(json.dumps(control, indent=1))


if __name__ == '__main__':
  main()
