#!/usr/bin/env python3
# DO NOT MERGE: live view of the MEB hold fuzz harness from a laptop, over ZMQ through the device's bridge.
#
#   device: cd /data/openpilot && ./openpilot/cereal/messaging/bridge
#   laptop: ZMQ=1 python openpilot/tools/meb_fuzz/watch.py <device ip> [--log out.jsonl]
#
# Prints one line whenever a hold-related signal changes, decoded from raw CAN with opendbc:
# hold manager (VMM_02.HMS_Status), Standstill, Motion_State, TSK_Status, camera ACC status, our ACC_18 request and
# accel, plus carState/carControl and the on-screen alertDebug text.
import argparse
import json
import os
import sys
import time
from datetime import datetime

from opendbc.can import CANParser
from openpilot.cereal import messaging

DBC = 'vw_meb_2024_generated'


def main():
  assert os.environ.get('ZMQ'), 'run with ZMQ=1 to connect to the device bridge'
  parser = argparse.ArgumentParser()
  parser.add_argument('addr', help='device IP')
  parser.add_argument('--log', help='append every printed state to this jsonl file')
  args = parser.parse_args()

  sm = messaging.SubMaster(['carState', 'carControl', 'alertDebug', 'selfdriveState'], addr=args.addr)
  can_sock = messaging.sub_sock('can', addr=args.addr, conflate=False, timeout=10)
  sendcan_sock = messaging.sub_sock('sendcan', addr=args.addr, conflate=False, timeout=10)
  pt = CANParser(DBC, [], 0)
  cam = CANParser(DBC, [], 2)
  tx = CANParser(DBC, [], 0)
  log = open(args.log, 'a') if args.log else None

  last_key, last_print = None, 0.
  while True:
    for m in messaging.drain_sock(can_sock):
      frames = [(c.address, bytes(c.dat), c.src) for c in m.can]
      pt.update((m.logMonoTime, [f for f in frames if f[2] == 0]))
      cam.update((m.logMonoTime, [f for f in frames if f[2] == 2]))
    for m in messaging.drain_sock(sendcan_sock):
      tx.update((m.logMonoTime, [(c.address, bytes(c.dat), 0) for c in m.sendcan if c.address == 333 and c.src == 0]))
    sm.update(0)

    cs, cc = sm['carState'], sm['carControl']
    state = {
      't': datetime.now().isoformat(timespec='milliseconds'),
      'v': round(cs.vEgo, 2), 'gas': cs.gasPressed, 'brake': cs.brakePressed, 'gear': str(cs.gearShifter),
      'accFaulted': cs.accFaulted, 'canValid': cs.canValid,
      'longActive': cc.longActive, 'lcs': str(cc.actuators.longControlState), 'cc_accel': round(cc.actuators.accel, 2),
      'hms': int(pt.vl['VMM_02']['HMS_Status']), 'inhibit': int(pt.vl['VMM_02']['Long_Control_Inhibit']),
      'ss': int(pt.vl['ESC_50']['Standstill']), 'motion': int(pt.vl['ESC_50']['Motion_State']),
      'epb': int(pt.vl['ESC_50']['EPB_Status']), 'tsk': int(pt.vl['Motor_51']['TSK_Status']),
      'cam': int(cam.vl['ACC_18']['ACC_Status_ACC']),
      'req': int(tx.vl['ACC_18']['ACC_Anforderung_HMS']), 'tx_accel': round(tx.vl['ACC_18']['ACC_Sollbeschleunigung_02'], 2),
      'anhalten': int(tx.vl['ACC_18']['ACC_Anhalten']), 'tx_status': int(tx.vl['ACC_18']['ACC_Status_ACC']),
      'alert': [sm['alertDebug'].alertText1, sm['alertDebug'].alertText2],
    }
    key = tuple(state[k] for k in ('hms', 'ss', 'motion', 'epb', 'tsk', 'cam', 'req', 'anhalten', 'tx_status', 'longActive',
                                   'lcs', 'gas', 'brake', 'accFaulted', 'canValid', 'gear'))
    now = time.monotonic()
    if key != last_key or now - last_print > 1.0:
      print(' '.join([
        state['t'][11:23],
        f"v={state['v']:5.2f} lcs={state['lcs']:8s} cc_a={state['cc_accel']:+.2f}",
        f"req={state['req']} tx_a={state['tx_accel']:+.2f} anh={state['anhalten']} |",
        f"hms={state['hms']} ss={state['ss']} mot={state['motion']} epb={state['epb']} tsk={state['tsk']} cam={state['cam']}",
        f"inh={state['inhibit']} |",
        f"la={int(state['longActive'])} gas={int(state['gas'])} brk={int(state['brake'])} fault={int(state['accFaulted'])}",
        f"valid={int(state['canValid'])} | {state['alert'][0]}",
      ]), flush=True)
      last_key, last_print = key, now
      if log:
        log.write(json.dumps(state) + '\n')
        log.flush()
    time.sleep(0.01)


if __name__ == '__main__':
  sys.exit(main())
