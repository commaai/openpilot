### old can parser just used by plant.py for regression tests
import os
import opendbc
from collections import defaultdict

from selfdrive.car.honda.hondacan import fix
from common.realtime import sec_since_boot
from common.dbc import dbc

class CANParser():
  def __init__(self, dbc_f, signals, checks=None):
    ### input:
    # dbc_f   : dbc file
    # signals : List of tuples (name, address, ival) where
    #             - name is the signal name.
    #             - address is the corresponding message address.
    #             - ival is the initial value.
    # checks  : List of pairs (address, frequency) where
    #             - address is the message address of a message for which health should be
    #               monitored.
    #             - frequency is the frequency at which health should be monitored.

    checks = [] if checks is None else checks
    self.msgs_ck = {check[0] for check in checks}
    self.frqs = dict(checks)
    self.can_valid = False  # start with False CAN assumption
    # list of received msg we want to monitor counter and checksum for
    # read dbc file
    self.can_dbc = dbc(os.path.join(opendbc.DBC_PATH, dbc_f))
    # initialize variables to initial values
    self.vl = {}    # signal values
    self.ts = {}    # time stamp recorded in log
    self.ct = {}    # current time stamp
    self.ok = {}    # valid message?
    self.cn = {}    # message counter
    self.cn_vl = {} # message counter mismatch value
    self.ck = {}    # message checksum status

    for _, addr, _ in signals:
      self.vl[addr] = {}
      self.ts[addr] = {}
      self.ct[addr] = sec_since_boot()
      self.ok[addr] = True
      self.cn[addr] = 0
      self.cn_vl[addr] = 0
      self.ck[addr] = False

    for name, addr, ival in signals:
      self.vl[addr][name] = ival
      self.ts[addr][name] = 0

    self._msgs = [s[1] for s in signals]
    self._sgs = [s[0] for s in signals]

    self._message_indices = defaultdict(list)
    for i, x in enumerate(self._msgs):
      self._message_indices[x].append(i)

  def update_can(self, can_recv):
    msgs_upd = []
    cn_vl_max = 5   # no more than 5 wrong counter checks

    self.sec_since_boot_cached = sec_since_boot()

    # we are subscribing to PID_XXX, else data from USB
    for msg, ts, cdat, _ in can_recv:
      idxs = self._message_indices[msg]
      if idxs:
        msgs_upd.append(msg)
        # read the entire message
        out = self.can_dbc.decode((msg, 0, cdat))[1]
        # checksum check
        self.ck[msg] = True
        if "CHECKSUM" in out.keys() and msg in self.msgs_ck:
          # remove checksum (half byte)
          ck_portion = cdat[:-1] + (cdat[-1] & 0xF0).to_bytes(1, 'little')
          # recalculate checksum
          msg_vl = fix(ck_portion, msg)
          # compare recalculated vs received checksum
          if msg_vl != cdat:
            print("CHECKSUM FAIL: {0}".format(hex(msg)))
            self.ck[msg] = False
            self.ok[msg] = False
        # counter check
        cn = 0
        if "COUNTER" in out.keys():
          cn = out["COUNTER"]
        # check counter validity if it's a relevant message
        if cn != ((self.cn[msg] + 1) % 4) and msg in self.msgs_ck and "COUNTER" in out.keys():
          #print("FAILED COUNTER: {0}".format(hex(msg)()
          self.cn_vl[msg] += 1   # counter check failed
        else:
          self.cn_vl[msg] -= 1   # counter check passed
        # message status is invalid if we received too many wrong counter values
        if self.cn_vl[msg] >= cn_vl_max:
          print("COUNTER WRONG: {0}".format(hex(msg)))
          self.ok[msg] = False

        # update msg time stamps and counter value
        self.ct[msg] = self.sec_since_boot_cached
        self.cn[msg] = cn
        self.cn_vl[msg] = min(max(self.cn_vl[msg], 0), cn_vl_max)

        # set msg valid status if checksum is good and wrong counter counter is zero
        if self.ck[msg] and self.cn_vl[msg] == 0:
          self.ok[msg] = True

        # update value of signals in the
        for ii in idxs:
          sg = self._sgs[ii]
          self.vl[msg][sg] = out[sg]
          self.ts[msg][sg] = ts

    # for each message, check if it's too long since last time we received it
    self._check_dead_msgs()

    # assess overall can validity: if there is one relevant message invalid, then set can validity flag to False
    self.can_valid = True

    if False in self.ok.values():
      #print("CAN INVALID!")
      self.can_valid = False

    return msgs_upd

  def _check_dead_msgs(self):
    ### input:
    ## simple stuff for now: msg is not valid if a message isn't received for 10 consecutive steps
    for msg in set(self._msgs):
      if msg in self.msgs_ck and self.sec_since_boot_cached - self.ct[msg] > 10./self.frqs[msg]:
        self.ok[msg] = False
