#!/usr/bin/env python3
"""Lightweight modem manager replacing ModemManager for openpilot.

Direct AT command control over serial. Supports:
  - Quectel EG25-G (comma 3X / tizi) — LTE Cat 4, ECM data mode
  - Cavli / EG916 (comma four / mici) — LTE Cat 1 bis, RMNET data mode

State is published to /dev/shm/modem_state.json for other processes.
Serial port access is coordinated via file lock at /tmp/modem_at.lock.
"""

import fcntl
import json
import os
import re
import subprocess
import threading
import time

import serial

from openpilot.common.swaglog import cloudlog

MODEM_STATE_FILE = "/dev/shm/modem_state.json"
MODEM_LOCK = "/tmp/modem_at.lock"
LTE_METRIC = 1000
POLL_INTERVAL = 10
CONNECT_RETRY = 30


def find_modem_iface():
  for iface in ('wwan0', 'usb0', 'usb1'):
    if os.path.exists(f'/sys/class/net/{iface}'):
      return iface
  return None


def read_modem_state():
  try:
    with open(MODEM_STATE_FILE) as f:
      return json.load(f)
  except Exception:
    return None


class ModemPort:
  """Context manager providing locked serial port access for AT commands.

  Uses file-based locking so multiple processes (modem daemon, qcomgpsd,
  LPA) can safely share the same physical AT port.
  """

  def __init__(self, port, timeout=0.5):
    self.port = port
    self.timeout = timeout
    self._ser = None
    self._lf = None

  def __enter__(self):
    self._lf = open(MODEM_LOCK, 'w')
    fcntl.flock(self._lf, fcntl.LOCK_EX)
    self._ser = serial.Serial(self.port, 115200, timeout=self.timeout)
    self._ser.reset_input_buffer()
    return self

  def __exit__(self, *a):
    if self._ser:
      self._ser.close()
    if self._lf:
      fcntl.flock(self._lf, fcntl.LOCK_UN)
      self._lf.close()

  def cmd(self, at_cmd, timeout=1.0):
    """Send an AT command and return the full response text, or None."""
    self._ser.reset_input_buffer()
    self._ser.write(f"{at_cmd}\r\n".encode())
    buf, deadline = b"", time.monotonic() + timeout
    while time.monotonic() < deadline:
      chunk = self._ser.read(max(1, self._ser.in_waiting))
      if chunk:
        buf += chunk
        text = buf.decode('utf-8', errors='ignore')
        if '\nOK' in text or '\nERROR' in text or '+CME ERROR' in text:
          return text.strip()
      time.sleep(0.01)
    return buf.decode('utf-8', errors='ignore').strip() if buf else None

  def upload_file(self, local_path, remote_name):
    """Upload binary file to modem via AT+QFUPL (Quectel)."""
    with open(local_path, 'rb') as f:
      data = f.read()
    self._ser.reset_input_buffer()
    self._ser.write(f'AT+QFUPL="{remote_name}",{len(data)},60\r\n'.encode())
    buf, deadline = b"", time.monotonic() + 5
    while time.monotonic() < deadline:
      buf += self._ser.read(max(1, self._ser.in_waiting))
      if b"CONNECT" in buf:
        break
    else:
      return False
    self._ser.write(data)
    buf, deadline = b"", time.monotonic() + 30
    while time.monotonic() < deadline:
      buf += self._ser.read(max(1, self._ser.in_waiting))
      if b"OK" in buf:
        return True
      if b"ERROR" in buf:
        return False
    return False


def at_cmd(cmd, port='/dev/ttyUSB2', timeout=1.0):
  """Send a single AT command (standalone, for use by other processes)."""
  try:
    with ModemPort(port, timeout=max(0.5, timeout)) as m:
      return m.cmd(cmd, timeout=timeout)
  except Exception:
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse(resp, prefix):
  if not resp:
    return ''
  for line in resp.split('\n'):
    s = line.strip()
    if prefix and s.startswith(prefix):
      return s[len(prefix):].strip().strip('"')
    if not prefix and s and s not in ('OK', 'ERROR') and not s.startswith('AT'):
      return s.strip().strip('"')
  return ''


def _find_port():
  for port in ('/dev/ttyUSB3', '/dev/ttyUSB2'):
    if not os.path.exists(port):
      continue
    r = at_cmd("AT", port)
    if r and "OK" in r:
      return port
  return None


def _setup_data(port, iface):
  """Activate PDP context and bring up the host network interface."""
  try:
    with ModemPort(port) as m:
      m.cmd('AT+CGDCONT=1,"IPV4V6",""', timeout=5)
      m.cmd("AT+CGATT=1", timeout=30)
      r = m.cmd("AT+CGACT=1,1", timeout=30)
      if r and "ERROR" in r:
        if "+CGACT: 1,1" not in (m.cmd("AT+CGACT?") or ""):
          return

    subprocess.run(["sudo", "ip", "link", "set", iface, "up"],
                   capture_output=True, timeout=5)

    dhcp_ok = subprocess.run(
      ["sudo", "udhcpc", "-i", iface, "-n", "-q", "-t", "5"],
      capture_output=True, timeout=30
    ).returncode == 0

    if not dhcp_ok:
      subprocess.run(["sudo", "dhclient", "-1", "-timeout", "10", iface],
                     capture_output=True, timeout=30)

    _fix_route(iface)
    cloudlog.info(f"Data connection up on {iface}")
  except Exception:
    cloudlog.exception("Data setup failed")


def _fix_route(iface):
  """Ensure cellular default route has metric > WiFi so WiFi wins."""
  try:
    r = subprocess.run(["ip", "route", "show", "default", "dev", iface],
                       capture_output=True, text=True, timeout=5)
    for line in r.stdout.strip().split('\n'):
      if not line or f'metric {LTE_METRIC}' in line:
        continue
      subprocess.run(f"sudo ip route del default dev {iface}".split(),
                     capture_output=True, timeout=5)
      gw = re.search(r'via (\S+)', line)
      cmd = ["sudo", "ip", "route", "add", "default"]
      if gw:
        cmd += ["via", gw.group(1)]
      cmd += ["dev", iface, "metric", str(LTE_METRIC)]
      subprocess.run(cmd, capture_output=True, timeout=5)
  except Exception:
    pass


# ---------------------------------------------------------------------------
# Modem daemon — the main loop replacing ModemManager
# ---------------------------------------------------------------------------

def modem_daemon(end_event: threading.Event, device_type: str):
  """Manage the modem lifecycle. Started as a thread from hardwared."""
  state = dict(
    initialized=False, connected=False, registered=False,
    technology='', operator='', band='', channel=0,
    signal_quality=0, network_type='none',
    revision='', imei='', sim_id='', mcc_mnc='',
    sim_present=False, temperatures=[], extra='',
    state_name='unknown', manufacturer='',
  )
  port = None
  configured = False
  last_connect = 0.0

  def _write():
    try:
      tmp = MODEM_STATE_FILE + ".tmp"
      with open(tmp, 'w') as f:
        json.dump(state, f)
      os.replace(tmp, MODEM_STATE_FILE)
    except Exception:
      pass

  while not end_event.is_set():
    try:
      if not port or not os.path.exists(port):
        port = _find_port()
        if not port:
          end_event.wait(5)
          continue

      with ModemPort(port) as m:
        # --- initialise ---
        if not state['initialized']:
          m.cmd("ATE0")
          m.cmd("AT+CMEE=1")
          if "+CFUN: 0" in (m.cmd("AT+CFUN?") or ""):
            m.cmd("AT+CFUN=1", timeout=10)
            time.sleep(3)

          state['revision'] = _parse(m.cmd("AT+CGMR"), '') or _parse(m.cmd("AT+CGMR"), '+CGMR:')
          state['manufacturer'] = _parse(m.cmd("AT+CGMI"), '')
          state['imei'] = _parse(m.cmd("AT+CGSN"), '')

          resp = m.cmd("AT+CPIN?")
          state['sim_present'] = resp is not None and "READY" in resp
          if state['sim_present']:
            iccid = _parse(m.cmd("AT+CCID"), "+CCID:") or _parse(m.cmd("AT+QCCID"), "+QCCID:")
            state['sim_id'] = iccid
            imsi = _parse(m.cmd("AT+CIMI"), '')
            if len(imsi) >= 5:
              state['mcc_mnc'] = imsi[:5]

          state['initialized'] = True
          cloudlog.info(f"Modem init: rev={state['revision']} imei={state['imei']}")

        # --- device-specific configuration (once) ---
        if not configured:
          if device_type == 'tizi':
            for c in ('AT+QSIMDET=1,0', 'AT+QSIMSTAT=1',
                       'AT+QNVW=5280,0,"0102000000000000"',
                       'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
                       'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01'):
              m.cmd(c, timeout=2)
            r = m.cmd('AT+QCFG="usbnet"')
            if r and '"usbnet",0' in r:
              cloudlog.warning("Switching EG25 to ECM mode")
              m.cmd('AT+QCFG="usbnet",1')
              m.cmd("AT+CFUN=1,1", timeout=5)
              state['initialized'] = False
              end_event.wait(10)
              port = None
              continue

          elif state['manufacturer'] == 'Cavli Inc.':
            for c in ('AT^SIMSWAP=1', 'AT$QCSIMSLEEP=0',
                       'AT$QCSIMCFG=SimPowerSave,0',
                       'AT$QCPCFG=usbNet,0', 'AT$QCNETDEVCTL=3,1'):
              m.cmd(c, timeout=2)

          else:
            if not state['sim_id']:
              for c in ('AT$QCSIMSLEEP=0', 'AT$QCSIMCFG=SimPowerSave,0',
                         'AT$QCPCFG=usbNet,1'):
                m.cmd(c, timeout=2)

          configured = True
          cloudlog.info("Modem configured")

        # --- poll registration ---
        state['registered'] = False
        for at_c, pfx in (("AT+CEREG?", "+CEREG:"), ("AT+CREG?", "+CREG:")):
          r = m.cmd(at_c)
          if r:
            mt = re.search(rf'{re.escape(pfx)} \d,(\d)', r)
            if mt and int(mt.group(1)) in (1, 5):
              state['registered'] = True
              state['state_name'] = 'REGISTERED' if mt.group(1) == '1' else 'ROAMING'
              break
        if not state['registered']:
          state['state_name'] = 'SEARCHING'

        # --- signal quality (3GPP standard) ---
        r = m.cmd("AT+CSQ")
        if r:
          mt = re.search(r'\+CSQ: (\d+),', r)
          if mt:
            rssi = int(mt.group(1))
            state['signal_quality'] = 0 if rssi == 99 else min(100, int(rssi / 31.0 * 100))

        # --- Quectel-specific info (EG25 only) ---
        if device_type == 'tizi':
          r = m.cmd("AT+QNWINFO")
          if r and "+QNWINFO:" in r:
            parts = r.split("+QNWINFO:")[1].split('\n')[0].replace('"', '').strip().split(',')
            if len(parts) >= 4:
              state['technology'] = parts[0].strip()
              state['operator'] = parts[1].strip()
              state['band'] = parts[2].strip()
              try:
                state['channel'] = int(parts[3].strip())
              except ValueError:
                pass
              tech = state['technology'].upper()
              if 'LTE' in tech:
                state['network_type'] = '4G'
              elif any(t in tech for t in ('WCDMA', 'HSDPA', 'HSUPA', 'HSPA', 'UMTS')):
                state['network_type'] = '3G'
              elif any(t in tech for t in ('GSM', 'GPRS', 'EDGE')):
                state['network_type'] = '2G'

          r = m.cmd('AT+QENG="servingcell"')
          if r and "+QENG:" in r:
            state['extra'] = (r.split("+QENG:")[1].split('\n')[0]
                              .replace('"servingcell",', '').replace('"', '').strip())

        # --- temperature ---
        r = m.cmd("AT+QTEMP")
        if r and "+QTEMP:" in r:
          try:
            t = r.split("+QTEMP:")[1].split('\n')[0].strip()
            state['temperatures'] = [int(x) for x in t.split(',')
                                     if x.strip().isdigit() and int(x.strip()) != 255]
          except Exception:
            pass

      # --- data connection (outside lock) ---
      iface = find_modem_iface()
      if iface:
        try:
          r = subprocess.run(["ip", "addr", "show", iface],
                             capture_output=True, text=True, timeout=3)
          state['connected'] = "inet " in r.stdout and "UP" in r.stdout
        except Exception:
          state['connected'] = False
      else:
        state['connected'] = False

      now = time.monotonic()
      if not state['connected'] and state['registered'] and now - last_connect > CONNECT_RETRY:
        last_connect = now
        _setup_data(port, iface or 'wwan0')

      _write()

    except Exception:
      cloudlog.exception("Modem daemon error")
      state['initialized'] = False
      configured = False

    end_event.wait(POLL_INTERVAL)
