#!/usr/bin/env python3
import fcntl
import json
import logging
import os
import queue
import serial
import signal
import subprocess
import tempfile
import termios
import threading
import time

from enum import Enum

log = logging.getLogger("modem")
if not log.handlers:
  _h = logging.StreamHandler()
  _h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
  log.addHandler(_h)
  log.setLevel(logging.DEBUG)
  log.propagate = False

AT_PORT = "/dev/modem_at0"
PPP_PORT = "/dev/modem_at1"
STATE_PATH = "/dev/shm/modem"
AT_LOCK = "/dev/shm/modem_lpa.lock"  # shared with LPA
AT_INIT = ["ATE0", "ATV1", "AT+CMEE=1", "ATX4", "AT&C1"]
CREG = {0: "not_registered", 1: "home", 2: "searching", 3: "denied", 4: "unknown", 5: "roaming"}
PPPD = [
  "sudo", "pppd", PPP_PORT, "460800", "noauth", "nodetach", "noipdefault", "usepeerdns",
  "nodefaultroute", "connect",
  "/usr/sbin/chat -v ABORT 'NO CARRIER' ABORT 'NO DIALTONE' ABORT 'BUSY' " +
  "ABORT 'NO ANSWER' ABORT 'ERROR' TIMEOUT 5 '' AT OK ATD*99***1# CONNECT ''",
  "lcp-echo-interval", "30", "lcp-echo-failure", "4", "mtu", "1500", "mru", "1500",
  "novj", "novjccomp", "ipcp-accept-local", "ipcp-accept-remote", "nomagic",
  "user", '""', "password", '""',
]


class State(Enum):
  WAITING_PORT = "waiting_port"
  INIT = "init"
  REGISTERING = "registering"
  CONNECTING = "connecting"
  CONNECTED = "connected"
  RECONNECTING = "reconnecting"


class Modem:
  def __init__(self):
    self._ser = None
    self._ppp = None
    self._ppp_lines = queue.Queue()
    self._ppp_fails = 0
    self._sim_change = False
    self._apn = ""
    self._roaming_allowed = True
    self._ps_wait_start = 0.0
    self._ps_attach_tried = False
    self.running = True
    self.S = {
      "state": "init",
      "connected": False,
      "ip_address": "",
      "iccid": "",
      "imei": "",
      "modem_version": "",
      "signal_strength": 0,
      "signal_quality": 0,
      "network_type": "unknown",
      "operator": "",
      "band": "",
      "channel": 0,
      "registration": "unknown",
      "temperatures": [],
      "extra": "",
      "tx_bytes": 0,
      "rx_bytes": 0,
      "error": "",
      "carrier_error": "",
    }

  @staticmethod
  def _read_param(key):
    try:
      with open(f"/data/params/d/{key}") as f:
        return f.read().strip()
    except FileNotFoundError:
      return ""

  def _update(self, **kwargs):
    """Update state and atomically write to disk."""
    self.S.update(kwargs)
    fd = tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False)
    json.dump(self.S, fd)
    fd.flush()
    os.fsync(fd.fileno())
    os.chmod(fd.name, 0o644)
    tmp = fd.name
    fd.close()
    os.replace(tmp, STATE_PATH)

  # -- AT commands --

  def _at(self, cmd):
    """Send AT command, return response lines. [] on error or if LPA holds port."""
    fd = os.open(AT_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      os.close(fd)
      return []
    try:
      self._ser.write((cmd + "\r").encode())
      lines = []
      while True:
        raw = self._ser.readline()
        if not raw:
          raise TimeoutError("AT timeout")
        line = raw.decode(errors="ignore").strip()
        if not line:
          continue
        if line == "OK":
          break
        if line == "ERROR" or line.startswith("+CME ERROR"):
          raise RuntimeError(line)
        if "+QUSIM:" in line:
          log.warning(f"URC: {line}")
          self._sim_change = True
          subprocess.run(["sudo", "killall", "-9", "pppd"], capture_output=True)
        lines.append(line)
      return lines
    except (RuntimeError, TimeoutError, OSError, serial.SerialException) as e:
      log.info(f"AT {cmd} failed: {e}")
      return []
    finally:
      fcntl.flock(fd, fcntl.LOCK_UN)
      os.close(fd)

  def _atv(self, cmd, pfx):
    for line in self._at(cmd):
      if pfx in line and ":" in line:
        return line.split(":", 1)[1].strip()
    return None

  # -- teardown helpers --

  def _kill_ppp(self):
    subprocess.run(["sudo", "killall", "-9", "pppd"], capture_output=True)
    if self._ppp:
      try:
        self._ppp.wait(timeout=5)
      except subprocess.TimeoutExpired:
        pass
      self._ppp = None

  def _cleanup_routes(self):
    subprocess.run(["sudo", "ip", "route", "del", "default", "dev", "ppp0"], capture_output=True)
    subprocess.run(["sudo", "ip", "route", "flush", "table", "1000"], capture_output=True)
    while True:
      r = subprocess.run(["ip", "rule", "show", "table", "1000"], capture_output=True, text=True, timeout=2)
      if not r.stdout.strip():
        break
      subprocess.run(["sudo", "ip", "rule", "del", "table", "1000"], capture_output=True)

  @staticmethod
  def _reset_data_port():
    try:
      s = serial.Serial(PPP_PORT, 460800, timeout=1)
      attrs = termios.tcgetattr(s.fd)
      attrs[4] = attrs[5] = termios.B0
      termios.tcsetattr(s.fd, termios.TCSANOW, attrs)
      time.sleep(1)
      attrs[4] = attrs[5] = termios.B460800
      termios.tcsetattr(s.fd, termios.TCSANOW, attrs)
      s.reset_input_buffer()
      for cmd in AT_INIT:
        s.write((cmd + "\r").encode())
        time.sleep(0.1)
        s.read(100)
      s.close()
    except Exception as e:
      log.warning(f"data port reset failed: {e}")

  # -- state handlers --

  def _do_waiting_port(self):
    if os.path.exists(AT_PORT):
      log.info("port found")
      return State.INIT
    time.sleep(1)
    return State.WAITING_PORT

  def _do_init(self):
    log.info("opening serial port")
    if self._ser:
      try:
        self._ser.close()
      except Exception:
        pass
    try:
      self._ser = serial.Serial(AT_PORT, 9600, timeout=5)
    except (OSError, serial.SerialException) as e:
      log.warning(f"serial open failed: {e}")
      return State.WAITING_PORT
    time.sleep(1)

    # kill any stale pppd from previous run
    self._kill_ppp()
    self._cleanup_routes()

    # AT init
    for c in AT_INIT + ["AT$QCSIMSLEEP=0", "AT$QCSIMCFG=SimPowerSave,0", "AT+CREG=2", "AT+CGREG=2"]:
      self._at(c)

    # device-specific config
    try:
      with open("/sys/firmware/devicetree/base/model") as f:
        device = f.read().strip('\x00').split('comma ')[-1]
    except Exception:
      device = ""
    if device == "tizi":
      for c in [
        "AT+QSIMDET=1,0", "AT+QSIMSTAT=1",
        'AT+QNVW=5280,0,"0102000000000000"',
        'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
        'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
      ]:
        self._at(c)
    else:
      self._at('AT$QCPCFG=usbNet,1')

    # read identity
    imei, iccid, modem_version = "", "", ""
    r = self._at("AT+CGSN")
    if r:
      imei = r[0].strip()
    v = self._atv("AT+QCCID", "+QCCID:")
    if v:
      iccid = v
    r = self._at("AT+GMR")
    if r:
      modem_version = r[0].strip()
    log.info(f"imei={imei} iccid={iccid} ver={modem_version}")

    # configure APN on CID 1
    self._apn = self._read_param("GsmApn")
    self._roaming_allowed = self._read_param("GsmRoaming") != "0"
    self._at(f'AT+CGDCONT=1,"IP","{self._apn}"')
    log.info(f"APN '{self._apn or '(auto)'}' CID 1, roaming={'on' if self._roaming_allowed else 'off'}, sim_change={self._sim_change}")

    self._sim_change = False  # clear — we just re-read identity with the new SIM
    self._update(imei=imei, iccid=iccid, modem_version=modem_version)
    return State.REGISTERING

  def _do_registering(self):
    # check for param changes while waiting
    new_roaming = self._read_param("GsmRoaming") != "0"
    if new_roaming != self._roaming_allowed:
      log.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      self._roaming_allowed = new_roaming

    v = self._atv("AT+CREG?", "+CREG:")
    if v:
      try:
        reg = CREG.get(int(v.split(",")[1].strip('"')), "unknown")
      except (ValueError, IndexError):
        reg = "unknown"
      # also check packet-switched registration
      greg = "unknown"
      gv = self._atv("AT+CGREG?", "+CGREG:")
      if gv:
        try:
          greg = CREG.get(int(gv.split(",")[1].strip('"')), "unknown")
        except (ValueError, IndexError):
          pass
      log.debug(f"creg={reg} cgreg={greg} roaming_allowed={self._roaming_allowed}")

      # check roaming policy
      if reg in ("home", "roaming") and not self._roaming_allowed and reg == "roaming":
        self._update(registration=reg, error="roaming_disabled")
        time.sleep(0.5)
        return State.REGISTERING

      if reg in ("home", "roaming"):
        if greg in ("home", "roaming"):
          # both circuit and packet registered — ready to connect
          self._update(registration=reg, error="", carrier_error="")
          return State.CONNECTING

        # circuit registered but packet not attached
        if self._ps_wait_start == 0.0:
          self._ps_wait_start = time.monotonic()

        elapsed = time.monotonic() - self._ps_wait_start
        if elapsed > 10 and not self._ps_attach_tried:
          # mimic MM: force PS attach after 10s
          log.info(f"forcing PS attach (waited {elapsed:.0f}s)")
          self._at("AT+CGATT=1")
          self._ps_attach_tried = True
        elif elapsed > 30:
          # give up waiting for PS, try connecting anyway
          log.warning(f"PS attach timeout ({elapsed:.0f}s), proceeding anyway")
          self._update(registration=reg, error="", carrier_error="")
          return State.CONNECTING

      if reg != self.S.get("registration"):
        carrier_err = ""
        if reg in ("denied", "not_registered"):
          # query extended error reason — helps diagnose why the network rejected attach
          ceer = self._atv("AT+CEER", "+CEER:")
          if ceer:
            parts = [p.strip().strip('"') for p in ceer.split(",")]
            carrier_err = parts[-1] if parts else ceer
            # PLMN_NOT_ALLOWED often clears with the right APN — hint the user
            # (EPS_SERVICES_NOT_ALLOWED is a subscription/plan issue, APN won't help)
            if "PLMN_NOT_ALLOWED" in carrier_err:
              carrier_err += " (check GsmApn)"
        self._update(registration=reg, carrier_error=carrier_err)
    else:
      log.debug("CREG returned None")
      self._ps_wait_start = 0.0
      self._ps_attach_tried = False

    if self._sim_change or not os.path.exists(AT_PORT):
      log.info(f"-> reconnecting (sim_change={self._sim_change} port={os.path.exists(AT_PORT)})")
      return State.RECONNECTING
    time.sleep(0.5)
    return State.REGISTERING

  def _start_pppd(self):
    self._ppp = subprocess.Popen(PPPD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    self._ppp_lines = queue.Queue()

    def _read_pppd(proc, q):
      try:
        assert proc.stdout is not None
        for raw in proc.stdout:
          q.put(raw)
        proc.stdout.close()
      except Exception as e:
        log.warning(f"pppd reader error: {e}")

    threading.Thread(target=_read_pppd, args=(self._ppp, self._ppp_lines), daemon=True).start()
    log.info("PPP dialing")

  def _do_connecting(self):
    log.info("starting pppd")
    self._update(state="connecting")
    self._ppp_fails = 0
    self._sim_change = False
    self._ps_wait_start = 0.0
    self._ps_attach_tried = False
    self._start_pppd()
    return State.CONNECTED

  def _do_connected(self):
    # drain pppd output from queue
    while not self._ppp_lines.empty():
      try:
        raw = self._ppp_lines.get_nowait()
      except queue.Empty:
        break
      line = raw.decode(errors="ignore").strip()
      if not line:
        continue
      log.info(f"pppd: {line}")
      if "local  IP address" in line:
        ip = line.split("local  IP address")[-1].strip()
        self._update(ip_address=ip, connected=True, state="connected")
      elif "remote IP address" in line and self.S["ip_address"]:
        peer = line.split("remote IP address")[-1].strip()
        self._cleanup_routes()
        subprocess.run(["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "metric", "1000"],
                       capture_output=True)
        subprocess.run(["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "table", "1000"],
                       capture_output=True)
        subprocess.run(["sudo", "ip", "rule", "add", "from", self.S["ip_address"], "table", "1000"],
                       capture_output=True)
        log.info(f"route set up for {self.S['ip_address']} via {peer}")
      elif "Connection terminated" in line or "Modem hangup" in line:
        self._update(connected=False, state="disconnected", ip_address="")

    # check if pppd exited
    if self._ppp and self._ppp.poll() is not None:
      self._ppp = None
      if self._sim_change or not os.path.exists(AT_PORT):
        return State.RECONNECTING
      self._ppp_fails += 1
      if self._ppp_fails >= 3:
        log.warning(f"PPP fail {self._ppp_fails}/3, reconnecting")
        return State.RECONNECTING
      log.warning(f"PPP fail {self._ppp_fails}/3, retrying")
      self._reset_data_port()
      if not os.path.exists(AT_PORT):
        return State.RECONNECTING
      self._start_pppd()
      return State.CONNECTED

    # check for SIM change, port loss, or param changes
    if self._sim_change or not os.path.exists(AT_PORT):
      return State.RECONNECTING
    new_apn = self._read_param("GsmApn")
    if new_apn != self._apn:
      log.info(f"APN changed: '{self._apn}' -> '{new_apn}'")
      return State.RECONNECTING
    new_roaming = self._read_param("GsmRoaming") != "0"
    if new_roaming != self._roaming_allowed:
      log.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      return State.RECONNECTING

    # poll modem status
    self._poll()
    return State.CONNECTED

  def _do_reconnecting(self):
    log.warning("reconnecting")
    self._update(
      state="reconnecting", connected=False, ip_address="",
      iccid="", imei="", modem_version="",
      signal_strength=0, signal_quality=0,
      network_type="unknown", operator="", band="", channel=0,
      registration="unknown", temperatures=[], extra="",
      tx_bytes=0, rx_bytes=0, error="",
    )
    self._kill_ppp()
    self._cleanup_routes()
    self._reset_data_port()
    self._sim_change = False
    return State.WAITING_PORT

  # -- poll --

  def _poll(self):
    s = {}

    v = self._atv("AT+CSQ", "+CSQ:")
    if v:
      try:
        rssi = int(v.split(",")[0])
        if rssi != 99:
          s["signal_strength"] = rssi
          s["signal_quality"] = min(100, int(rssi / 31 * 100))
      except (ValueError, IndexError):
        pass

    v = self._atv("AT+COPS?", "+COPS:")
    if v:
      p = v.split(",")
      try:
        if len(p) >= 3:
          s["operator"] = p[2].strip('"')
        if len(p) >= 4:
          s["network_type"] = {0: "gsm", 2: "utran", 7: "lte"}.get(int(p[3]), "unknown")
      except (ValueError, IndexError):
        pass

    v = self._atv("AT+QNWINFO", "+QNWINFO:")
    if v:
      info = v.replace('"', '').split(",")
      try:
        if len(info) >= 4:
          s["band"] = info[2]
          s["channel"] = int(info[3])
      except (ValueError, IndexError):
        pass

    v = self._atv('AT+QENG="servingcell"', "+QENG:")
    if v:
      s["extra"] = v.replace('"', '')

    v = self._atv("AT+QTEMP", "+QTEMP:")
    if v:
      try:
        s["temperatures"] = [t for t in (int(x) for x in v.split(",") if x.strip()) if t != 255]
      except (ValueError, IndexError):
        pass

    # ppp0 interface status
    try:
      r = subprocess.run(["ip", "-4", "addr", "show", "ppp0"], capture_output=True, text=True, timeout=2)
      ip = next((l.strip().split()[1].split("/")[0] for l in r.stdout.splitlines() if "inet " in l), None)
      if ip:
        s.update(ip_address=ip, connected=True, state="connected")
      elif self.S["connected"]:
        s.update(connected=False, state="registered", ip_address="")
    except Exception:
      pass

    try:
      with open("/sys/class/net/ppp0/statistics/tx_bytes") as f:
        s["tx_bytes"] = int(f.read().strip())
      with open("/sys/class/net/ppp0/statistics/rx_bytes") as f:
        s["rx_bytes"] = int(f.read().strip())
    except Exception:
      pass

    self._update(**s)

  # -- main loop --

  def run(self):
    log.info("starting")
    # publish initial state so callers short-circuit MM DBus activation from the start
    self._update(state="init")
    subprocess.run(["sudo", "systemctl", "stop", "ModemManager"], capture_output=True)
    subprocess.run(["sudo", "killall", "pppd"], capture_output=True)

    state = State.INIT

    handlers = {
      State.WAITING_PORT: self._do_waiting_port,
      State.INIT: self._do_init,
      State.REGISTERING: self._do_registering,
      State.CONNECTING: self._do_connecting,
      State.CONNECTED: self._do_connected,
      State.RECONNECTING: self._do_reconnecting,
    }

    while self.running:
      try:
        prev = state
        state = handlers[state]()
        if state != prev:
          self._update(state=state.value)
          log.info(f"{prev.value} -> {state.value}")
      except Exception:
        log.exception(f"error in {state.value}")
        state = State.RECONNECTING
      if state not in (State.REGISTERING, State.WAITING_PORT):
        time.sleep(2)

  def stop(self):
    self.running = False
    self._kill_ppp()
    if self._ser:
      self._ser.close()
    self._cleanup_routes()
    try:
      os.remove(STATE_PATH)
    except FileNotFoundError:
      pass
    subprocess.run(["sudo", "systemctl", "unmask", "ModemManager"], capture_output=True)
    subprocess.run(["sudo", "systemctl", "start", "ModemManager"], capture_output=True)


def main():
  m = Modem()

  def _sig(*_):
    m.running = False

  signal.signal(signal.SIGINT, _sig)
  signal.signal(signal.SIGTERM, _sig)
  m.run()
  m.stop()


if __name__ == "__main__":
  main()
