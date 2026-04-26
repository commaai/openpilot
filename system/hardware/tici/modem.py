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

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s.%(msecs)03d %(levelname)-7s modem: %(message)s",
  datefmt="%H:%M:%S",
)

AT_PORT = "/dev/modem_at0"
PPP_PORT = "/dev/modem_at1"
STATE_PATH = "/dev/shm/modem"
AT_LOCK = "/dev/shm/modem_lpa.lock"  # shared with LPA
AT_INIT = ["ATE0", "ATV1", "AT+CMEE=1", "ATX4", "AT&C1"]
CREG = {0: "not_registered", 1: "home", 2: "searching", 3: "denied", 4: "unknown", 5: "roaming"}

# error.type values published in the state file
ERR_NONE = ""
ERR_ROAMING_DISABLED = "roaming_disabled"
ERR_CARRIER_REJECT = "carrier_reject"
PPPD = [
  "sudo", "pppd", PPP_PORT, "460800", "noauth", "nodetach", "noipdefault", "usepeerdns",
  "nodefaultroute", "connect",
  "/usr/sbin/chat -v ABORT 'NO CARRIER' ABORT 'NO DIALTONE' ABORT 'BUSY' " +
  "ABORT 'NO ANSWER' ABORT 'ERROR' TIMEOUT 5 '' AT OK ATD*99***1# CONNECT ''",
  "lcp-echo-interval", "30", "lcp-echo-failure", "4", "mtu", "1500", "mru", "1500",
  "novj", "novjccomp", "ipcp-accept-local", "ipcp-accept-remote", "nomagic",
  "user", '""', "password", '""',
]
INITIAL_STATE = {
  "state": "INITIALIZING",
  "connected": False, "ip_address": "",
  "iccid": "", "mcc_mnc": "", "imei": "", "modem_version": "",
  "signal_strength": 0, "signal_quality": 0,
  "network_type": "unknown", "operator": "", "band": "", "channel": 0,
  "registration": "unknown", "temperatures": [], "extra": "",
  "tx_bytes": 0, "rx_bytes": 0, "error": {},
}


class State(Enum):
  WAITING_PORT = "waiting_port"
  INIT = "init"
  REGISTERING = "registering"
  CONNECTING = "connecting"
  CONNECTED = "connected"
  RECONNECTING = "reconnecting"

  @property
  def label(self) -> str:
    return {
      State.WAITING_PORT: "INITIALIZING",
      State.INIT: "INITIALIZING",
      State.REGISTERING: "SEARCHING",
      State.CONNECTING: "CONNECTING",
      State.CONNECTED: "CONNECTED",
      State.RECONNECTING: "DISCONNECTING",
    }[self]


# seconds to wait after each state handler returns; states omitted use STATE_WAIT_DEFAULT
STATE_WAIT_DEFAULT = 2.0
STATE_WAIT = {
  State.WAITING_PORT: 1.0,
  State.REGISTERING: 0.5,
}


class Modem:
  def __init__(self):
    self._ppp = None
    self._ppp_lines = queue.Queue()
    self._ppp_fails = 0
    self._sim_change = False
    self._apn = ""
    self._roaming_allowed = True
    # byte counters persist across pppd restarts (sysfs counters reset when ppp0 is recreated)
    self._tx_base = 0
    self._rx_base = 0
    self._last_tx = 0
    self._last_rx = 0
    self.running = True
    self.S = INITIAL_STATE.copy()

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
      with serial.Serial(AT_PORT, 9600, timeout=5) as ser:
        ser.reset_input_buffer()
        ser.write((cmd + "\r").encode())
        lines = []
        while True:
          raw = ser.readline()
          if not raw:
            raise TimeoutError("AT timeout")
          line = raw.decode(errors="ignore").strip()
          if not line:
            continue
          if line == "OK":
            break
          if line == "ERROR" or line.startswith("+CME ERROR"):
            raise RuntimeError(line)
          lines.append(line)
        return lines
    except (RuntimeError, TimeoutError, OSError, serial.SerialException) as e:
      logging.info(f"AT {cmd} failed: {e}")
      return []
    finally:
      fcntl.flock(fd, fcntl.LOCK_UN)
      os.close(fd)

  def _atv(self, cmd, pfx):
    for line in self._at(cmd):
      if pfx in line and ":" in line:
        return line.split(":", 1)[1].strip()
    return None

  @staticmethod
  def _parse_reg(v: str) -> str:
    try:
      return CREG.get(int(v.split(",")[1].strip('"')), "unknown")
    except (ValueError, IndexError):
      return "unknown"

  def _carrier_reject_error(self, reg: str) -> dict:
    if reg not in ("denied", "not_registered"):
      return {}
    ceer = self._atv("AT+CEER", "+CEER:")
    if not ceer:
      return {}
    parts = [p.strip().strip('"') for p in ceer.split(",")]
    msg = parts[-1] if parts else ceer
    # PLMN_NOT_ALLOWED often clears with the right APN; EPS_SERVICES_NOT_ALLOWED is a plan issue
    if "PLMN_NOT_ALLOWED" in msg:
      msg += " (check GsmApn)"
    return {"type": ERR_CARRIER_REJECT, "description": msg}

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
      logging.warning(f"data port reset failed: {e}")

  # -- state handlers --

  def _do_waiting_port(self):
    if os.path.exists(AT_PORT):
      logging.info("port found")
      return State.INIT
    return State.WAITING_PORT

  def _configure_modem(self, modem_version: str, sim_id: str):
    cmds: list[str] = []
    if modem_version.startswith("EG25"):
      cmds += [
        # clear old blue prime initial EPS bearer APN (was `mmcli --3gpp-set-initial-eps-bearer-settings="apn="`)
        'AT+CGDCONT=0,"IP",""',

        # SIM hot swap
        'AT+QSIMDET=1,0',
        'AT+QSIMSTAT=1',

        # configure modem as data-centric
        'AT+QNVW=5280,0,"0102000000000000"',
        'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
        'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
      ]
    elif modem_version.startswith("EG916") and not sim_id:
      # EG916 gets upset with too many AT commands; only run when no SIM is provisioned
      cmds += [
        # SIM sleep disable
        'AT$QCSIMSLEEP=0',
        'AT$QCSIMCFG=SimPowerSave,0',

        # ethernet config
        'AT$QCPCFG=usbNet,1',
      ]

    for c in cmds:
      self._at(c)

  def _do_init(self):
    logging.info("init")
    # kill any stale pppd from previous run
    self._kill_ppp()
    self._cleanup_routes()

    # AT init
    for c in AT_INIT + ["AT+CREG=2", "AT+CGREG=2"]:
      self._at(c)

    identity = self._read_identity()
    self._configure_modem(identity["modem_version"], identity["iccid"])
    self._configure_apn_and_roaming()

    self._sim_change = False  # clear — we just re-read identity with the new SIM
    self._update(**identity)
    return State.REGISTERING

  def _read_identity(self):
    imei, iccid, mcc_mnc, modem_version = "", "", "", ""
    r = self._at("AT+CGSN")
    if r:
      imei = r[0].strip()
    v = self._atv("AT+QCCID", "+QCCID:")
    if v:
      iccid = v
    r = self._at("AT+CIMI")
    if r:
      # IMSI = MCC (3 digits) + MNC (2 or 3 digits) + MSIN; we include both 5- and 6-digit forms in MNC,
      # but standard convention for mcc_mnc is first 5-6 digits — use 6 by default, consumers can truncate
      imsi = r[0].strip()
      if len(imsi) >= 6 and imsi.isdigit():
        mcc_mnc = imsi[:6]
    r = self._at("AT+GMR")
    if r:
      modem_version = r[0].strip()
    logging.info(f"imei={imei} iccid={iccid} mcc_mnc={mcc_mnc} ver={modem_version}")
    return {"imei": imei, "iccid": iccid, "mcc_mnc": mcc_mnc, "modem_version": modem_version}

  def _configure_apn_and_roaming(self):
    self._apn = self._read_param("GsmApn")
    self._roaming_allowed = self._read_param("GsmRoaming") != "0"
    self._at(f'AT+CGDCONT=1,"IP","{self._apn}"')
    logging.info(f"APN '{self._apn or '(auto)'}' CID 1, roaming={'on' if self._roaming_allowed else 'off'}, sim_change={self._sim_change}")

  def _refresh_roaming_param(self):
    new_roaming = self._read_param("GsmRoaming") != "0"
    if new_roaming != self._roaming_allowed:
      logging.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      self._roaming_allowed = new_roaming

  def _do_registering(self):
    self._refresh_roaming_param()

    v = self._atv("AT+CREG?", "+CREG:")
    if not v:
      return self._registering_idle()

    reg = self._parse_reg(v)
    greg = self._parse_reg(self._atv("AT+CGREG?", "+CGREG:") or "")
    logging.debug(f"creg={reg} cgreg={greg} roaming_allowed={self._roaming_allowed}")

    if reg == "roaming" and not self._roaming_allowed:
      self._update(registration=reg, error={"type": ERR_ROAMING_DISABLED,
                                            "description": "roaming blocked by GsmRoaming param"})
      return State.REGISTERING

    if reg in ("home", "roaming") and greg in ("home", "roaming"):
      self._update(registration=reg, error={})
      return State.CONNECTING

    # not connectable yet — record current reg and any carrier reject reason
    if reg != self.S.get("registration"):
      self._update(registration=reg, error=self._carrier_reject_error(reg))

    return self._registering_idle()

  def _registering_idle(self):
    if self._sim_change or not os.path.exists(AT_PORT):
      logging.info(f"-> reconnecting (sim_change={self._sim_change} port={os.path.exists(AT_PORT)})")
      return State.RECONNECTING
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
        logging.warning(f"pppd reader error: {e}")

    threading.Thread(target=_read_pppd, args=(self._ppp, self._ppp_lines), daemon=True).start()
    logging.info("PPP dialing")

  def _do_connecting(self):
    logging.info("starting pppd")
    self._ppp_fails = 0
    self._sim_change = False
    self._start_pppd()
    return State.CONNECTED

  def _install_ppp_routes(self, peer: str):
    self._cleanup_routes()
    subprocess.run(["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "metric", "1000"],
                   capture_output=True)
    subprocess.run(["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "table", "1000"],
                   capture_output=True)
    subprocess.run(["sudo", "ip", "rule", "add", "from", self.S["ip_address"], "table", "1000"],
                   capture_output=True)
    logging.info(f"route set up for {self.S['ip_address']} via {peer}")

  def _handle_pppd_line(self, line: str):
    logging.info(f"pppd: {line}")
    if "local  IP address" in line:
      ip = line.split("local  IP address")[-1].strip()
      self._update(ip_address=ip, connected=True)
    elif "remote IP address" in line and self.S["ip_address"]:
      # only install routes once both local IP and remote peer are known
      peer = line.split("remote IP address")[-1].strip()
      self._install_ppp_routes(peer)
    elif "Connection terminated" in line or "Modem hangup" in line:
      self._update(connected=False, ip_address="")

  def _drain_pppd_output(self):
    while not self._ppp_lines.empty():
      try:
        raw = self._ppp_lines.get_nowait()
      except queue.Empty:
        break
      line = raw.decode(errors="ignore").strip()
      if line:
        self._handle_pppd_line(line)

  def _handle_pppd_exit(self):
    """Called when pppd has exited. Returns next State."""
    self._ppp = None
    if self._sim_change or not os.path.exists(AT_PORT):
      return State.RECONNECTING
    self._ppp_fails += 1
    if self._ppp_fails >= 3:
      logging.warning(f"PPP fail {self._ppp_fails}/3, reconnecting")
      return State.RECONNECTING
    logging.warning(f"PPP fail {self._ppp_fails}/3, retrying")
    self._reset_data_port()
    if not os.path.exists(AT_PORT):
      return State.RECONNECTING
    self._start_pppd()
    return State.CONNECTED

  def _params_changed(self) -> bool:
    new_apn = self._read_param("GsmApn")
    if new_apn != self._apn:
      logging.info(f"APN changed: '{self._apn}' -> '{new_apn}'")
      return True
    new_roaming = self._read_param("GsmRoaming") != "0"
    if new_roaming != self._roaming_allowed:
      logging.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      return True
    return False

  def _check_iccid(self):
    """Detect SIM swaps. Skip if port missing or no prior identity."""
    if not os.path.exists(AT_PORT) or not self.S["iccid"]:
      return
    iccid = self._atv("AT+QCCID", "+QCCID:")
    if iccid and iccid != self.S["iccid"]:
      logging.warning(f"iccid changed: {self.S['iccid']} -> {iccid}")
      self._sim_change = True

  def _do_connected(self):
    self._drain_pppd_output()

    if self._ppp and self._ppp.poll() is not None:
      return self._handle_pppd_exit()

    if self._sim_change or not os.path.exists(AT_PORT) or self._params_changed():
      return State.RECONNECTING

    self._poll()
    return State.CONNECTED

  def _do_reconnecting(self):
    logging.warning("reconnecting")
    self._tx_base = self._rx_base = self._last_tx = self._last_rx = 0
    self._update(**INITIAL_STATE)
    self._kill_ppp()
    self._cleanup_routes()
    self._reset_data_port()
    self._sim_change = False
    return State.WAITING_PORT

  # -- poll --

  def _poll_signal(self) -> dict:
    v = self._atv("AT+CSQ", "+CSQ:")
    if not v:
      return {}
    try:
      rssi = int(v.split(",")[0])
      if rssi == 99:
        return {}
      return {"signal_strength": rssi, "signal_quality": min(100, int(rssi / 31 * 100))}
    except (ValueError, IndexError):
      return {}

  @staticmethod
  def _act_to_network_type(act: int) -> str:
    # 3GPP TS 27.007 AcT: 0-1 GSM, 2 UTRAN, 3 GSM+EGPRS, 4-6 UTRAN HS*, 7/9/10 E-UTRAN, 11-13 NR
    if act in (0, 1, 3):
      return "gsm"
    if act in (2, 4, 5, 6):
      return "utran"
    if act in (7, 9, 10):
      return "lte"
    if act in (11, 12, 13):
      return "nr"
    return "unknown"

  def _poll_operator(self) -> dict:
    v = self._atv("AT+COPS?", "+COPS:")
    if not v:
      return {}
    p = v.split(",")
    out: dict = {}
    try:
      if len(p) >= 3:
        out["operator"] = p[2].strip('"')
      if len(p) >= 4:
        out["network_type"] = self._act_to_network_type(int(p[3]))
    except (ValueError, IndexError):
      pass
    return out

  def _poll_band(self) -> dict:
    v = self._atv("AT+QNWINFO", "+QNWINFO:")
    if not v:
      return {}
    info = v.replace('"', '').split(",")
    try:
      if len(info) >= 4:
        return {"band": info[2], "channel": int(info[3])}
    except (ValueError, IndexError):
      pass
    return {}

  def _poll_extra(self) -> dict:
    v = self._atv('AT+QENG="servingcell"', "+QENG:")
    return {"extra": v.replace('"', '')} if v else {}

  def _poll_temps(self) -> dict:
    v = self._atv("AT+QTEMP", "+QTEMP:")
    if not v:
      return {}
    try:
      return {"temperatures": [t for t in (int(x) for x in v.split(",") if x.strip()) if t != 255]}
    except (ValueError, IndexError):
      return {}

  def _poll_iface(self) -> dict:
    try:
      r = subprocess.run(["ip", "-4", "addr", "show", "ppp0"], capture_output=True, text=True, timeout=2)
      ip = next((l.strip().split()[1].split("/")[0] for l in r.stdout.splitlines() if "inet " in l), None)
      if ip:
        return {"ip_address": ip, "connected": True}
      if self.S["connected"]:
        return {"connected": False, "ip_address": ""}
    except Exception:
      pass
    return {}

  def _poll_byte_counters(self) -> dict:
    try:
      with open("/sys/class/net/ppp0/statistics/tx_bytes") as f:
        tx = int(f.read().strip())
      with open("/sys/class/net/ppp0/statistics/rx_bytes") as f:
        rx = int(f.read().strip())
    except Exception:
      return {}
    # ppp0 sysfs counters reset each time pppd recreates the interface;
    # carry a base across restarts so cumulative stays monotonic within a session
    self._tx_base += tx if tx < self._last_tx else tx - self._last_tx
    self._rx_base += rx if rx < self._last_rx else rx - self._last_rx
    self._last_tx, self._last_rx = tx, rx
    return {"tx_bytes": self._tx_base, "rx_bytes": self._rx_base}

  def _poll(self):
    s: dict = {}
    for fn in (self._poll_signal, self._poll_operator, self._poll_band,
               self._poll_extra, self._poll_temps, self._poll_iface,
               self._poll_byte_counters):
      s.update(fn())
    if s:
      self._update(**s)

  # -- main loop --

  def run(self):
    logging.info("starting")
    # publish initial state so callers see modem.py is active from the start
    self._update(state=State.INIT.label)
    # mask before stop so anything trying to activate ModemManager (NetworkManager, dbus) can't race us
    subprocess.run(["sudo", "systemctl", "mask", "--runtime", "ModemManager"], capture_output=True)
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
        if state not in (State.INIT, State.RECONNECTING):
          self._check_iccid()
        prev = state
        state = handlers[state]()
        if state != prev:
          self._update(state=state.label)
          logging.info(f"{prev.value} -> {state.value}")
      except Exception:
        logging.exception(f"error in {state.value}")
        state = State.RECONNECTING
      time.sleep(STATE_WAIT.get(state, STATE_WAIT_DEFAULT))

  def stop(self):
    self.running = False
    self._kill_ppp()
    self._cleanup_routes()
    try:
      os.remove(STATE_PATH)
    except FileNotFoundError:
      pass
    subprocess.run(["sudo", "systemctl", "unmask", "--runtime", "ModemManager"], capture_output=True)
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
