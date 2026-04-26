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
# 3GPP TS 27.007 AcT codes -> network type
NETWORK_TYPE = {0: "gsm", 1: "gsm", 3: "gsm",
                2: "utran", 4: "utran", 5: "utran", 6: "utran",
                7: "lte", 9: "lte", 10: "lte",
                11: "nr", 12: "nr", 13: "nr"}

# 3GPP CEER reason substring -> user-facing message
CEER_MESSAGES = {
  "PLMN_NOT_ALLOWED": "Carrier rejected SIM. The APN may be wrong.",
  "EPS_SERVICES_NOT_ALLOWED": "Cellular data not allowed on this SIM.",
  "GPRS_SERVICES_NOT_ALLOWED": "Cellular data not allowed on this SIM.",
  "OPERATOR_DETERMINED_BARRING": "Carrier has blocked this SIM.",
  "IMSI_UNKNOWN": "SIM not recognized by carrier. The eSIM profile may not be active.",
  "ILLEGAL_ME": "Device blocked by carrier.",
  "IMEI_NOT_ACCEPTED": "Device blocked by carrier.",
  "ROAMING_NOT_ALLOWED": "Roaming not allowed on this SIM.",
}
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
  INITIALIZING = "INITIALIZING"
  SEARCHING = "SEARCHING"
  CONNECTING = "CONNECTING"
  CONNECTED = "CONNECTED"
  DISCONNECTING = "DISCONNECTING"


# seconds to wait after each state handler returns; states omitted use STATE_WAIT_DEFAULT
STATE_WAIT_DEFAULT = 2.0
STATE_WAIT = {
  State.INITIALIZING: 1.0,
  State.SEARCHING: 0.5,
}


class Modem:
  def __init__(self):
    self._ppp = None
    self._ppp_lines = queue.Queue()
    self._ppp_fails = 0
    self._sim_change = False
    self._apn = ""
    self._roaming_allowed = True
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
    self.S.update(kwargs)
    with tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False) as f:
      json.dump(self.S, f)
    os.chmod(f.name, 0o644)
    os.replace(f.name, STATE_PATH)

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
    except (RuntimeError, TimeoutError, OSError) as e:
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
    code = parts[-1] if parts else ceer
    msg = next((m for k, m in CEER_MESSAGES.items() if k in code), f"Carrier rejected connection ({code}).")
    return {"type": "carrier_reject", "description": msg}

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
    # rules don't have a flush; delete until none remain
    while subprocess.run(["sudo", "ip", "rule", "del", "table", "1000"], capture_output=True).returncode == 0:
      pass

  @staticmethod
  def _reset_data_port():
    """Drop DTR on PPP_PORT so the modem terminates any stuck PPP session."""
    try:
      with serial.Serial(PPP_PORT, 460800, timeout=1) as s:
        s.dtr = False
        time.sleep(0.2)
        s.dtr = True
    except Exception as e:
      logging.warning(f"data port reset failed: {e}")

  # -- state handlers --

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

  def _do_initializing(self):
    if not os.path.exists(AT_PORT):
      return State.INITIALIZING
    logging.info("port found, initializing")
    # kill any stale pppd from previous run
    self._kill_ppp()
    self._cleanup_routes()

    # AT init
    for c in AT_INIT + ["AT+CREG=2", "AT+CGREG=2"]:
      self._at(c)

    identity = self._read_identity()
    self._configure_modem(identity["modem_version"], identity["iccid"])

    self._apn = self._read_param("GsmApn")
    self._roaming_allowed = self._read_param("GsmRoaming") != "0"
    self._at(f'AT+CGDCONT=1,"IP","{self._apn}"')
    logging.info(f"APN '{self._apn or '(auto)'}' CID 1, roaming={'on' if self._roaming_allowed else 'off'}")

    self._sim_change = False  # clear — we just re-read identity with the new SIM
    self._update(**identity)
    return State.SEARCHING

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

  def _do_searching(self):
    new_roaming = self._read_param("GsmRoaming") != "0"
    if new_roaming != self._roaming_allowed:
      logging.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      self._roaming_allowed = new_roaming

    v = self._atv("AT+CREG?", "+CREG:")
    if not v:
      return self._searching_idle()

    reg = self._parse_reg(v)
    greg = self._parse_reg(self._atv("AT+CGREG?", "+CGREG:") or "")
    logging.debug(f"creg={reg} cgreg={greg} roaming_allowed={self._roaming_allowed}")

    if reg == "roaming" and not self._roaming_allowed:
      self._update(registration=reg, error={"type": "roaming_disabled", "description": "Roaming is disabled."})
      return State.SEARCHING

    if reg in ("home", "roaming") and greg in ("home", "roaming"):
      self._update(registration=reg, error={})
      return State.CONNECTING

    # not connectable yet — record current reg and any carrier reject reason
    if reg != self.S.get("registration"):
      self._update(registration=reg, error=self._carrier_reject_error(reg))

    return self._searching_idle()

  def _searching_idle(self):
    if self._sim_change or not os.path.exists(AT_PORT):
      logging.info(f"-> reconnecting (sim_change={self._sim_change} port={os.path.exists(AT_PORT)})")
      return State.DISCONNECTING
    return State.SEARCHING

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

  def _handle_pppd_exit(self):
    self._ppp = None
    if self._sim_change or not os.path.exists(AT_PORT):
      return State.DISCONNECTING
    self._ppp_fails += 1
    if self._ppp_fails >= 3:
      logging.warning(f"PPP fail {self._ppp_fails}/3, reconnecting")
      return State.DISCONNECTING
    logging.warning(f"PPP fail {self._ppp_fails}/3, retrying")
    self._reset_data_port()
    if not os.path.exists(AT_PORT):
      return State.DISCONNECTING
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

  def _check_iccid(self, state):
    # skip when port may be gone or identity not yet known
    if state in (State.INITIALIZING, State.DISCONNECTING) or not self.S["iccid"]:
      return
    iccid = self._atv("AT+QCCID", "+QCCID:")
    if iccid and iccid != self.S["iccid"]:
      logging.warning(f"iccid changed: {self.S['iccid']} -> {iccid}")
      self._sim_change = True

  def _do_connected(self):
    while True:
      try:
        line = self._ppp_lines.get_nowait().decode(errors="ignore").strip()
      except queue.Empty:
        break
      if line:
        self._handle_pppd_line(line)

    if self._ppp and self._ppp.poll() is not None:
      return self._handle_pppd_exit()

    if self._sim_change or not os.path.exists(AT_PORT) or self._params_changed():
      return State.DISCONNECTING

    self._poll()
    return State.CONNECTED

  def _do_disconnecting(self):
    logging.warning("reconnecting")
    self._update(**INITIAL_STATE)
    self._kill_ppp()
    self._cleanup_routes()
    self._reset_data_port()
    self._sim_change = False
    return State.INITIALIZING

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
        out["network_type"] = NETWORK_TYPE.get(int(p[3]), "unknown")
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
    return {"tx_bytes": tx, "rx_bytes": rx}

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
    self._update(state=State.INITIALIZING.value)
    # mask before stop so anything trying to activate ModemManager (NetworkManager, dbus) can't race us
    subprocess.run(["sudo", "systemctl", "mask", "--runtime", "ModemManager"], capture_output=True)
    subprocess.run(["sudo", "systemctl", "stop", "ModemManager"], capture_output=True)
    subprocess.run(["sudo", "killall", "pppd"], capture_output=True)

    state = State.INITIALIZING

    handlers = {
      State.INITIALIZING: self._do_initializing,
      State.SEARCHING: self._do_searching,
      State.CONNECTING: self._do_connecting,
      State.CONNECTED: self._do_connected,
      State.DISCONNECTING: self._do_disconnecting,
    }

    while self.running:
      try:
        self._check_iccid(state)
        prev = state
        state = handlers[state]()
        if state != prev:
          self._update(state=state.value)
          logging.info(f"{prev.value} -> {state.value}")
      except Exception:
        logging.exception(f"error in {state.value}")
        state = State.DISCONNECTING
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
