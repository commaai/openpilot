#!/usr/bin/env python3
import fcntl
import json
import logging
import os
import serial
import signal
import subprocess
import tempfile
import time

from ipaddress import IPv4Address, AddressValueError

from enum import Enum

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s.%(msecs)03d %(levelname)-7s modem: %(message)s",
  datefmt="%H:%M:%S",
)

AT_PORT = "/dev/modem_at0"
PPP_PORT = "/dev/modem_at1"
STATE_PATH = "/dev/shm/modem"
AT_LOCK = "/dev/shm/modem.lock"  # shared with LPA
AT_INIT = [
  "ATE0",       # disable command echo
  "ATV1",       # verbose result codes (CONNECT/BUSY/NO CARRIER, not numeric)
  "AT+CMEE=1",  # numeric +CME ERROR codes on failures (per 3GPP 27.007)
  "ATX4",       # extended result codes: busy + dial tone detection, line speed in CONNECT
  "AT&C1",      # DCD pin follows carrier state (V.250 default)
  "AT+CREG=2",  # registration URCs include location info
  "AT+CGREG=2", # GPRS registration URCs include location info
]
CREG = {0: "not_registered", 1: "home", 2: "searching", 3: "denied", 4: "unknown", 5: "roaming"}
# 3GPP TS 27.007 +COPS <AcT> -> network type
NETWORK_TYPE = {0: "gsm", 1: "gsm", 3: "gsm", 8: "gsm",
                2: "utran", 4: "utran", 5: "utran", 6: "utran",
                7: "lte", 9: "lte", 10: "lte",
                11: "nr", 12: "nr", 13: "nr"}

DIAL_CID = 1

PPPD_CMD = [
  "sudo", "pppd", PPP_PORT, "460800", "noauth", "nodetach", "noipdefault", "usepeerdns",
  "nodefaultroute", "connect",
  "/usr/sbin/chat -v ABORT 'NO CARRIER' ABORT 'NO DIALTONE' ABORT 'BUSY' " +
  f"ABORT 'NO ANSWER' ABORT 'ERROR' TIMEOUT 5 '' AT OK ATD*99***{DIAL_CID}# CONNECT ''",
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
  "tx_bytes": 0, "rx_bytes": 0,
}


class State(Enum):
  INITIALIZING = "INITIALIZING"
  SEARCHING = "SEARCHING"
  CONNECTING = "CONNECTING"
  CONNECTED = "CONNECTED"
  DISCONNECTING = "DISCONNECTING"


STATE_WAIT = 1.0  # seconds to wait after each state handler returns


class PPPSession:
  """Owns pppd lifecycle, fail tracking, and PPP routing."""
  MAX_FAILS = 3

  def __init__(self):
    self._proc: subprocess.Popen | None = None
    self._fails = 0
    self._peer = ""

  def start(self):
    self._proc = subprocess.Popen(PPPD_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self._peer = ""
    logging.info(f"PPP dialing CID {DIAL_CID}")

  def kill(self):
    subprocess.run(["sudo", "killall", "-9", "pppd"], capture_output=True)
    self._peer = ""

  @staticmethod
  def reset_data_port():
    """Drop DTR on PPP_PORT so the modem terminates any stuck PPP session."""
    try:
      with serial.Serial(PPP_PORT, 460800, timeout=1) as s:
        s.dtr = False
        time.sleep(0.2)
        s.dtr = True
    except Exception as e:
      logging.warning(f"data port reset failed: {e}")

  def has_exited(self) -> bool:
    return self._proc is not None and self._proc.poll() is not None

  def reset_fail_counter(self):
    self._fails = 0

  def record_fail(self) -> bool:
    """Bump fail counter; return True if at the give-up limit."""
    self._fails += 1
    return self._fails >= self.MAX_FAILS

  @property
  def fails(self) -> int:
    return self._fails

  def maybe_install_routes(self, ip: str, peer: str):
    """Install routes if peer changed; kill the session on failure so the state machine reconnects."""
    if not peer or peer == self._peer:
      return
    try:
      IPv4Address(ip)
      IPv4Address(peer)
    except AddressValueError:
      logging.warning(f"refusing route install with non-IPv4 ip={ip!r} peer={peer!r}")
      self.kill()
      return
    self.cleanup_routes()
    cmds = [
      ["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "metric", "1000"],
      ["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "table", "1000"],
      ["sudo", "ip", "rule", "add", "from", ip, "table", "1000"],
    ]
    for cmd in cmds:
      r = subprocess.run(cmd, capture_output=True, text=True)
      if r.returncode != 0:
        logging.warning(f"route install failed ({' '.join(cmd[1:])}): {r.stderr.strip()}")
        self.cleanup_routes()
        self.kill()
        return
    logging.info(f"route set up for {ip} via {peer}")
    self._peer = peer

  @staticmethod
  def cleanup_routes():
    subprocess.run(["sudo", "ip", "route", "del", "default", "dev", "ppp0"], capture_output=True)
    subprocess.run(["sudo", "ip", "route", "flush", "table", "1000"], capture_output=True)
    # rules don't have a flush; delete until none remain
    while subprocess.run(["sudo", "ip", "rule", "del", "table", "1000"], capture_output=True).returncode == 0:
      pass


class Modem:
  def __init__(self):
    self._ppp = PPPSession()
    self._sim_change = False
    self._apn = ""  # blank = network-provided via PCO
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

  @staticmethod
  def _parse_reg(v: str) -> str:
    try:
      return CREG.get(int(v.split(",")[1].strip('"')), "unknown")
    except (ValueError, IndexError):
      return "unknown"

  @staticmethod
  def _has_modem_manager() -> bool:
    return os.path.isfile("/lib/systemd/system/ModemManager.service")

  def _is_roaming_allowed(self) -> bool:
    return self._read_param("GsmRoaming") == "1"

  def _publish_state(self, **kwargs):
    self.S.update(kwargs)
    with tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False) as f:
      json.dump(self.S, f)
    os.chmod(f.name, 0o644)
    os.replace(f.name, STATE_PATH)

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

  def _init_at_channel(self) -> bool:
    """Run AT_INIT and confirm ATE0 took effect. Returns False if echo is still on."""
    for c in AT_INIT:
      self._at(c)
    r = self._at("AT+CGMI")
    return bool(r) and not r[0].startswith("AT")

  def _configure_modem(self, modem_version: str):
    if not modem_version.startswith("EG25"):
      return
    cmds = [
      # clear initial EPS bearer APN (some carriers reject the default)
      'AT+CGDCONT=0,"IP",""',

      # SIM hot swap
      'AT+QSIMDET=1,0',
      'AT+QSIMSTAT=1',

      # configure modem as data-centric
      'AT+QNVW=5280,0,"0102000000000000"',
      'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
      'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
    ]
    for c in cmds:
      self._at(c)

  def _do_initializing(self):
    if not os.path.exists(AT_PORT):
      return State.INITIALIZING
    logging.info("port found, initializing")
    self._ppp.kill()
    self._ppp.cleanup_routes()

    if not self._init_at_channel():
      logging.warning("AT echo still on, retrying")
      return State.INITIALIZING

    identity = self._read_identity()
    if not identity["iccid"] or not identity["imei"]:
      logging.warning(f"identity read incomplete: {identity}, retrying")
      return State.INITIALIZING

    self._configure_modem(identity["modem_version"])

    self._apn = self._read_param("GsmApn")
    self._roaming_allowed = self._is_roaming_allowed()
    # blank APN lets the carrier supply one via PCO
    self._at(f'AT+CGDCONT={DIAL_CID},"IP","{self._apn}"')
    logging.info(f"APN '{self._apn or '(network-provided)'}' written to CID {DIAL_CID}, roaming={'on' if self._roaming_allowed else 'off'}")

    self._sim_change = False  # clear since we just re-read identity with the new SIM
    self._publish_state(**identity)
    return State.SEARCHING

  def _read_identity(self):
    def first_line(cmd):
      r = self._at(cmd)
      return r[0].strip() if r else ""

    imei = first_line("AT+CGSN")
    if not (imei.isdigit() and 14 <= len(imei) <= 17):  # 3GPP TS 23.003
      imei = ""

    iccid = self._atv("AT+QCCID", "+QCCID:") or ""
    if not iccid.isdigit():
      iccid = ""

    imsi = first_line("AT+CIMI")
    mcc_mnc = imsi[:6] if imsi.isdigit() and len(imsi) >= 6 else ""

    modem_version = first_line("AT+GMR")

    logging.info(f"imei={imei} iccid={iccid} mcc_mnc={mcc_mnc} ver={modem_version}")
    return {"imei": imei, "iccid": iccid, "mcc_mnc": mcc_mnc, "modem_version": modem_version}

  def _do_searching(self):
    new_roaming = self._is_roaming_allowed()
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
      self._publish_state(registration=reg)
      return State.SEARCHING

    if reg in ("home", "roaming") and greg in ("home", "roaming"):
      self._publish_state(registration=reg)
      return State.CONNECTING

    if reg != self.S.get("registration"):
      self._publish_state(registration=reg)
    return self._searching_idle()

  def _searching_idle(self):
    if self._sim_change or not os.path.exists(AT_PORT):
      logging.info(f"-> reconnecting (sim_change={self._sim_change} port={os.path.exists(AT_PORT)})")
      return State.DISCONNECTING
    return State.SEARCHING

  def _do_connecting(self):
    logging.info("starting pppd")
    self._ppp.reset_fail_counter()
    self._sim_change = False
    self._ppp.start()
    return State.CONNECTED

  def _handle_pppd_exit(self):
    if self._sim_change or not os.path.exists(AT_PORT):
      return State.DISCONNECTING
    give_up = self._ppp.record_fail()
    if give_up:
      logging.warning(f"PPP fail {self._ppp.fails}/{self._ppp.MAX_FAILS}, reconnecting")
      return State.DISCONNECTING
    logging.warning(f"PPP fail {self._ppp.fails}/{self._ppp.MAX_FAILS}, retrying")
    self._ppp.reset_data_port()
    if not os.path.exists(AT_PORT):
      return State.DISCONNECTING
    self._ppp.start()
    return State.CONNECTED

  def _params_changed(self) -> bool:
    new_apn = self._read_param("GsmApn")
    if new_apn != self._apn:
      logging.info(f"GsmApn changed: '{self._apn}' -> '{new_apn}'")
      return True
    new_roaming = self._is_roaming_allowed()
    if new_roaming != self._roaming_allowed:
      logging.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      return True
    return False

  def _check_iccid(self, state):
    if state in (State.INITIALIZING, State.DISCONNECTING) or not self.S["iccid"]:
      return
    iccid = self._atv("AT+QCCID", "+QCCID:")
    if iccid and iccid != self.S["iccid"]:
      logging.warning(f"iccid changed: {self.S['iccid']} -> {iccid}")
      self._sim_change = True

  def _do_connected(self):
    if self._ppp.has_exited():
      return self._handle_pppd_exit()

    if self._sim_change or not os.path.exists(AT_PORT) or self._params_changed():
      return State.DISCONNECTING

    self._poll()
    return State.CONNECTED

  def _do_disconnecting(self):
    logging.warning("reconnecting")
    self._publish_state(**INITIAL_STATE)
    self._ppp.kill()
    self._ppp.cleanup_routes()
    self._ppp.reset_data_port()
    self._sim_change = False
    return State.INITIALIZING

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
    except ValueError:
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
      ip, peer = "", ""
      for line in r.stdout.splitlines():
        # `inet 10.x.x.x peer 10.64.64.64/32 ...`
        parts = line.strip().split()
        if "inet" in parts:
          i = parts.index("inet")
          ip = parts[i + 1].split("/")[0]
          if "peer" in parts:
            peer = parts[parts.index("peer") + 1].split("/")[0]
          break
      if ip:
        self._ppp.maybe_install_routes(ip, peer)
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
      self._publish_state(**s)

  def run(self):
    logging.info("starting")
    self._publish_state(state=State.INITIALIZING.value)
    if self._has_modem_manager():
      subprocess.run(["sudo", "systemctl", "mask", "--runtime", "ModemManager"], capture_output=True)
      subprocess.run(["sudo", "systemctl", "stop", "ModemManager"], capture_output=True)
    self._ppp.kill()

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
          self._publish_state(state=state.value)
          logging.info(f"{prev.value} -> {state.value}")
      except Exception:
        logging.exception(f"error in {state.value}")
        state = State.DISCONNECTING
      time.sleep(STATE_WAIT)

  def stop(self):
    self.running = False
    self._ppp.kill()
    self._ppp.cleanup_routes()
    try:
      os.remove(STATE_PATH)
    except FileNotFoundError:
      pass
    if self._has_modem_manager():
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
