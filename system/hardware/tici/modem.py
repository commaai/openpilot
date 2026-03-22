"""
modem.py - Replaces ModemManager with direct AT command control.

Supports:
  - comma 3X: Quectel EG25  (device type "tizi")
  - comma four: Quectel EG916 (all other tici devices)

State is published to MODEM_STATE_FILE as a JSON blob so that other
processes (e.g. hardware.py) can read modem information without going
through D-Bus or ModemManager.
"""

import json
import os
import re
import serial
import subprocess
import time
from pathlib import Path

MODEM_AT_PORT = "/dev/modem_at0"
MODEM_STATE_FILE = "/dev/shm/modem_state.txt"
LOOP_INTERVAL = 5.0  # seconds between state refresh cycles

# Default/empty state written on startup
EMPTY_STATE: dict = {
  "state": "UNKNOWN",
  "access_technology": 0,
  "signal_quality": 0,
  "imei": "",
  "revision": "",
  "sim_id": "",
  "mcc_mnc": "",
  "temperatures": [],
  "network_info": None,
  "data_tx": -1,
  "data_rx": -1,
}

# ---------------------------------------------------------------------------
# Low-level AT helpers
# ---------------------------------------------------------------------------

def at_cmd(port: serial.Serial, cmd: str, timeout: float = 1.0) -> str:
  """Send a single AT command and return the response string."""
  port.reset_input_buffer()
  port.write((cmd + "\r\n").encode())
  port.flush()

  end = time.monotonic() + timeout
  response = b""
  while time.monotonic() < end:
    chunk = port.read(port.in_waiting or 1)
    response += chunk
    decoded = response.decode(errors="replace")
    if "OK" in decoded or "ERROR" in decoded:
      break
    time.sleep(0.01)

  return response.decode(errors="replace").strip()


def open_at_port(path: str = MODEM_AT_PORT, retries: int = 10) -> serial.Serial:
  """Open the modem AT serial port, retrying until it becomes available."""
  for _ in range(retries):
    try:
      port = serial.Serial(path, baudrate=115200, timeout=0.2)
      # Probe with a simple AT to confirm it works
      resp = at_cmd(port, "AT", timeout=1.0)
      if "OK" in resp:
        return port
      port.close()
    except Exception:
      pass
    time.sleep(2)
  raise RuntimeError(f"Could not open AT port {path} after {retries} retries")


# ---------------------------------------------------------------------------
# Modem detection
# ---------------------------------------------------------------------------

def get_modem_type(port: serial.Serial) -> str:
  """Return 'EG25' or 'EG916' based on ATI response."""
  resp = at_cmd(port, "ATI", timeout=2.0)
  if "EG25" in resp:
    return "EG25"
  return "EG916"


# ---------------------------------------------------------------------------
# Modem configuration
# ---------------------------------------------------------------------------

# EG25 (comma 3X / tizi) configuration commands
EG25_CMDS = [
  # SIM hot swap
  'AT+QSIMDET=1,0',
  'AT+QSIMSTAT=1',

  # configure modem as data-centric
  'AT+QNVW=5280,0,"0102000000000000"',
  'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
  'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
]

# EG916 (comma four) commands when no SIM is present
EG916_CMDS_NO_SIM = [
  'AT$QCSIMSLEEP=0',
  'AT$QCSIMCFG=SimPowerSave,0',
  'AT$QCPCFG=usbNet,1',
]


def configure_modem(port: serial.Serial, modem_type: str, sim_id: str) -> None:
  """Send initial configuration AT commands to the modem."""
  if modem_type == "EG25":
    # Clear old blue prime initial APN
    subprocess.run(
      ['mmcli', '-m', 'any', '--3gpp-set-initial-eps-bearer-settings=apn='],
      timeout=5, capture_output=True
    ) if shutil_which("mmcli") else None

    for cmd in EG25_CMDS:
      try:
        at_cmd(port, cmd, timeout=1.0)
      except Exception:
        pass
  else:
    # EG916 — only configure when SIM is absent (modem gets upset otherwise)
    if not sim_id:
      for cmd in EG916_CMDS_NO_SIM:
        try:
          at_cmd(port, cmd, timeout=1.0)
        except Exception:
          pass


def shutil_which(name: str) -> bool:
  import shutil
  return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------

def query_imei(port: serial.Serial) -> str:
  resp = at_cmd(port, "AT+CGSN", timeout=1.0)
  for line in resp.splitlines():
    line = line.strip()
    if re.fullmatch(r'\d{15}', line):
      return line
  return ""


def query_revision(port: serial.Serial) -> str:
  resp = at_cmd(port, "AT+CGMR", timeout=1.0)
  for line in resp.splitlines():
    line = line.strip()
    if line and line not in ("AT+CGMR", "OK", "ERROR"):
      return line
  return ""


def query_sim_id(port: serial.Serial) -> str:
  """Return ICCID (SIM identifier)."""
  resp = at_cmd(port, "AT+CCID", timeout=1.0)
  for line in resp.splitlines():
    line = line.strip()
    # ICCID is 19-20 digits
    m = re.search(r'\d{19,20}', line)
    if m:
      return m.group(0)
  return ""


def query_mcc_mnc(port: serial.Serial) -> str:
  """Return current operator MCC+MNC as a string."""
  resp = at_cmd(port, "AT+COPS?", timeout=2.0)
  # +COPS: 0,0,"Operator",7  or  +COPS: 0,2,"31026",7
  m = re.search(r'\+COPS:\s*\d+,\d+,"([^"]+)"', resp)
  if m:
    val = m.group(1)
    # If it looks like a numeric MCC+MNC, return it; otherwise return raw
    return val
  return ""


def query_signal_quality(port: serial.Serial) -> int:
  """Return signal quality as 0-100 percentage."""
  resp = at_cmd(port, "AT+CSQ", timeout=1.0)
  m = re.search(r'\+CSQ:\s*(\d+),', resp)
  if m:
    rssi = int(m.group(1))
    if rssi == 99:
      return 0
    # Map 0-31 to 0-100
    return min(100, int(rssi / 31 * 100))
  return 0


def query_access_technology(port: serial.Serial) -> int:
  """Return access technology as a bitmask matching MM_MODEM_ACCESS_TECHNOLOGY."""
  # MM constants: UMTS = 1<<5 = 32, LTE = 1<<14 = 16384
  resp = at_cmd(port, "AT+COPS?", timeout=2.0)
  # Last field in +COPS response is AcT:
  #  0=GSM, 2=UTRAN, 7=E-UTRAN(LTE), 11=NR5G
  m = re.search(r'\+COPS:\s*\d+,\d+,"[^"]*",(\d+)', resp)
  if m:
    act = int(m.group(1))
    if act in (7, 11):  # LTE or 5G
      return 1 << 14
    elif act in (2, 4):  # UTRAN / HSDPA
      return 1 << 5
    elif act == 0:       # GSM
      return 1 << 0
  return 0


def query_modem_state(port: serial.Serial) -> str:
  """Return a simplified modem state string."""
  resp = at_cmd(port, "AT+CREG?", timeout=1.0)
  m = re.search(r'\+CREG:\s*\d+,(\d+)', resp)
  if m:
    stat = int(m.group(1))
    if stat in (1, 5):
      # Check if data is connected
      dresp = at_cmd(port, "AT+CGACT?", timeout=1.0)
      if re.search(r'\+CGACT:\s*1,1', dresp):
        return "CONNECTED"
      return "REGISTERED"
    elif stat == 2:
      return "SEARCHING"
    elif stat == 0:
      return "DISABLED"
  return "UNKNOWN"


def query_temperatures(port: serial.Serial) -> list:
  resp = at_cmd(port, "AT+QTEMP", timeout=0.5)
  # +QTEMP: 0,"PA-MD",45,46,47  (format varies by modem)
  m = re.search(r'\+QTEMP:\s*\S+\s+([\d,]+)', resp)
  if m:
    try:
      temps = [int(t) for t in m.group(1).split(',')]
      return [t for t in temps if t != 255]
    except ValueError:
      pass
  return []


def query_network_info(port: serial.Serial) -> dict | None:
  resp_nw = at_cmd(port, "AT+QNWINFO", timeout=1.0)
  resp_sc = at_cmd(port, 'AT+QENG="servingcell"', timeout=1.0)

  if "+QNWINFO:" not in resp_nw:
    return None

  info_line = ""
  for line in resp_nw.splitlines():
    if "+QNWINFO:" in line:
      info_line = line.replace("+QNWINFO:", "").replace('"', '').strip()
      break

  parts = info_line.split(',')
  if len(parts) != 4:
    return None

  technology, operator, band, channel = parts
  extra = ""
  for line in resp_sc.splitlines():
    if "+QENG:" in line:
      extra = line.replace('+QENG: "servingcell",', '').replace('"', '').strip()
      break

  state = query_modem_state(port)

  return {
    "technology": technology.strip(),
    "operator": operator.strip(),
    "band": band.strip(),
    "channel": int(channel.strip()) if channel.strip().isdigit() else 0,
    "extra": extra,
    "state": state,
  }


def query_data_usage() -> tuple[int, int]:
  """Read TX/RX byte counters from sysfs (avoids NetworkManager D-Bus)."""
  try:
    tx = int(Path("/sys/class/net/wwan0/statistics/tx_bytes").read_text().strip())
    rx = int(Path("/sys/class/net/wwan0/statistics/rx_bytes").read_text().strip())
    return tx, rx
  except Exception:
    return -1, -1


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

def write_state(state: dict) -> None:
  """Atomically write the state dict to MODEM_STATE_FILE."""
  tmp = MODEM_STATE_FILE + ".tmp"
  try:
    with open(tmp, "w") as f:
      json.dump(state, f)
    os.replace(tmp, MODEM_STATE_FILE)
  except Exception as e:
    print(f"modem.py: failed to write state: {e}")


def read_state() -> dict:
  """Read the state dict from MODEM_STATE_FILE, returning EMPTY_STATE on error."""
  try:
    with open(MODEM_STATE_FILE) as f:
      data = json.load(f)
    # Ensure all keys present
    return {**EMPTY_STATE, **data}
  except Exception:
    return dict(EMPTY_STATE)


# ---------------------------------------------------------------------------
# eSIM / NM connection setup (replaces configure_modem eSIM part)
# ---------------------------------------------------------------------------

def setup_esim_connection(sim_id: str) -> None:
  """Create NetworkManager connection file for eSIM prime if needed."""
  from openpilot.system.hardware.tici.lpa import TiciLPA  # type: ignore
  dest = "/etc/NetworkManager/system-connections/esim.nmconnection"
  if TiciLPA().is_comma_profile(sim_id) and not os.path.exists(dest):
    import tempfile
    tmpl = Path(__file__).parent / "esim.nmconnection"
    with open(tmpl) as f, tempfile.NamedTemporaryFile(mode="w", suffix=".nmconnection") as tf:
      dat = f.read().replace("sim-id=", f"sim-id={sim_id}")
      tf.write(dat)
      tf.flush()
      os.system(f"sudo cp {tf.name} {dest}")
    os.system(f"sudo nmcli con load {dest}")


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------

def run() -> None:
  """Main modem management loop. Run as a process/service."""
  # Write empty state immediately so readers don't block
  write_state(dict(EMPTY_STATE))

  print("modem.py: waiting for AT port...")
  try:
    port = open_at_port(MODEM_AT_PORT)
  except RuntimeError as e:
    print(f"modem.py: {e}")
    return

  modem_type = get_modem_type(port)
  print(f"modem.py: detected modem type: {modem_type}")

  # Initial queries
  imei = query_imei(port)
  revision = query_revision(port)
  sim_id = query_sim_id(port)
  mcc_mnc = query_mcc_mnc(port)

  # Configure modem on first run
  configure_modem(port, modem_type, sim_id)

  # Set up eSIM NM connection if needed
  if sim_id:
    try:
      setup_esim_connection(sim_id)
    except Exception as e:
      print(f"modem.py: esim setup failed: {e}")

  while True:
    try:
      state_str = query_modem_state(port)
      signal = query_signal_quality(port)
      access_tech = query_access_technology(port)
      temps = query_temperatures(port)
      net_info = query_network_info(port)
      tx, rx = query_data_usage()

      # Refresh slowly-changing values periodically
      sim_id = query_sim_id(port)
      mcc_mnc = query_mcc_mnc(port)

      state = {
        "state": state_str,
        "access_technology": access_tech,
        "signal_quality": signal,
        "imei": imei,
        "revision": revision,
        "sim_id": sim_id,
        "mcc_mnc": mcc_mnc,
        "temperatures": temps,
        "network_info": net_info,
        "data_tx": tx,
        "data_rx": rx,
      }
      write_state(state)
    except Exception as e:
      print(f"modem.py: error in main loop: {e}")

    time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
  run()
