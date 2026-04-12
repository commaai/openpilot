#!/usr/bin/env python3
import fcntl
import json
import os
import serial
import signal
import subprocess
import tempfile
import termios
import time
import threading

AT_PORT = "/dev/modem_at0"
PPP_PORT = "/dev/modem_at1"
STATE_PATH = "/dev/shm/modem"
AT_LOCK = "/dev/shm/modem_lpa.lock"  # shared with LPA
AT_INIT = ["ATE0", "ATV1", "AT+CMEE=1", "ATX4", "AT&C1"]
CREG = {0: "not_registered", 1: "home", 2: "searching", 3: "denied", 4: "unknown", 5: "roaming"}
PPPD = [
  "sudo", "pppd", PPP_PORT, "460800", "noauth", "nodetach", "noipdefault", "usepeerdns",
  "defaultroute", "replacedefaultroute", "connect", "/usr/sbin/chat -v -f /dev/shm/modem_chat",
  "lcp-echo-interval", "30", "lcp-echo-failure", "4", "mtu", "1500", "mru", "1500",
  "novj", "novjccomp", "ipcp-accept-local", "ipcp-accept-remote", "nomagic",
  "user", '""', "password", '""',
]
CHAT = (
  "ABORT 'NO CARRIER'\nABORT 'NO DIALTONE'\nABORT 'BUSY'\n"
  "ABORT 'NO ANSWER'\nABORT 'ERROR'\nTIMEOUT 5\n"
  "'' AT\nOK ATD*99***{cid}#\nCONNECT ''\n"
)


class Modem:
  def __init__(self):
    self._ser = None
    self.running = True
    self._t0 = time.monotonic()
    self._ppp = None
    self._reset = threading.Event()
    self._cid = 1
    self._reconnect_count = 0
    self.S = {
      "state": "init", "connected": False, "ip_address": "", "iccid": "", "imei": "",
      "modem_version": "", "signal_strength": 0, "signal_quality": 0,
      "network_type": "unknown", "operator": "", "band": "", "channel": 0,
      "registration": "unknown", "temperatures": [], "extra": "",
      "tx_bytes": 0, "rx_bytes": 0, "error": "",
    }

  def _ms(self):
    return (time.monotonic() - self._t0) * 1000

  def _ws(self):
    fd = tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False)
    json.dump(self.S, fd)
    fd.flush()
    os.fsync(fd.fileno())
    os.chmod(fd.name, 0o644)
    tmp = fd.name
    fd.close()
    os.replace(tmp, STATE_PATH)

  def _open(self):
    if self._ser:
      try:
        self._ser.close()
      except Exception:
        pass
    self._ser = serial.Serial(AT_PORT, 9600, timeout=5)

  def _at(self, cmd):
    """Send AT command, return response lines. [] on error or if LPA holds port."""
    fd = os.open(AT_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      os.close(fd)
      return []  # LPA is using the port
    try:
      t = time.monotonic()
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
          print(f"[urc] {line}")
          self._reset.set()
          os.system("sudo killall -9 pppd 2>/dev/null")
        lines.append(line)
      print(f"[at] {cmd} -> {len(lines)} ({(time.monotonic()-t)*1000:.0f}ms)")
      return lines
    except (RuntimeError, TimeoutError, OSError, serial.SerialException) as e:
      print(f"[at] {cmd} FAIL: {e}")
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
  def _stop_mm():
    """Stop ModemManager so we own the AT/PPP ports."""
    os.system("sudo systemctl stop ModemManager 2>/dev/null")
    print("[mm] stopped")

  def _init(self):
    for c in AT_INIT + [
      "AT$QCSIMSLEEP=0", "AT$QCSIMCFG=SimPowerSave,0",
      "AT+CREG=2", "AT+CGREG=2",
    ]:
      self._at(c)
    # device-specific modem configuration (moved from hardware.py configure_modem)
    try:
      with open("/sys/firmware/devicetree/base/model") as f:
        device = f.read().strip('\x00').split('comma ')[-1]
    except Exception:
      device = ""
    if device == "tizi":
      for c in [
        "AT+QSIMDET=1,0", "AT+QSIMSTAT=1",  # SIM hot swap detection
        'AT+QNVW=5280,0,"0102000000000000"',  # data-centric mode
        'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
        'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
      ]:
        self._at(c)
    else:
      self._at('AT$QCPCFG=usbNet,1')  # ethernet config for EG916
    r = self._at("AT+CGSN")
    if r:
      self.S["imei"] = r[0].strip()
    v = self._atv("AT+QCCID", "+QCCID:")
    if v:
      self.S["iccid"] = v
    r = self._at("AT+GMR")
    if r:
      self.S["modem_version"] = r[0].strip()

  def _pdp(self):
    # find highest CID with carrier APN
    self._cid = 1
    best = None
    for line in self._at("AT+CGDCONT?"):
      if "+CGDCONT:" not in line:
        continue
      p = line.split(":", 1)[1].strip().split(",")
      if len(p) >= 3:
        c, a = int(p[0]), p[2].strip('"')
        if a and a != "ims":
          best = (c, a)
    if best:
      self._cid = best[0]
      print(f"[pdp] APN '{best[1]}' CID {self._cid}")
    else:
      self._at('AT+CGDCONT=1,"IP",""')
      print("[pdp] no APN found, using CID 1")

  def _wait_reg(self, timeout=60):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
      v = self._atv("AT+CREG?", "+CREG:")
      if v:
        try:
          reg = CREG.get(int(v.split(",")[1].strip('"')), "unknown")
        except (ValueError, IndexError):
          reg = "unknown"
        if reg in ("home", "roaming"):
          print(f"[timing] reg: {(time.monotonic()-t)*1000:.0f}ms ({reg})")
          self.S["registration"] = reg
          return True
      time.sleep(0.5)
    return False

  def _boot(self):
    self._open()
    time.sleep(1)
    self._init()
    self._pdp()
    if not self._wait_reg(timeout=30):
      return False
    self.S["state"] = "connecting"
    self._ws()
    self._start_ppp()
    t = time.monotonic()
    while not self.S["connected"] and time.monotonic() - t < 30:
      time.sleep(0.2)
    return self.S["connected"]

  @staticmethod
  def _reset_data_port():
    """Flash data port (drop DTR per MM's mm-port-serial.c) then run AT init sequence."""
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
      print("[flash] data port reset")
    except Exception as e:
      print(f"[flash] {e}")

  # -- PPP --

  def _kill_ppp(self):
    os.system("sudo killall -9 pppd 2>/dev/null")
    if self._ppp and self._ppp.is_alive():
      self._ppp.join(timeout=5)

  def _start_ppp(self):
    with open("/dev/shm/modem_chat", "w") as f:
      f.write(CHAT.format(cid=self._cid))

    def run():
      fails = 0
      while self.running and not self._reset.is_set():
        if fails > 0:
          self._reset_data_port()
        print(f"[ppp] dial (T+{self._ms():.0f}ms)")
        try:
          proc = subprocess.Popen(PPPD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
          ok = False
          for raw in proc.stdout:
            line = raw.decode(errors="ignore").strip()
            if not line:
              continue
            print(f"[pppd T+{self._ms():.0f}ms] {line}")
            if "local  IP address" in line:
              ip = line.split("local  IP address")[-1].strip()
              self.S.update(ip_address=ip, connected=True, state="connected")
              self._ws()
              ok, fails = True, 0
              self._reconnect_count = 0
              print(f"[timing] ppp: {self._ms():.0f}ms (IP: {ip})")
            elif "Connection terminated" in line or "Modem hangup" in line:
              self.S.update(connected=False, state="disconnected", ip_address="")
              self._ws()
          proc.wait()
          if not ok:
            fails += 1
            print(f"[ppp] fail {fails}/3")
        except Exception as e:
          print(f"[ppp] {e}")
          fails += 1
        if fails >= 3:
          self._reset.set()
          return
        if not ok:
          self.S.update(connected=False, state="reconnecting")
          self._ws()

    self._ppp = threading.Thread(target=run, daemon=True)
    self._ppp.start()

  # -- health / recovery --

  def _healthy(self):
    if not os.path.exists(AT_PORT):
      return False
    if self._reset.is_set():
      return False
    return True

  def _reconnect(self):
    self._reconnect_count += 1
    print(f"\n{'='*60}\n[reconnect] attempt {self._reconnect_count}\n{'='*60}")
    self.S.update(state="reconnecting", connected=False, ip_address="")
    self._ws()
    self._reset.set()
    self._kill_ppp()
    self._reset_data_port()
    self._reset.clear()
    if self._boot():
      self._reconnect_count = 0

  def _poll(self):
    v = self._atv("AT+CSQ", "+CSQ:")
    if v:
      try:
        rssi = int(v.split(",")[0])
        if rssi != 99:
          self.S["signal_strength"] = rssi
          self.S["signal_quality"] = min(100, int(rssi / 31.0 * 100))
      except (ValueError, IndexError):
        pass
    v = self._atv("AT+COPS?", "+COPS:")
    if v:
      p = v.split(",")
      try:
        if len(p) >= 3:
          self.S["operator"] = p[2].strip('"')
        if len(p) >= 4:
          self.S["network_type"] = {0: "gsm", 2: "utran", 7: "lte"}.get(int(p[3]), "unknown")
      except (ValueError, IndexError):
        pass
    v = self._atv("AT+QNWINFO", "+QNWINFO:")
    if v:
      info = v.replace('"', '').split(",")
      try:
        if len(info) >= 4:
          self.S["band"] = info[2]
          self.S["channel"] = int(info[3])
      except (ValueError, IndexError):
        pass
    v = self._atv('AT+QENG="servingcell"', "+QENG:")
    if v:
      self.S["extra"] = v.replace('"', '')
    v = self._atv("AT+QTEMP", "+QTEMP:")
    if v:
      try:
        self.S["temperatures"] = [t for t in (int(x) for x in v.split(",") if x.strip()) if t != 255]
      except (ValueError, IndexError):
        pass
    # ppp0 IP and data usage
    try:
      r = subprocess.run(["ip", "-4", "addr", "show", "ppp0"], capture_output=True, text=True, timeout=2)
      ip = next((l.strip().split()[1].split("/")[0] for l in r.stdout.splitlines() if "inet " in l), None)
      if ip:
        self.S.update(ip_address=ip, connected=True, state="connected")
      elif self.S["connected"]:
        self.S.update(connected=False, state="registered", ip_address="")
    except Exception:
      pass
    try:
      with open("/sys/class/net/ppp0/statistics/tx_bytes") as f:
        self.S["tx_bytes"] = int(f.read().strip())
      with open("/sys/class/net/ppp0/statistics/rx_bytes") as f:
        self.S["rx_bytes"] = int(f.read().strip())
    except Exception:
      pass
    self._ws()

  def run(self):
    print(f"{'='*60}\nmodem.py {time.strftime('%H:%M:%S')}\n{'='*60}")
    self._stop_mm()
    os.system("sudo killall pppd 2>/dev/null")
    if self._boot():
      print(f"\n{'='*60}\nBOOT {self._ms():.0f}ms\n{'='*60}")

    last_poll = 0.0
    while self.running:
      try:
        if not self._healthy():
          self._reconnect()
          last_poll = time.monotonic()
        elif time.monotonic() - last_poll >= 10:
          self._poll()
          last_poll = time.monotonic()
      except Exception as e:
        print(f"[err] {e}")
      time.sleep(2)

  def stop(self):
    self.running = False
    self._reset.set()
    self._kill_ppp()
    if self._ser:
      self._ser.close()


if __name__ == "__main__":
  m = Modem()

  def _sig(*_):
    m.running = False

  signal.signal(signal.SIGINT, _sig)
  signal.signal(signal.SIGTERM, _sig)
  m.run()
  m.stop()
