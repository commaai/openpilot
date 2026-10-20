#!/usr/bin/python3
"""Wi-Fi policy over wpa_supplicant's control socket; no D-Bus or NetworkManager.

The root daemon owns credentials, DHCP and hotspot processes. The UI sends one
JSON request per Unix stream connection. Only root/netdev can access the socket.
"""
import configparser
import hashlib
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import select
import signal
import sys
import socket
import subprocess
import time

SOCKET = '/run/comma-wifi.sock'
CONFIG = Path('/data/etc/wifi.json')
STATE = Path('/dev/shm/wifi')
HOTSPOT_IP = '192.168.43.1'


def atomic_json(path, value, mode=0o600):
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + '.tmp')
  with open(tmp, 'w', opener=lambda p, flags: os.open(p, flags, mode)) as f:
    json.dump(value, f)
    f.flush()
    os.fsync(f.fileno())
  tmp.chmod(mode)
  tmp.replace(path)
  fd = os.open(path.parent, os.O_DIRECTORY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def request(command, timeout=3, **args):
  with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    sock.settimeout(timeout)
    sock.connect(SOCKET)
    sock.sendall(json.dumps({'command': command, **args}).encode() + b'\n')
    with sock.makefile('rb') as f:
      result = json.loads(f.readline(262144))
    if 'error' in result:
      raise ValueError(result['error'])
    return result


def unescape_ssid(value):
  # Supplicant uses printf escapes, not Python's unicode_escape (which mangles UTF-8).
  return re.sub(rb'\\(x[0-9a-fA-F]{2}|[\\"netr])',
                lambda m: bytes([int(m[1][1:], 16)]) if m[1].startswith(b'x') else
                {b'\\': b'\\', b'"': b'"', b'n': b'\n', b'e': b'\x1b', b't': b'\t', b'r': b'\r'}[m[1]],
                value.encode()).decode('utf-8', 'replace')


def validate_network(ssid, password='', hidden=False, metered=0):
  if not isinstance(ssid, str) or not 1 <= len(ssid.encode()) <= 32 or '\x00' in ssid:
    raise ValueError('SSID must be 1–32 UTF-8 bytes')
  if not isinstance(password, str) or (password and not (8 <= len(password.encode()) <= 63 or re.fullmatch('[0-9a-fA-F]{64}', password))):
    raise ValueError('password must be 8–63 bytes or a 64-digit PSK')
  if metered not in (0, 1, 2):
    raise ValueError('invalid metered setting')
  return {'ssid': ssid, 'password': password, 'hidden': bool(hidden), 'metered': metered}


def keyfile_string(value):
  return re.sub(r'\\([sntr\\])', lambda m: {'s': ' ', 'n': '\n', 't': '\t', 'r': '\r', '\\': '\\'}[m[1]], value)


def migrate_connections(directory):
  networks, hotspot = {}, None
  for path in sorted(directory.glob('*')):
    if not path.is_file():
      continue
    try:
      cp = configparser.ConfigParser(interpolation=None)
      cp.read(path)
      wifi = 'wifi' if cp.has_section('wifi') else '802-11-wireless'
      security = 'wifi-security' if cp.has_section('wifi-security') else '802-11-wireless-security'
      ssid = cp.get(wifi, 'ssid')
      if re.fullmatch(r'(\d+;)+', ssid):
        ssid = bytes(int(x) for x in ssid.split(';') if x).decode('utf-8')
      else:
        ssid = keyfile_string(ssid)
      if cp.get(security, 'key-mgmt', fallback='none') not in ('none', 'wpa-psk'):
        logging.warning('Skipping unsupported Wi-Fi security in %s', path.name)
        continue
      profile = validate_network(ssid, keyfile_string(cp.get(security, 'psk', fallback='')), cp.getboolean(wifi, 'hidden', fallback=False),
                                 cp.getint('connection', 'metered', fallback=0))
      if cp.get(wifi, 'mode', fallback='infrastructure') == 'ap':
        hotspot = profile
      else:
        networks[ssid] = profile
    except (configparser.Error, ValueError, OSError):
      logging.warning('Could not migrate %s; leaving original file intact', path.name)
  return networks, hotspot


class Supplicant:
  def __init__(self, path='/run/wpa_supplicant/wlan0'):
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    self.sock.settimeout(2)
    self.sock.bind(f'\0wifi-{os.getpid()}-{time.monotonic_ns()}')
    self.sock.connect(path)

  def call(self, command):
    self.sock.send(command.encode())
    result = self.sock.recv(262144).decode('utf-8', 'replace').strip()
    if result.startswith(('FAIL', 'UNKNOWN COMMAND')):
      # Do not include the command: SET_NETWORK may contain credentials.
      raise OSError('wpa_supplicant rejected request')
    return result

  def close(self):
    self.sock.close()


def stop_process(proc):
  if proc is not None:
    proc.terminate()
    try:
      proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait()


class Wifi:
  def __init__(self, ctrl, config=CONFIG):
    self.ctrl, self.config = ctrl, config
    if config.exists():
      self.data = json.loads(config.read_text())
      if self.data.get('version') != 1:
        raise ValueError('unsupported Wi-Fi config version')
    else:
      networks, hotspot = migrate_connections(Path('/data/etc/NetworkManager/system-connections'))
      dongle = Path('/data/params/d/DongleId')
      name = 'weedle-' + dongle.read_text().strip()[:4] if dongle.exists() else 'weedle'
      self.data = {'version': 1, 'networks': networks, 'hotspot': hotspot or validate_network(name, 'swagswagcomma')}
      self.save()
    self.ids = {}
    self.pending = None
    self.auth_error = None
    self.error_id = 0
    self.dhcp = None
    self.dnsmasq = None
    self.link = None
    self.hotspot = False
    self.forward = False
    self.radio = True
    self.lease_token = ''
    self.snapshot = {}
    self.reload()

  def save(self):
    atomic_json(self.config, self.data)

  def add(self, profile, ap=False):
    profile = validate_network(**profile)
    nid = self.ctrl.call('ADD_NETWORK')
    if not nid.isdecimal():
      raise OSError('invalid supplicant network ID')
    try:
      values = {'ssid': profile['ssid'].encode().hex(), 'scan_ssid': int(profile['hidden'])}
      if profile['password']:
        psk = profile['password']
        if len(psk) != 64:
          psk = hashlib.pbkdf2_hmac('sha1', psk.encode(), profile['ssid'].encode(), 4096, 32).hex()
        values.update(key_mgmt='WPA-PSK', psk=psk)
      else:
        values['key_mgmt'] = 'NONE'
      if ap:
        values.update(mode=2, frequency=2412, proto='RSN', pairwise='CCMP', group='CCMP')
      for key, value in values.items():
        self.ctrl.call(f'SET_NETWORK {nid} {key} {value}')
    except Exception:
      self.ctrl.call(f'REMOVE_NETWORK {nid}')
      raise
    return nid

  def reload(self):
    self.ctrl.call('REMOVE_NETWORK all')
    self.ids = {ssid: self.add(p) for ssid, p in self.data['networks'].items()}
    if self.radio:
      self.ctrl.call('ENABLE_NETWORK all')
      self.ctrl.call('RECONNECT')

  def clear_address(self):
    stop_process(self.dhcp)
    self.dhcp = None
    self.link = None
    Path('/run/dhcp-wlan0.json').unlink(missing_ok=True)
    subprocess.run(['ip', '-4', 'route', 'flush', 'dev', 'wlan0'], check=False)
    subprocess.run(['ip', '-4', 'addr', 'flush', 'dev', 'wlan0'], check=False)
    subprocess.run(['resolvconf', '-d', 'wlan0'], check=False)

  def stop_hotspot(self):
    stop_process(self.dnsmasq)
    self.dnsmasq = None
    self.hotspot = False
    self.clear_address()
    subprocess.run(['iptables', '-t', 'nat', '-D', 'POSTROUTING', '-s', '192.168.43.0/24', '!', '-o', 'wlan0', '-j', 'MASQUERADE'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Path('/proc/sys/net/ipv4/ip_forward').write_text('0')

  def command(self, command, **args):
    if command == 'status':
      return {**self.snapshot, 'hotspot_password': self.data['hotspot']['password']}
    if command == 'scan':
      if not self.hotspot and self.radio:
        self.ctrl.call('SCAN')
    elif command in ('connect', 'activate'):
      if command == 'connect':
        profile = validate_network(args['ssid'], args.get('password', ''), args.get('hidden', False))
      else:
        profile = self.data['networks'][args['ssid']]
      if self.hotspot:
        self.stop_hotspot()
        self.reload()
      self.clear_address()
      self.auth_error = None
      ssid = profile['ssid']
      if ssid in self.ids:
        self.ctrl.call(f'REMOVE_NETWORK {self.ids.pop(ssid)}')
      nid = self.add(profile)
      self.ids[ssid] = nid
      self.pending = profile
      self.ctrl.call(f'SELECT_NETWORK {nid}')
    elif command == 'forget':
      ssid = args['ssid']
      if ssid in self.ids:
        self.ctrl.call(f'REMOVE_NETWORK {self.ids.pop(ssid)}')
      if self.pending and self.pending['ssid'] == ssid:
        self.pending = None
      self.data['networks'].pop(ssid, None)
      self.save()
      if self.pending is None and not self.hotspot:
        self.ctrl.call('ENABLE_NETWORK all')
    elif command == 'metered':
      ssid = self.snapshot.get('ssid')
      if ssid in self.data['networks']:
        self.data['networks'][ssid] = validate_network(**{**self.data['networks'][ssid], 'metered': args['metered']})
        self.save()
    elif command == 'hotspot_config':
      profile = validate_network(args.get('ssid', self.data['hotspot']['ssid']), args.get('password', self.data['hotspot']['password']))
      if not profile['password']:
        raise ValueError('hotspot requires a password')
      self.data['hotspot'] = profile
      self.save()
      if self.hotspot:
        self.command('hotspot', active=True, forward=self.forward)
    elif command == 'hotspot':
      self.stop_hotspot()
      self.pending = None
      if args['active']:
        self.ctrl.call('REMOVE_NETWORK all')
        nid = self.add(self.data['hotspot'], ap=True)
        self.ctrl.call(f'SELECT_NETWORK {nid}')
        self.hotspot = True
        self.forward = bool(args.get('forward', False))
        subprocess.run(['ip', 'addr', 'replace', f'{HOTSPOT_IP}/24', 'dev', 'wlan0'], check=True)
        if self.forward:
          subprocess.run(['iptables', '-t', 'nat', '-A', 'POSTROUTING', '-s', '192.168.43.0/24', '!', '-o', 'wlan0', '-j', 'MASQUERADE'], check=True)
          Path('/proc/sys/net/ipv4/ip_forward').write_text('1')
        self.dnsmasq = subprocess.Popen(['dnsmasq', '--keep-in-foreground', '--conf-file=/dev/null', '--interface=wlan0', '--bind-interfaces',
                                        '--except-interface=lo', '--dhcp-range=192.168.43.10,192.168.43.200,12h',
                                        '--dhcp-leasefile=/run/hotspot.leases', '--resolv-file=/run/resolv.conf'])
      else:
        self.reload()
    elif command == 'radio':
      self.radio = bool(args['enabled'])
      self.pending = None
      self.stop_hotspot()
      self.reload()
    else:
      raise ValueError('unsupported Wi-Fi command')
    return {'ok': True}

  def event(self, event):
    if 'CTRL-EVENT-SSID-TEMP-DISABLED' in event and 'WRONG_KEY' in event:
      match = re.search(r'\bid=(\d+)', event)
      if match:
        ssid = next((name for name, nid in self.ids.items() if nid == match[1]), None)
        if ssid and (self.pending is None or self.pending['ssid'] == ssid):
          self.auth_error = ssid
          self.error_id += 1
          self.ctrl.call(f'REMOVE_NETWORK {self.ids.pop(ssid)}')
          self.pending = None
          self.ctrl.call('ENABLE_NETWORK all')

  def poll(self):
    if self.hotspot and (self.dnsmasq is None or self.dnsmasq.poll() is not None):
      raise OSError('hotspot DHCP server exited')
    status = dict(line.split('=', 1) for line in self.ctrl.call('STATUS').splitlines() if '=' in line)
    ssid = unescape_ssid(status.get('ssid', '')) or None
    associated = (status.get('wpa_state') == 'COMPLETED' and
                  (self.pending is None or status.get('id') == self.ids.get(self.pending['ssid'])))
    key = status.get('id') if associated and not self.hotspot else None
    if not self.hotspot:
      if key != self.link or (self.dhcp is not None and self.dhcp.poll() is not None):
        self.clear_address()
        if key:
          self.lease_token = str(time.monotonic_ns())
          self.dhcp = subprocess.Popen(['busybox', 'udhcpc', '-f', '-R', '-i', 'wlan0', '-x', 'hostname:' + socket.gethostname(), '-s', '/usr/comma/dhcp.py'],
                                       env={**os.environ, 'DHCP_TOKEN': self.lease_token})
          self.link = key
    address = HOTSPOT_IP if self.hotspot and associated else ''
    if self.dhcp:
      try:
        lease = json.loads(Path('/run/dhcp-wlan0.json').read_text())
        if lease['token'] == self.lease_token:
          address = lease['ip']
      except (OSError, ValueError, KeyError):
        pass
    if address and self.pending and self.pending['ssid'] == ssid:
      self.data['networks'][ssid] = self.pending
      self.pending = None
      self.save()
      self.ctrl.call('ENABLE_NETWORK all')
    networks = {}
    for line in self.ctrl.call('SCAN_RESULTS').splitlines()[1:]:
      cols = line.split('\t', 4)
      if len(cols) != 5:
        continue
      _, _, level, flags, name = cols
      name = unescape_ssid(name)
      if not name:
        continue
      try:
        strength = max(0, min(100, 2 * (int(level) + 100)))
      except ValueError:
        continue
      security = 4 if any(x in flags for x in ('WEP', 'EAP', 'SAE', 'OWE')) and 'PSK' not in flags else 1 if 'PSK' in flags else 0
      if name not in networks or strength > networks[name]['strength']:
        networks[name] = {'ssid': name, 'strength': strength, 'security_type': security, 'is_tethering': False}
    if self.hotspot:
      networks[self.data['hotspot']['ssid']] = {'ssid': self.data['hotspot']['ssid'], 'strength': 100, 'security_type': 2, 'is_tethering': True}
    connecting = status.get('wpa_state', 'DISCONNECTED') not in ('DISCONNECTED', 'INACTIVE', 'INTERFACE_DISABLED')
    state = 2 if address else 1 if connecting or self.pending or self.hotspot else 0
    self.snapshot = {'ssid': (self.pending or {}).get('ssid') or ssid, 'status': state, 'ipv4_address': address,
                     'metered': self.data['networks'].get(ssid, {}).get('metered', 0), 'saved': list(self.data['networks']),
                     'networks': list(networks.values()), 'hotspot': self.hotspot, 'hotspot_ssid': self.data['hotspot']['ssid'],
                     'auth_error': self.auth_error, 'error_id': self.error_id}
    atomic_json(STATE, self.snapshot, 0o644)

  def close(self):
    self.stop_hotspot()
    STATE.unlink(missing_ok=True)


def main():
  logging.basicConfig(level=logging.INFO)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
  lock = open('/run/wifi.lock', 'w')
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  ctrl = Supplicant()
  events = Supplicant()
  events.call('ATTACH')
  wifi = Wifi(ctrl)
  wifi.stop_hotspot()  # Clear leases/NAT left by a previous unclean daemon exit.
  server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  Path(SOCKET).unlink(missing_ok=True)
  server.bind(SOCKET)
  import grp
  os.chown(SOCKET, 0, grp.getgrnam('netdev').gr_gid)
  os.chmod(SOCKET, 0o660)
  server.listen(8)
  last_poll = 0
  try:
    while True:
      ready, _, _ = select.select([server, events.sock], [], [], .2)
      if events.sock in ready:
        wifi.event(events.sock.recv(65536).decode('utf-8', 'replace'))
      if server in ready:
        conn, _ = server.accept()
        with conn:
          conn.settimeout(1)
          args = {}
          try:
            with conn.makefile('rb') as f:
              args = json.loads(f.readline(8192))
            result = wifi.command(**args)
            if args.get('command') != 'status':
              wifi.poll()
          except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            logging.exception('Wi-Fi request failed')
            if isinstance(args, dict) and args.get('command') in ('connect', 'activate', 'hotspot'):
              wifi.pending = None
              wifi.stop_hotspot()
              wifi.reload()
            result = {'error': 'Wi-Fi request failed'}
          try:
            conn.sendall(json.dumps(result).encode() + b'\n')
          except OSError:
            pass
      if time.monotonic() - last_poll > 1:
        wifi.poll()
        last_poll = time.monotonic()
  finally:
    wifi.close()
    server.close()
    events.close()
    ctrl.close()
    Path(SOCKET).unlink(missing_ok=True)
    lock.close()


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == 'radio' and sys.argv[2] in ('on', 'off'):
    request('radio', enabled=sys.argv[2] == 'on')
  else:
    main()
