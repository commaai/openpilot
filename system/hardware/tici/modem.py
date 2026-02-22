#!/usr/bin/env python3
"""
modem.py - Lightweight modem manager for commaai devices
Replaces ModemManager with ~200 lines of Python using AT commands

Supports:
- Quectel EG25 (comma 3X)
- Quectel EG916 (comma four)

Usage:
  python modem.py          # Run as daemon
  python modem.py status   # Check modem status
  python modem.py init     # Initialize modem
"""
import os
import sys
import time
import json
import glob
import serial
import subprocess
from datetime import datetime

# Configuration
STATE_FILE = "/dev/shm/modem_state.txt"
LOG_FILE = "/dev/shm/modem.log"
SERIAL_PORTS = ["/dev/ttyUSB2", "/dev/ttyUSB0", "/dev/ttyACM0"]
TIMEOUT = 5  # AT command timeout

class ModemManager:
    def __init__(self):
        self.serial = None
        self.manufacturer = None
        self.model = None
        self.revision = None
        self.sim_id = None
        self.connected = False
        
    def log(self, message):
        """Log message to file and stdout"""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {message}\n"
        print(log_line, end='')
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(log_line)
        except Exception as e:
            pass
    
    def write_state(self, state):
        """Write modem state to shared file (replaces dbus)"""
        try:
            state['timestamp'] = datetime.now().isoformat()
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.log(f"Failed to write state: {e}")
    
    def find_port(self):
        """Find available serial port for modem"""
        for port in SERIAL_PORTS:
            if os.path.exists(port):
                return port
        # Try to find any ttyUSB port
        ports = glob.glob("/dev/ttyUSB*")
        return ports[0] if ports else None
    
    def connect(self):
        """Open serial connection to modem"""
        port = self.find_port()
        if not port:
            self.log("ERROR: No modem port found")
            return False
        
        try:
            self.serial = serial.Serial(port, 115200, timeout=TIMEOUT)
            self.log(f"Connected to modem on {port}")
            return True
        except Exception as e:
            self.log(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial:
            try:
                self.serial.close()
            except:
                pass
            self.serial = None
    
    def send_at(self, command, timeout=TIMEOUT):
        """Send AT command and return response"""
        if not self.serial:
            return None
        
        try:
            self.serial.write(f"{command}\r\n".encode())
            response = self.serial.read(self.serial.in_waiting or 1024).decode('utf-8', errors='ignore')
            return response.strip()
        except Exception as e:
            self.log(f"AT command failed: {command} - {e}")
            return None
    
    def init(self):
        """Initialize modem with AT commands"""
        self.log("Initializing modem...")
        
        # Basic commands
        cmds = [
            ("AT", "OK"),  # Test
            ("ATI", None),  # Get manufacturer info
            ("AT+CGMR", None),  # Get revision
            ("AT+CGSN", None),  # Get IMEI
            ("AT+QSIMDET=1,0", None),  # SIM detection
            ("AT+QSIMSTAT=1", None),  # SIM status
        ]
        
        for cmd, expected in cmds:
            resp = self.send_at(cmd)
            if expected and expected not in resp:
                self.log(f"Command failed: {cmd} - {resp}")
            else:
                self.log(f"OK: {cmd}")
            time.sleep(0.1)
        
        # Configure as data-centric
        self.send_at('AT+QNVW=5280,0,"0102000000000000"')
        self.send_at('AT+QNVFW="/nv/item_files/ims/IMS_enable",00')
        self.send_at('AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01')
        
        # Disable SIM sleep
        self.send_at("AT$QCSIMSLEEP=0")
        self.send_at("AT$QCSIMCFG=SimPowerSave,0")
        
        self.log("Modem initialization complete")
        return True
    
    def get_info(self):
        """Get modem information"""
        if not self.serial:
            return None
        
        info = {
            'manufacturer': None,
            'model': None,
            'revision': None,
            'imei': None,
            'sim_id': None,
        }
        
        # Get manufacturer
        resp = self.send_at("ATI")
        if resp:
            lines = resp.split('\n')
            for line in lines:
                if 'Quectel' in line or 'Cavli' in line:
                    info['manufacturer'] = line.strip()
        
        # Get revision
        resp = self.send_at("AT+CGMR")
        if resp:
            info['revision'] = resp.split('\n')[0].strip()
        
        # Get IMEI
        resp = self.send_at("AT+CGSN")
        if resp:
            info['imei'] = resp.split('\n')[0].strip()
        
        # Get SIM ID
        resp = self.send_at("AT+QCCID")
        if resp and 'OK' in resp:
            info['sim_id'] = resp.split('\n')[0].strip()
        
        return info
    
    def get_signal(self):
        """Get signal strength"""
        resp = self.send_at("AT+CSQ")
        if resp and 'OK' in resp:
            try:
                csq = int(resp.split(':')[1].split(',')[0].strip())
                return {'csq': csq, 'percent': min(100, csq * 100 / 31)}
            except:
                pass
        return {'csq': 0, 'percent': 0}
    
    def get_temps(self):
        """Get modem temperatures"""
        resp = self.send_at("AT+QTEMP")
        if resp and 'OK' in resp:
            try:
                # Format: +QTEMP: <mode>,<temp1>,<temp2>,...
                parts = resp.split(':')[1].strip().split(',')
                temps = [int(t) for t in parts[1:] if t.strip().isdigit()]
                return list(filter(lambda t: t != 255, temps))
            except:
                pass
        return []
    
    def connect_lte(self):
        """Establish LTE data connection"""
        self.log("Establishing LTE connection...")
        
        # Set APN (adjust for your carrier)
        self.send_at('AT+CGDCONT=1,"IP","ctlte"')
        
        # Attach to network
        self.send_at("AT+CGATT=1")
        time.sleep(2)
        
        # Activate bearer
        self.send_at("AT+QIACT=1")
        time.sleep(5)
        
        # Check connection
        resp = self.send_at("AT+QISTATE")
        if resp and 'CONNECTED' in resp:
            self.connected = True
            self.log("LTE connected")
            return True
        
        self.log("LTE connection pending")
        return False
    
    def get_data_usage(self):
        """Get TX/RX bytes"""
        # Simple approach: read from /sys/class/net/wwan0/statistics
        tx = rx = 0
        try:
            with open('/sys/class/net/wwan0/statistics/tx_bytes', 'r') as f:
                tx = int(f.read().strip())
            with open('/sys/class/net/wwan0/statistics/rx_bytes', 'r') as f:
                rx = int(f.read().strip())
        except:
            pass
        return tx, rx
    
    def run_daemon(self):
        """Run as background daemon"""
        self.log("Starting modem daemon...")
        
        # Initial setup
        if not self.connect():
            self.log("Failed to connect to modem")
            sys.exit(1)
        
        # Get info
        info = self.get_info()
        if info:
            self.manufacturer = info.get('manufacturer')
            self.revision = info.get('revision')
            self.sim_id = info.get('sim_id')
            self.log(f"Modem: {self.manufacturer}, Rev: {self.revision}, SIM: {self.sim_id}")
        
        # Initialize
        self.init()
        
        # Connect to LTE
        self.connect_lte()
        
        # Main loop - update state periodically
        while True:
            try:
                signal = self.get_signal()
                temps = self.get_temps()
                tx, rx = self.get_data_usage()
                
                state = {
                    'status': 'connected' if self.connected else 'disconnected',
                    'manufacturer': self.manufacturer,
                    'revision': self.revision,
                    'sim_id': self.sim_id,
                    'signal': signal,
                    'temps': temps,
                    'tx_bytes': tx,
                    'rx_bytes': rx,
                }
                
                self.write_state(state)
                self.log(f"State: signal={signal['csq']}, temps={temps}, tx={tx}, rx={rx}")
                
            except Exception as e:
                self.log(f"Error in daemon loop: {e}")
            
            time.sleep(10)  # Update every 10 seconds


def main():
    manager = ModemManager()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'status':
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    print(json.dumps(json.load(f), indent=2))
            else:
                print("Modem not initialized")
        elif cmd == 'init':
            if manager.connect():
                manager.init()
                manager.get_info()
        elif cmd == 'test':
            if manager.connect():
                print("AT Response:", manager.send_at("AT"))
        else:
            print(f"Unknown command: {cmd}")
    else:
        # Run as daemon
        manager.run_daemon()


if __name__ == '__main__':
    main()
