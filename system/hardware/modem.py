#!/usr/bin/env python3
import asyncio
import json
import logging
import time
from pathlib import Path

try:
    import aioserial
except ImportError:
    logging.warning("aioserial not found. Install via 'pip install aioserial' for production.")

try:
    from pydbus import SystemBus
    from gi.repository import GLib
    import threading
    DBUS_SUPPORT = True
except ImportError:
    DBUS_SUPPORT = False

# Openpilot #37277 Modem Prototype - Async & Fast Boot Edition
# Targets: Comma 3X (EG25) & Comma 4 (EG916)

LOG_FILE = "/dev/shm/modem_state.txt"
TTY_PORT = "/dev/ttyUSB2"
IFACE = "wwan0"
BAUD = 115200

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ModemDBusObject:
    """
    Minimal D-Bus object to satisfy Openpilot's legacy manager/hardwared.
    Emits the StateChanged signal to notify the system immediately.
    """
    dbus = """
    <node>
        <interface name="org.freedesktop.ModemManager1.Modem">
            <property name="State" type="i" access="read" />
            <signal name="StateChanged">
                <arg type="i" name="old" />
                <arg type="i" name="new" />
                <arg type="u" name="reason" />
            </signal>
        </interface>
    </node>
    """
    def __init__(self):
        self._state = 0  # 0: Failed/Unknown, 8: Connected (MM_MODEM_STATE_CONNECTED)

    @property
    def State(self):
        return self._state

    def set_state(self, new_state):
        if self._state != new_state:
            old = self._state
            self._state = new_state
            self.StateChanged(old, new_state, 0)

class AsyncATQueue:
    """Async AT Command Queue with Hard Reset fail-safe."""
    def __init__(self, port=TTY_PORT, baud=BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = asyncio.Lock()
        self.fail_count = 0

    async def connect(self):
        if not self.ser:
            self.ser = aioserial.AioSerial(port=self.port, baudrate=self.baud, timeout=0.1)
            self.ser.reset_input_buffer()

    async def send(self, cmd, expected="OK", timeout=2.0):
        await self.connect()
        async with self.lock:  # Prevent overlapping AT commands
            self.ser.write(f"{cmd}\r\n".encode())
            start_time = time.monotonic()
            lines = []
            
            while time.monotonic() - start_time < timeout:
                if self.ser.in_waiting:
                    line = (await self.ser.readline_async()).decode('utf-8', errors='ignore').strip()
                    if line:
                        lines.append(line)
                        if expected in line or "ERROR" in line:
                            break
                else:
                    await asyncio.sleep(0.01)
            
            # --- Hard Reset Trigger Logic ---
            if not any(expected in r for r in lines):
                self.fail_count += 1
                logging.warning(f"AT Command '{cmd}' failed. Count: {self.fail_count}/3")
                if self.fail_count >= 3:
                    logging.error("HARD RESET TRIGGERED: 3 consecutive AT failures. Rebooting modem...")
                    self.ser.write(b"AT+CFUN=1,1\r\n")  # Modem hardware reboot
                    self.fail_count = 0
                    await asyncio.sleep(10)  # Wait for modem to bounce back
            else:
                self.fail_count = 0  # Reset on success
                
            return lines

class AsyncModemController:
    def __init__(self):
        self.at = AsyncATQueue()
        self.state = {"state": "disabled", "signal_quality": 0}
        
        if DBUS_SUPPORT:
            self.bus = SystemBus()
            self.dbus_obj = ModemDBusObject()
            try:
                self.bus.publish("org.freedesktop.ModemManager1", self.dbus_obj)
            except Exception as e:
                logging.error(f"Failed to publish D-Bus interface: {e}")

    def update_state(self, **kwargs):
        self.state.update(kwargs)
        # Write to shm file
        with open(LOG_FILE, "w") as f:
            json.dump(self.state, f)

        # Fire D-Bus signal for manager.py
        if DBUS_SUPPORT:
            mm_state = 8 if self.state["state"] == "connected" else 0
            self.dbus_obj.set_state(mm_state)

    async def run_shell(self, cmd):
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode().strip()

    async def initialize(self):
        self.update_state(state="initializing")
        await self.at.send("ATZ")
        await self.at.send("ATE0")
        
        # SIM Check
        if not any("READY" in r for r in await self.at.send("AT+CPIN?")):
            logging.error("SIM not ready")
            return False

        # Fast registration wait (Optimized for <1min boot)
        logging.info("Waiting for network registration...")
        for _ in range(15):
            res = await self.at.send("AT+CREG?", timeout=1.0)
            if any("+CREG: 0,1" in r or "+CREG: 0,5" in r for r in res):
                return True
            await asyncio.sleep(0.5)
        return False

    async def connect_data(self):
        self.update_state(state="connecting")
        
        # PDP Context & RMNET Call
        await self.at.send("AT+CGACT=1,1", timeout=5.0)
        await self.at.send("AT$QCRMCALL=1,1", timeout=5.0)
        
        # Link up
        await self.run_shell(f"ip link set {IFACE} up")
        
        # Fast DHCP
        logging.info("Requesting IP via DHCP...")
        await self.run_shell(f"udhcpc -i {IFACE} -q -t 3")
        
        # --- IP Route Logic (Metric 200) ---
        logging.info("Enforcing Wi-Fi Priority: Setting LTE route metric to 200...")
        # Remove default route added by DHCP (which might have metric 0) and re-add with metric 200
        await self.run_shell(f"ip route del default dev {IFACE}")
        await self.run_shell(f"ip route add default dev {IFACE} metric 200")
        
        self.update_state(state="connected")
        logging.info("Connection established! Ready for google.com ping.")

    async def monitor(self):
        while True:
            csq = await self.at.send("AT+CSQ")
            for r in csq:
                if "+CSQ:" in r:
                    try:
                        self.update_state(signal_quality=int(r.split(":")[1].split(",")[0]))
                    except ValueError:
                        pass
            await asyncio.sleep(5)

    async def loop(self):
        while True:
            try:
                if await self.initialize():
                    await self.connect_data()
                    await self.monitor()
            except Exception as e:
                logging.error(f"Modem loop crashed: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    if DBUS_SUPPORT:
        glib_loop = GLib.MainLoop()
        threading.Thread(target=glib_loop.run, daemon=True).start()

    controller = AsyncModemController()
    asyncio.run(controller.loop())
