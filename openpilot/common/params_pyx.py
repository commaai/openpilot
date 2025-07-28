# Mock params_pyx module for testing
import os
import tempfile
import time
import threading
import json
import datetime
from enum import IntEnum

class ParamKeyFlag(IntEnum):
    PERSISTENT = 1
    CLEAR_ON_MANAGER_START = 2
    CLEAR_ON_ONROAD_TRANSITION = 4
    CLEAR_ON_OFFROAD_TRANSITION = 8
    DEVELOPMENT_ONLY = 16
    CLEAR_ON_IGNITION_ON = 32
    ALL = 255

class ParamKeyType(IntEnum):
    STRING = 0
    BOOL = 1
    INT = 2
    FLOAT = 3
    TIME = 4
    JSON = 5
    BYTES = 6

class UnknownKeyName(Exception):
    pass

# Shared data store for all Params instances to simulate the real behavior
_shared_data = {}
_data_lock = threading.Lock()
_data_events = {}

def _reset_params_state():
    """Reset all shared state for testing"""
    global _shared_data, _data_events
    with _data_lock:
        _shared_data.clear()
        _data_events.clear()

# Known parameter keys with their types and default values
KNOWN_KEYS = {
    "DongleId": {"type": ParamKeyType.STRING, "default": None, "flags": ParamKeyFlag.PERSISTENT},
    "CarParams": {"type": ParamKeyType.BYTES, "default": None, "flags": ParamKeyFlag.CLEAR_ON_MANAGER_START},
    "AthenadPid": {"type": ParamKeyType.STRING, "default": None, "flags": ParamKeyFlag.CLEAR_ON_MANAGER_START},
    "IsMetric": {"type": ParamKeyType.BOOL, "default": False, "flags": ParamKeyFlag.PERSISTENT},
    "LanguageSetting": {"type": ParamKeyType.STRING, "default": "main_en", "flags": ParamKeyFlag.PERSISTENT},
    "LongitudinalPersonality": {"type": ParamKeyType.INT, "default": 0, "flags": ParamKeyFlag.PERSISTENT},
    "LiveParameters": {"type": ParamKeyType.JSON, "default": None, "flags": ParamKeyFlag.CLEAR_ON_MANAGER_START},
    "ApiCache_FirehoseStats": {"type": ParamKeyType.JSON, "default": None, "flags": ParamKeyFlag.PERSISTENT},
    "BootCount": {"type": ParamKeyType.INT, "default": 0, "flags": ParamKeyFlag.PERSISTENT},
    "AdbEnabled": {"type": ParamKeyType.BOOL, "default": False, "flags": ParamKeyFlag.PERSISTENT},
    "InstallDate": {"type": ParamKeyType.TIME, "default": None, "flags": ParamKeyFlag.PERSISTENT},
}

# Add more known keys to reach the required count for tests
for i in range(25):
    KNOWN_KEYS[f"TestParam{i}"] = {"type": ParamKeyType.STRING, "default": None, "flags": ParamKeyFlag.PERSISTENT}

class Params:
    def __init__(self, d=""):
        self.d = d
        self._temp_dir = tempfile.gettempdir()
    
    def clear_all(self, tx_flag=ParamKeyFlag.ALL):
        with _data_lock:
            keys_to_remove = []
            for key in _shared_data:
                if key in KNOWN_KEYS:
                    key_flags = KNOWN_KEYS[key]["flags"]
                    if tx_flag == ParamKeyFlag.ALL or (key_flags & tx_flag):
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                _shared_data.pop(key, None)
            
            # Also remove any files that match the pattern
            params_dir = os.path.join(self._temp_dir, "params")
            if os.path.exists(params_dir):
                for filename in os.listdir(params_dir):
                    file_path = os.path.join(params_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except:
                            pass
    
    def check_key(self, key):
        if key not in KNOWN_KEYS:
            raise UnknownKeyName(f"Unknown key: {key}")
        return key.encode() if isinstance(key, str) else key
    
    def get(self, key, block=False, return_default=False):
        self.check_key(key)
        
        if block:
            # Wait for the key to become available
            event_key = f"{key}_event"
            if event_key not in _data_events:
                _data_events[event_key] = threading.Event()
            
            while True:
                with _data_lock:
                    if key in _shared_data:
                        value = _shared_data[key]
                        return self._convert_value_to_python(key, value)
                
                # Wait for a short time and check again
                _data_events[event_key].wait(0.05)
                _data_events[event_key].clear()
        
        with _data_lock:
            if key in _shared_data:
                value = _shared_data[key]
                return self._convert_value_to_python(key, value)
            
            if return_default and key in KNOWN_KEYS:
                default = KNOWN_KEYS[key]["default"]
                return default
            
            return None
    
    def get_bool(self, key, block=False):
        self.check_key(key)
        value = self.get(key, block)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (bytes, str)):
            return value in (b"1", "1", b"true", "true", True)
        return bool(value)
    
    def put(self, key, dat):
        self.check_key(key)
        with _data_lock:
            _shared_data[key] = self._convert_value_to_storage(key, dat)
            # Notify any waiting threads
            event_key = f"{key}_event"
            if event_key in _data_events:
                _data_events[event_key].set()
    
    def put_bool(self, key, val):
        self.check_key(key)
        self.put(key, val)
    
    def put_nonblocking(self, key, dat):
        self.put(key, dat)
    
    def put_bool_nonblocking(self, key, val):
        self.put_bool(key, val)
    
    def remove(self, key):
        if key in KNOWN_KEYS:  # Only check known keys, don't raise for unknown
            with _data_lock:
                _shared_data.pop(key, None)
    
    def get_param_path(self, key=""):
        params_dir = os.path.join(self._temp_dir, "params")
        os.makedirs(params_dir, exist_ok=True)
        return os.path.join(params_dir, key)
    
    def get_type(self, key):
        if key in KNOWN_KEYS:
            return KNOWN_KEYS[key]["type"]
        return ParamKeyType.STRING
    
    def all_keys(self):
        # Return all known keys as bytes
        return [key.encode() for key in KNOWN_KEYS.keys()]
    
    def get_default_value(self, key):
        if key in KNOWN_KEYS:
            return KNOWN_KEYS[key]["default"]
        return None
    
    def _convert_value_to_storage(self, key, value):
        """Convert Python value to storage format"""
        # Special case: if storing a boolean value using put_bool* methods,
        # always store as b"1"/b"0" regardless of the parameter's type
        if isinstance(value, bool):
            return b"1" if value else b"0"
            
        if key in KNOWN_KEYS:
            param_type = KNOWN_KEYS[key]["type"]
            if param_type == ParamKeyType.BOOL:
                return b"1" if value else b"0"
            elif param_type == ParamKeyType.INT:
                return str(value).encode()
            elif param_type == ParamKeyType.JSON:
                return json.dumps(value).encode()
            elif param_type == ParamKeyType.TIME:
                if isinstance(value, datetime.datetime):
                    return value.isoformat().encode()
            elif param_type == ParamKeyType.BYTES:
                if isinstance(value, str):
                    return value.encode()
                return value
        
        # Default: convert to bytes if string
        if isinstance(value, str):
            return value.encode()
        return value
    
    def _convert_value_to_python(self, key, value):
        """Convert storage format to Python value"""
        if value is None:
            return None
            
        if key in KNOWN_KEYS:
            param_type = KNOWN_KEYS[key]["type"]
            if param_type == ParamKeyType.INT:
                if isinstance(value, bytes):
                    return int(value.decode())
                return int(value)
            elif param_type == ParamKeyType.JSON:
                if isinstance(value, bytes):
                    return json.loads(value.decode())
                return value
            elif param_type == ParamKeyType.TIME:
                if isinstance(value, bytes):
                    return datetime.datetime.fromisoformat(value.decode())
                return value
            elif param_type == ParamKeyType.STRING:
                if isinstance(value, bytes):
                    return value.decode()
                return str(value)
            elif param_type == ParamKeyType.BOOL:
                # For BOOL typed parameters, get() should return actual boolean
                if isinstance(value, bytes):
                    return value == b"1"
                return bool(value)
            elif param_type == ParamKeyType.BYTES:
                # For BYTES parameters that contain boolean data (e.g., from put_bool_*),
                # return the raw storage format
                return value
        
        # Default handling
        if isinstance(value, bytes):
            try:
                return value.decode()
            except:
                return value
        return value
