# Mock capnp module for testing purposes
import sys
import time
import random


class MockDynamicStructBuilder:
    """Mock for capnp._DynamicStructBuilder"""
    
    def __init__(self, which_name="mock"):
        self._which_name = which_name
        self.logMonoTime = int(time.monotonic() * 1e9)
        self.valid = False
        # Initialize with mock data for car state fields
        self.vEgo = 0.0
        self.aEgo = 0.0
        self.brake = 0.0
        self.steeringAngleDeg = 0.0
        # Create a mock carState attribute that can be accessed
        self.carState = self
    
    def __getattr__(self, name):
        # Return self for any attribute access to allow chaining
        return self
    
    def __setattr__(self, name, value):
        # Allow setting attributes
        self.__dict__[name] = value
    
    def to_bytes(self):
        # Instead of generic mock data, serialize this object's state
        import json
        # Create a serializable representation of the object state
        state = {}
        for key, value in self.__dict__.items():
            if not callable(value):
                # Skip self-referential attributes
                if value is self:
                    continue
                # Always include _which_name for reconstruction
                if key == '_which_name':
                    state[key] = value
                elif not key.startswith('_'):
                    state[key] = value if not hasattr(value, '__dict__') else str(value)
        return json.dumps(state).encode('utf-8')
    
    def init(self, name, size=None):
        # Set the which name when init is called
        self._which_name = name
        return self
    
    def which(self):
        return self._which_name
    
    def as_reader(self):
        return MockDynamicStructReader(self._which_name)
    
    def clear_write_flag(self):
        """Mock implementation of clear_write_flag"""
        pass


class MockDynamicStructReader:
    """Mock for capnp._DynamicStructReader"""
    
    def __init__(self, which_name="mock"):
        self._which_name = which_name
        self.logMonoTime = int(time.monotonic() * 1e9)
        self.valid = True
        # Initialize with mock data for car state fields
        self.vEgo = random.random() * 10
        self.aEgo = random.random() * 10
        self.brake = random.random() * 10
        self.steeringAngleDeg = random.random() * 10
        # Create a mock carState attribute that can be accessed
        self.carState = self
    
    def __getattr__(self, name):
        # For carState, return self to allow field access
        if name == "carState":
            return self
        # Return mock numeric values for common fields
        if name in ["vEgo", "aEgo", "brake", "steeringAngleDeg"]:
            return getattr(self, name, random.random() * 10)
        # Return self for any other attribute access to allow chaining
        return self
    
    def which(self):
        return self._which_name


class MockContextManager:
    """Context manager wrapper for mock objects"""
    
    def __init__(self, obj):
        self.obj = obj
    
    def __enter__(self):
        return self.obj
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MockEvent:
    """Mock for log.Event"""
    
    # Include all services from services.py as union fields
    _service_list = [
        'gyroscope', 'gyroscope2', 'accelerometer', 'accelerometer2', 'magnetometer',
        'lightSensor', 'temperatureSensor', 'temperatureSensor2', 'gpsNMEA', 'deviceState',
        'touch', 'can', 'controlsState', 'selfdriveState', 'pandaStates', 'peripheralState',
        'radarState', 'roadEncodeIdx', 'liveTracks', 'sendcan', 'logMessage', 'errorLogMessage',
        'liveCalibration', 'liveTorqueParameters', 'liveDelay', 'androidLog', 'carState',
        'carControl', 'carOutput', 'longitudinalPlan', 'driverAssistance', 'procLog',
        'gpsLocationExternal', 'gpsLocation', 'ubloxGnss', 'qcomGnss', 'gnssMeasurements',
        'clocks', 'ubloxRaw', 'livePose', 'liveParameters', 'cameraOdometry', 'thumbnail',
        'onroadEvents', 'carParams', 'roadCameraState', 'driverCameraState', 'driverEncodeIdx',
        'driverStateV2', 'driverMonitoringState', 'wideRoadEncodeIdx', 'wideRoadCameraState',
        'drivingModelData', 'modelV2', 'managerState', 'uploaderState', 'navInstruction',
        'navRoute', 'navThumbnail', 'qRoadEncodeIdx', 'userFlag', 'soundPressure', 'rawAudioData',
        'uiDebug', 'testJoystick', 'alertDebug'
    ]
    
    schema = type('MockSchema', (), {
        'union_fields': _service_list
    })()
    
    @classmethod
    def new_message(cls, **kwargs):
        builder = MockDynamicStructBuilder()
        # Set provided kwargs
        for key, value in kwargs.items():
            setattr(builder, key, value)
        return builder
    
    @classmethod
    def from_bytes(cls, data, traversal_limit_in_words=None):
        # Try to deserialize the JSON data back into a MockDynamicStructReader
        try:
            import json
            state = json.loads(data.decode('utf-8'))
            which_name = state.get('_which_name', 'mock')
            reader = MockDynamicStructReader(which_name)
            # Restore the state
            for key, value in state.items():
                if key == '_which_name':
                    reader._which_name = value
                elif not key.startswith('_'):
                    setattr(reader, key, value)
            # Ensure carState points to self
            reader.carState = reader
            return MockContextManager(reader)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fallback to default mock reader
            return MockContextManager(MockDynamicStructReader())


class MockCarState:
    """Mock for car.CarState"""
    
    schema = type('MockSchema', (), {
        'non_union_fields': ['vEgo', 'aEgo', 'brake', 'steeringAngleDeg', 'gas', 'gasPressed', 
                           'brakePressed', 'leftBlinker', 'rightBlinker', 'cruiseState']
    })()


class MockModule:
    """Mock module that can be used for both log and car"""
    
    def __init__(self, name):
        self.name = name
        if 'log' in name:
            self.Event = MockEvent
        elif 'car' in name:
            self.CarState = MockCarState
    
    def __getattr__(self, name):
        # Return appropriate mock based on attribute name
        if name == 'Event':
            return MockEvent
        elif name == 'CarState':
            return MockCarState
        else:
            return type(name, (), {})


class MockCapnpLib:
    """Mock for capnp.lib"""
    
    class MockCapnpClass:
        """Mock for capnp.lib.capnp"""
        
        class KjException(Exception):
            pass
        
        _DynamicStructReader = MockDynamicStructReader
        _DynamicStructBuilder = MockDynamicStructBuilder
        _StructModule = MockModule
    
    capnp = MockCapnpClass()


class MockCapnp:
    """Main mock capnp module"""
    
    def __init__(self):
        self.lib = MockCapnpLib()
        self._DynamicStructReader = MockDynamicStructReader
        self._DynamicStructBuilder = MockDynamicStructBuilder
    
    def remove_import_hook(self):
        """Mock implementation of remove_import_hook"""
        pass
    
    def load(self, file_path):
        """Mock implementation of capnp.load"""
        return MockModule(file_path)


# Mock for the capnp module to avoid import errors during testing
capnp_mock = MockCapnp()
sys.modules["capnp"] = capnp_mock

# Make the classes available at the module level too
sys.modules["capnp"]._DynamicStructReader = MockDynamicStructReader
sys.modules["capnp"]._DynamicStructBuilder = MockDynamicStructBuilder

# Mock for msgq modules that are needed for messaging
# Global message queues to simulate pub/sub
_global_message_queues = {}

class MockSocket:
    """Mock socket for pub/sub operations"""
    
    def __init__(self, service_name=None, timeout=1000, is_publisher=False, conflate=False):
        self.service_name = service_name
        self.timeout = timeout
        self.is_publisher = is_publisher
        self.conflate = conflate
        self._messages = []
        
        # Initialize global queue for this service if it doesn't exist
        if service_name and service_name not in _global_message_queues:
            _global_message_queues[service_name] = []
    
    def send(self, data):
        """Mock send - put data into global queue for this service"""
        if self.service_name and self.service_name in _global_message_queues:
            # Always append messages for drain operations to work correctly
            _global_message_queues[self.service_name].append(data)
    
    def receive(self, non_blocking=False):
        """Mock receive - get data from global queue for this service"""
        if self.service_name and self.service_name in _global_message_queues:
            queue = _global_message_queues[self.service_name]
            if queue:
                if self.conflate and len(queue) > 1:
                    # For conflate sockets, return only the latest message and clear queue
                    latest = queue[-1]
                    _global_message_queues[self.service_name] = []
                    return latest
                else:
                    # For non-conflate sockets, return first message
                    return queue.pop(0)
        
        # If non_blocking and no messages, return None
        if non_blocking:
            return None
        
        # For blocking calls with no messages, simulate timeout by returning None
        # This mimics the behavior when the socket timeout is hit
        return None
    
    def all_readers_updated(self):
        return True


class MockPoller:
    """Mock poller for messaging"""
    
    def __init__(self):
        self._sockets = []
    
    def poll(self, timeout=100):
        """Mock poll - return sockets that have messages"""
        # Simulate timeout by sleeping for the timeout duration if no messages
        import time
        start_time = time.monotonic()
        
        # Check for messages in any of the registered sockets
        ready_sockets = []
        for sock in self._sockets:
            if (hasattr(sock, 'service_name') and 
                sock.service_name in _global_message_queues and 
                _global_message_queues[sock.service_name]):
                ready_sockets.append(sock)
        
        if ready_sockets:
            return ready_sockets
        
        # If no messages, simulate timeout
        time.sleep(timeout / 1000.0)
        return []


class MockContext:
    """Mock context for messaging"""
    pass


class MockMsgq:
    """Mock msgq module"""
    
    context = MockContext()
    
    def drain_sock_raw(self, sock, wait_for_one=False):
        """Mock drain_sock_raw - return list of mock messages"""
        if hasattr(sock, 'service_name') and sock.service_name in _global_message_queues:
            queue = _global_message_queues[sock.service_name]
            if queue:
                # Return all messages and clear the queue
                messages = queue.copy()
                _global_message_queues[sock.service_name] = []
                return messages
        return []
    
    def pub_sock(self, service_name, **kwargs):
        """Mock pub_sock - return mock publisher socket"""
        return MockSocket(service_name)
    
    def sub_sock(self, service_name, **kwargs):
        """Mock sub_sock - return mock subscriber socket"""
        socket = MockSocket(service_name, kwargs.get('timeout', 1000), conflate=kwargs.get('conflate', False))
        # Register socket with poller if provided
        poller = kwargs.get('poller')
        if poller:
            poller._sockets.append(socket)
        return socket
    
    def __getattr__(self, name):
        # Return appropriate mock functions
        if name == "drain_sock_raw":
            return self.drain_sock_raw
        elif name == "pub_sock":
            return self.pub_sock
        elif name == "sub_sock":
            return self.sub_sock
        return lambda *args, **kwargs: None


class MockIpcPyx:
    """Mock for msgq.ipc_pyx"""
    
    Context = MockContext
    Poller = MockPoller
    SubSocket = MockSocket


# Install the mocks
msgq_mock = MockMsgq()
sys.modules["msgq"] = msgq_mock
sys.modules["msgq.ipc_pyx"] = MockIpcPyx()
sys.modules["msgq.fake_event_handle"] = MockMsgq()

# Make sure the functions are available at module level
sys.modules["msgq"].drain_sock_raw = msgq_mock.drain_sock_raw
sys.modules["msgq"].pub_sock = msgq_mock.pub_sock
sys.modules["msgq"].sub_sock = msgq_mock.sub_sock

# Mock cereal.log module
class MockLogModule:
    Event = MockEvent

sys.modules["cereal.log"] = MockLogModule()

# Mock cereal.car module  
class MockCarModule:
    CarState = MockCarState

sys.modules["cereal.car"] = MockCarModule()

# Mock cereal.services module
class MockService:
    def __init__(self, should_log, frequency, decimation=None):
        self.should_log = should_log
        self.frequency = frequency
        self.decimation = decimation

# Mock SERVICE_LIST with common services from services.py
_mock_services = {
    'gyroscope': MockService(True, 104.0, 104),
    'gyroscope2': MockService(True, 100.0, 100),
    'accelerometer': MockService(True, 104.0, 104),
    'accelerometer2': MockService(True, 100.0, 100),
    'magnetometer': MockService(True, 25.0),
    'lightSensor': MockService(True, 100.0, 100),
    'temperatureSensor': MockService(True, 2.0, 200),
    'temperatureSensor2': MockService(True, 2.0, 200),
    'gpsNMEA': MockService(True, 9.0),
    'deviceState': MockService(True, 2.0, 1),
    'touch': MockService(True, 20.0, 1),
    'can': MockService(True, 100.0, 2053),
    'controlsState': MockService(True, 100.0, 10),
    'selfdriveState': MockService(True, 100.0, 10),
    'pandaStates': MockService(True, 10.0, 1),
    'peripheralState': MockService(True, 2.0, 1),
    'radarState': MockService(True, 20.0, 5),
    'roadEncodeIdx': MockService(False, 20.0, 1),
    'liveTracks': MockService(True, 20.0),
    'sendcan': MockService(True, 100.0, 139),
    'logMessage': MockService(True, 0.0),
    'errorLogMessage': MockService(True, 0.0, 1),
    'liveCalibration': MockService(True, 4.0, 4),
    'liveTorqueParameters': MockService(True, 4.0, 1),
    'liveDelay': MockService(True, 4.0, 1),
    'androidLog': MockService(True, 0.0),
    'carState': MockService(True, 100.0, 10),
    'carControl': MockService(True, 100.0, 10),
    'carOutput': MockService(True, 100.0, 10),
    'longitudinalPlan': MockService(True, 20.0, 10),
    'driverAssistance': MockService(True, 20.0, 20),
    'procLog': MockService(True, 0.5, 15),
    'gpsLocationExternal': MockService(True, 10.0, 10),
    'gpsLocation': MockService(True, 1.0, 1),
    'ubloxGnss': MockService(True, 10.0),
    'qcomGnss': MockService(True, 2.0),
    'gnssMeasurements': MockService(True, 10.0, 10),
    'clocks': MockService(True, 0.1, 1),
    'ubloxRaw': MockService(True, 20.0),
    'livePose': MockService(True, 20.0, 4),
    'liveParameters': MockService(True, 20.0, 5),
    'cameraOdometry': MockService(True, 20.0, 10),
    'thumbnail': MockService(True, 1/60.0, 1),
    'onroadEvents': MockService(True, 1.0, 1),
    'carParams': MockService(True, 0.02, 1),
    'roadCameraState': MockService(True, 20.0, 20),
    'driverCameraState': MockService(True, 20.0, 20),
    'driverEncodeIdx': MockService(False, 20.0, 1),
    'driverStateV2': MockService(True, 20.0, 10),
    'driverMonitoringState': MockService(True, 20.0, 10),
    'wideRoadEncodeIdx': MockService(False, 20.0, 1),
    'wideRoadCameraState': MockService(True, 20.0, 20),
    'drivingModelData': MockService(True, 20.0, 10),
    'modelV2': MockService(True, 20.0),
    'managerState': MockService(True, 2.0, 1),
    'uploaderState': MockService(True, 0.0, 1),
    'navInstruction': MockService(True, 1.0, 10),
    'navRoute': MockService(True, 0.0),
    'navThumbnail': MockService(True, 0.0),
    'qRoadEncodeIdx': MockService(False, 20.0),
    'userFlag': MockService(True, 0.0, 1),
    'soundPressure': MockService(True, 10.0, 10),
    'rawAudioData': MockService(False, 20.0),
    'uiDebug': MockService(True, 0.0, 1),
    'testJoystick': MockService(True, 0.0),
    'alertDebug': MockService(True, 20.0, 5),
}

class MockServicesModule:
    SERVICE_LIST = _mock_services
    
    # Use the actual mock_capnp.py file path so it can be executed
    __file__ = __file__  # Points to this mock_capnp.py file
    
    def build_header(self):
        # Mock the build_header function that generates valid C code
        h = ""
        h += "/* THIS IS AN AUTOGENERATED FILE, PLEASE EDIT services.py */\n"
        h += "#ifndef __SERVICES_H\n"
        h += "#define __SERVICES_H\n"
        h += "\n"
        h += "#include <map>\n"
        h += "#include <string>\n"
        h += "\n"
        h += "struct service { std::string name; bool should_log; int frequency; int decimation; };\n"
        h += "static std::map<std::string, service> services = {\n"
        for k, v in self.SERVICE_LIST.items():
            should_log = "true" if v.should_log else "false"
            decimation = -1 if v.decimation is None else v.decimation
            h += '  { "%s", {"%s", %s, %d, %d}},\n' % (k, k, should_log, int(v.frequency), decimation)
        h += "};\n"
        h += "\n"
        h += "#endif\n"
        return h

# Add main execution block to make this file executable
if __name__ == "__main__":
    # When executed as a script, output the header
    mock_services = MockServicesModule()
    print(mock_services.build_header())

sys.modules["cereal.services"] = MockServicesModule()
