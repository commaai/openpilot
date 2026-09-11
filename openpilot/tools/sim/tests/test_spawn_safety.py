import multiprocessing
import unittest

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.lib.common import SIM_MP_CTX, World


def _touch(bridge_state, lock, event, arr, conn):
  """Runs in a spawned child: proves every shared object survived the trip."""
  lock.release()
  event.set()
  arr[0] = 7
  conn.send(bridge_state)


class TestSpawnSafety(unittest.TestCase):
  def test_context_is_spawn(self):
    # a panda3d/OpenGL context can't be inherited across fork()
    assert SIM_MP_CTX.get_start_method() == "spawn"

  def test_world_primitives_come_from_the_sim_context(self):
    # mixing contexts for synchronization primitives is unsupported and fails at spawn time
    ctx_types = SIM_MP_CTX.Semaphore(0).__class__, SIM_MP_CTX.Event().__class__
    world_lock = World.__init__.__globals__["SIM_MP_CTX"].Semaphore(0)
    assert isinstance(world_lock, ctx_types[0])
    assert SIM_MP_CTX is not multiprocessing

  def test_shared_objects_survive_a_spawn(self):
    import ctypes
    lock = SIM_MP_CTX.Semaphore(value=0)
    event = SIM_MP_CTX.Event()
    arr = SIM_MP_CTX.Array(ctypes.c_uint8, 4)
    parent_conn, child_conn = SIM_MP_CTX.Pipe()

    p = SIM_MP_CTX.Process(target=_touch, args=("hello", lock, event, arr, child_conn))
    p.start()
    try:
      assert parent_conn.recv() == "hello"
      assert lock.acquire(timeout=10)
      assert event.wait(timeout=10)
      assert arr[0] == 7
    finally:
      p.join(timeout=10)
      if p.is_alive():
        p.kill()

  def test_bridge_is_picklable(self):
    # the bridge is handed to the spawned process, so the objects it owns must be dropped
    import pickle

    bridge = _StubBridge()
    state = bridge.__getstate__()
    assert "_exit_event" not in state and "_threads" not in state and "world" not in state
    pickle.dumps(state)

    revived = _StubBridge()
    revived.__setstate__(state)
    assert revived._threads == [] and revived._exit_event is None and revived.world is None
    assert revived.dual_camera is False


class _StubBridge(SimulatorBridge):
  """Skips SimulatorBridge.__init__ so this doesn't need a MetaDrive install or Params."""
  def __init__(self):  # deliberately not calling super(): no Params or MetaDrive needed here
    import threading
    self.dual_camera = False
    self.high_quality = False
    self._exit_event = threading.Event()
    self._threads = [object()]
    self.world = None
    self._keep_alive = True

  def spawn_world(self, q, /):
    raise NotImplementedError
