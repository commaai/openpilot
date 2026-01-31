import os
import pytest
import msgq

@pytest.fixture(params=[False, True], ids=["msgq", "zmq"], autouse=True)
def zmq_mode(request):
  if request.param:
    os.environ["ZMQ"] = "1"
  else:
    os.environ.pop("ZMQ", None)
  msgq.context = msgq.Context()
  assert msgq.context_is_zmq() == request.param
  yield request.param
  os.environ.pop("ZMQ", None)
