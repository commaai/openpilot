import pytest
import msgq

@pytest.fixture(autouse=True)
def msgq_context():
  msgq.context = msgq.Context()
