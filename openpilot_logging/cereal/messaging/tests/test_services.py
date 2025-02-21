import os
import tempfile
from typing import Dict
from parameterized import parameterized

import cereal.services as services
from cereal.services import SERVICE_LIST


class TestServices:

  @parameterized.expand(SERVICE_LIST.keys())
  def test_services(self, s):
    service = SERVICE_LIST[s]
    assert service.frequency <= 104
    assert service.decimation != 0

  def test_generated_header(self):
    with tempfile.NamedTemporaryFile(suffix=".h") as f:
      ret = os.system(f"python3 {services.__file__} > {f.name} && clang++ {f.name} -std=c++11")
      assert ret == 0, "generated services header is not valid C"
