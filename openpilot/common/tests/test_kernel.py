import os
from unittest import mock

from openpilot.common import kernel
from openpilot.common.test import OpenpilotTestCase


class TestKernel(OpenpilotTestCase):
  def setUp(self):
    kernel.upstream_kernel.cache_clear()
    self.addCleanup(kernel.upstream_kernel.cache_clear)

  def _upstream(self, agnos: bool, release: str) -> bool:
    kernel.upstream_kernel.cache_clear()
    uname = os.uname_result(("Linux", "comma", release, "#1", "aarch64"))
    with mock.patch("os.path.isfile", side_effect=lambda p: agnos and p == "/AGNOS"), mock.patch("os.uname", return_value=uname):
      return kernel.upstream_kernel()

  def test_upstream_kernel(self):
    assert self._upstream(True, "7.3.1")
    # the downstream kernel
    assert not self._upstream(True, "4.9.103")
    # a PC, whatever its kernel
    assert not self._upstream(False, "6.8.0-139-generic")

  def test_agrees_with_visionipc(self):
    # msgq picks DMA-BUF heaps exactly when this is the upstream kernel
    from msgq.visionipc import visionbuf_backend_name
    assert (visionbuf_backend_name() == "dma_heap") == kernel.upstream_kernel()
