from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.system.manager.process_config import managed_processes


class TestFeedbackd(OpenpilotTestCase):
  def setup_method(self):
    self.pm = messaging.PubMaster(['bookmarkButton'])
    self.sm = messaging.SubMaster(['userBookmark'])

  def test_bookmark_button(self):
    managed_processes["feedbackd"].start()
    assert self.pm.wait_for_readers_to_update('bookmarkButton', timeout=5)

    msg = messaging.new_message('bookmarkButton')
    self.pm.send('bookmarkButton', msg)
    self.sm.update(timeout=1000)

    assert self.sm.updated['userBookmark'], "userBookmark should be published on bookmarkButton press"

    managed_processes["feedbackd"].stop()
