from openpilot.common.markdown import parse_markdown
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.html_render import HtmlRenderer


class TestHtmlRenderer(OpenpilotTestCase):
  def test_character_references(self, mocker):
    mocker.patch.object(gui_app, "font")
    renderer = HtmlRenderer(text=parse_markdown('* Described in our CVPR paper: "Learning to Drive from a World Model"\n* Honda & Acura'))
    assert [e.content for e in renderer.elements] == ['• Described in our CVPR paper: "Learning to Drive from a World Model"', '• Honda & Acura']
