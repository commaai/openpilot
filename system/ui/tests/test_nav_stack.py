import pytest
from openpilot.system.ui.lib.application import gui_app


class Widget:
  def __init__(self):
    self.enabled, self.shown, self.hidden = True, False, False

  def set_enabled(self, e): self.enabled = e
  def show_event(self): self.shown = True
  def hide_event(self): self.hidden = True


@pytest.fixture(autouse=True)
def clean_stack():
  gui_app._nav_stack = []
  yield
  gui_app._nav_stack = []


def test_push():
  a, b = Widget(), Widget()
  gui_app.push_widget(a)
  gui_app.push_widget(b)
  assert not a.enabled and not a.hidden
  assert b.enabled and b.shown


def test_pop_re_enables():
  widgets = [Widget() for _ in range(4)]
  for w in widgets:
    gui_app.push_widget(w)
  assert all(not w.enabled for w in widgets[:-1])
  gui_app.pop_widget()
  assert widgets[-2].enabled


@pytest.mark.parametrize("pop_fn", [gui_app.pop_widgets_to, gui_app.request_pop_widgets_to])
def test_pop_widgets_to(pop_fn):
  widgets = [Widget() for _ in range(4)]
  for w in widgets:
    gui_app.push_widget(w)

  root = widgets[0]
  pop_fn(root)

  assert gui_app._nav_stack == [root]
  assert root.enabled and not root.hidden
  for w in widgets[1:]:
    assert w.enabled and w.hidden and w.shown
