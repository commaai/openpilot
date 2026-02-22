from openpilot.system.ui.widgets import Widget


class Layout(Widget):
  """
  A Widget that lays out child Widgets.
  """

  def __init__(self):
    super().__init__()
    self._children: list[Widget] = []
