"""Convert jotpluggler layout JSON into Rerun blueprints."""

from __future__ import annotations

import json
from pathlib import Path

import rerun.blueprint as rrb

from openpilot.common.basedir import BASEDIR

LAYOUT_DIR = Path(BASEDIR) / "openpilot" / "tools" / "jotpluggler" / "layouts"

CAMERA_ORIGINS = {
  "road": "camera/road",
  "driver": "camera/driver",
  "wide_road": "camera/wide_road",
  "qroad": "camera/q_road",
  "q_road": "camera/q_road",
}


def _hex_to_rgb(color: str) -> list[int]:
  color = color.lstrip("#")
  if len(color) != 6:
    return [160, 170, 180]
  return [int(color[i:i + 2], 16) for i in (0, 2, 4)]


def _curve_entity(path: str) -> str:
  if path.startswith("/custom/"):
    return path.lstrip("/")
  return path.lstrip("/")


def _plot_view_from_pane(pane: dict) -> rrb.TimeSeriesView:
  curves = pane.get("curves", [])
  origins = sorted({_curve_entity(c["name"]) for c in curves if c.get("name")})
  origin = origins[0].rsplit("/", 1)[0] if origins else "/"
  if origin == "":
    origin = "/"

  axis_y = None
  y_limits = pane.get("y_limits")
  range_data = pane.get("range", {})
  if y_limits:
    axis_y = rrb.ScalarAxis(range=(y_limits["min"], y_limits["max"]))
  elif "top" in range_data and "bottom" in range_data:
    axis_y = rrb.ScalarAxis(range=(range_data["bottom"], range_data["top"]))

  axis_x = None
  if "left" in range_data and "right" in range_data:
    import rerun as rr
    axis_x = rrb.TimeAxis(view_range=rr.TimeRange(
      start=rrb.TimeRangeBoundary.absolute(seconds=range_data["left"]),
      end=rrb.TimeRangeBoundary.absolute(seconds=range_data["right"]),
    ))

  contents = rrb.TimeSeriesView(
    name=pane.get("title") or "Plot",
    origin=origin,
    axis_y=axis_y,
    axis_x=axis_x,
    plot_legend=rrb.PlotLegend(visible=True),
  )
  return contents


def _pane_to_view(pane: dict):
  kind = pane.get("kind", "plot")
  if kind == "map":
    return rrb.MapView(name=pane.get("title") or "Map", origin="map")
  if kind == "camera":
    view = pane.get("camera_view", "road")
    return rrb.Spatial2DView(name=pane.get("title") or "Camera", origin=CAMERA_ORIGINS.get(view, f"camera/{view}"))
  return _plot_view_from_pane(pane)


def _node_to_container(node: dict):
  children = node.get("children", [])
  if not children:
    pane = node if node.get("curves") is not None or node.get("kind") in {"map", "camera"} else None
    if pane is None:
      return rrb.TextLogView(name="Logs", origin="logs")
    return _pane_to_view(pane)

  views = [_node_to_container(child) for child in children]
  split = node.get("split", "vertical")
  if split == "horizontal":
    return rrb.Horizontal(contents=views, name="split")
  return rrb.Vertical(contents=views, name="split")


def layout_to_blueprint(layout: dict) -> rrb.Blueprint:
  tabs = layout.get("tabs", [])
  if not tabs:
    return rrb.Blueprint(rrb.TimeSeriesView(origin="carState"), collapse_panels=True)

  tab_views = []
  for tab in tabs:
    root = tab.get("root", {})
    tab_views.append(_node_to_container(root))

  if len(tab_views) == 1:
    body = tab_views[0]
  else:
    body = rrb.Tabs(contents=tab_views)

  return rrb.Blueprint(
    rrb.Vertical(
      contents=[
        body,
        rrb.TextLogView(name="Logs", origin="logs"),
      ],
    ),
    collapse_panels=True,
  )


def load_layout(layout_name: str) -> dict:
  path = LAYOUT_DIR / f"{layout_name}.json"
  if not path.exists():
    raise FileNotFoundError(f"layout not found: {path}")
  return json.loads(path.read_text())


def make_blueprint(layout_name: str | None) -> rrb.Blueprint:
  if not layout_name:
    return rrb.Blueprint(
      rrb.Tabs(
        rrb.Vertical(
          rrb.TimeSeriesView(name="Vehicle", origin="carState"),
          rrb.MapView(name="Map", origin="map"),
        ),
        rrb.Vertical(
          rrb.Spatial2DView(name="Road", origin="camera/road"),
          rrb.Spatial2DView(name="Driver", origin="camera/driver"),
        ),
      ),
      collapse_panels=True,
    )
  layout = load_layout(layout_name)
  return layout_to_blueprint(layout)