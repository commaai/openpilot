"""Shared route timeline for all Rerun logging."""

ROUTE_TIMELINE = "route_time"


def set_route_time(rr, seconds: float) -> None:
  rr.set_time(ROUTE_TIMELINE, duration=float(seconds))