#!/usr/bin/env python3
"""
NGVT Braid Offline Visualizer
==============================
Reads JSON output from tools/ngvt_analysis.py and renders 3D torus
manifold plots of instability zones and lead tracking.

OFFLINE ONLY — reads static JSON files, no live vehicle data.

Usage:
  # Generate data first:
  python tools/ngvt_analysis.py /path/to/rlog.bz2 --out ngvt_results.json

  # Then visualize:
  python tools/ngvt_visualizer.py ngvt_results.json
  python tools/ngvt_visualizer.py ngvt_results.json --show-all-leads
  python tools/ngvt_visualizer.py ngvt_results.json --save plot.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# ---------------------------------------------------------------------------
# Torus mesh helpers
# ---------------------------------------------------------------------------

def torus_mesh(R: float = 10.0, r: float = 3.0, n: int = 60):
  """Return (X, Y, Z) arrays for the torus wireframe."""
  u = np.linspace(0, 2 * np.pi, n)
  v = np.linspace(0, 2 * np.pi, n)
  U, V = np.meshgrid(u, v)
  X = (R + r * np.cos(V)) * np.cos(U)
  Y = (R + r * np.cos(V)) * np.sin(U)
  Z = r * np.sin(V)
  return X, Y, Z


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_instability_zones(
  frames: list[dict],
  R: float = 10.0,
  r: float = 3.0,
  save_path: str | None = None,
) -> None:
  """
  Plot all failure-zone manifold points (nodes the engine flagged as
  unstable, cached for Braid lookback amplification on the next frame).
  """
  zone_points = []
  for frame in frames:
    for coords in frame.get("active_failure_zones", []):
      if len(coords) == 3:
        zone_points.append(coords)

  fig = plt.figure(figsize=(11, 8))
  ax = fig.add_subplot(111, projection="3d")

  # Translucent torus wireframe for context
  TX, TY, TZ = torus_mesh(R, r, n=40)
  ax.plot_surface(TX, TY, TZ, alpha=0.07, color="steelblue", linewidth=0)

  if zone_points:
    pts = np.array(zone_points)
    sc = ax.scatter(
      pts[:, 0], pts[:, 1], pts[:, 2],
      c=np.linalg.norm(pts, axis=1),
      cmap="plasma", s=30, alpha=0.85,
      label=f"Instability zones ({len(pts)})",
    )
    plt.colorbar(sc, ax=ax, label="Distance from manifold origin (au)")
  else:
    ax.text(0, 0, 0, "No instability zones found", ha="center", va="center")

  ax.scatter(0, 0, 0, color="red", s=120, marker="^", zorder=5, label="Origin (ego)")
  ax.set_title("NGVT Manifold — Spatial Instability Zones")
  ax.set_xlabel("Manifold X")
  ax.set_ylabel("Manifold Y")
  ax.set_zlabel("Manifold Z")
  ax.legend(loc="upper left")

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
  else:
    plt.show()


def plot_all_leads(
  frames: list[dict],
  R: float = 10.0,
  r: float = 3.0,
  save_path: str | None = None,
) -> None:
  """
  Plot every analyzed lead node, colored by adjusted Braid score.
  Unstable nodes are rendered larger and with a distinct marker.
  """
  stable_pts, stable_scores = [], []
  unstable_pts = []

  for frame in frames:
    for node in frame.get("nodes", []):
      coords = node["torus_coords"]
      if len(coords) != 3:
        continue
      if node["flagged_unstable"]:
        unstable_pts.append(coords)
      else:
        stable_pts.append(coords)
        stable_scores.append(node["adjusted_score"])

  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(111, projection="3d")

  TX, TY, TZ = torus_mesh(R, r, n=40)
  ax.plot_surface(TX, TY, TZ, alpha=0.07, color="steelblue", linewidth=0)

  if stable_pts:
    spt = np.array(stable_pts)
    sc = ax.scatter(spt[:, 0], spt[:, 1], spt[:, 2],
                    c=stable_scores, cmap="viridis", s=15, alpha=0.6,
                    label=f"Stable leads ({len(stable_pts)})")
    plt.colorbar(sc, ax=ax, label="Braid-adjusted score")

  if unstable_pts:
    upt = np.array(unstable_pts)
    ax.scatter(upt[:, 0], upt[:, 1], upt[:, 2],
               color="red", s=60, marker="x", alpha=0.9,
               label=f"Unstable (flagged) ({len(unstable_pts)})")

  ax.scatter(0, 0, 0, color="gold", s=120, marker="^", zorder=5, label="Origin (ego)")
  ax.set_title("NGVT Manifold — All Analyzed Lead Nodes")
  ax.set_xlabel("Manifold X")
  ax.set_ylabel("Manifold Y")
  ax.set_zlabel("Manifold Z")
  ax.legend(loc="upper left")

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
  else:
    plt.show()


def print_stats(frames: list[dict]) -> None:
  total = sum(len(f["nodes"]) for f in frames)
  unstable = sum(sum(1 for n in f["nodes"] if n["flagged_unstable"]) for f in frames)
  print(f"Frames: {len(frames)}  |  Leads: {total}  |  Unstable: {unstable} ({100*unstable/max(total,1):.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
  p = argparse.ArgumentParser(description="NGVT Braid offline 3D visualizer.")
  p.add_argument("results_json", help="JSON file written by tools/ngvt_analysis.py")
  p.add_argument("--show-all-leads", action="store_true", help="Also plot all lead nodes (not just failure zones)")
  p.add_argument("--save", help="Save plot to file instead of showing interactively")
  p.add_argument("--major-radius", type=float, default=10.0)
  p.add_argument("--minor-radius", type=float, default=3.0)
  args = p.parse_args()

  data = json.loads(Path(args.results_json).read_text())
  print_stats(data)

  if args.show_all_leads:
    plot_all_leads(data, args.major_radius, args.minor_radius, args.save)
  else:
    plot_instability_zones(data, args.major_radius, args.minor_radius, args.save)


if __name__ == "__main__":
  main()
