#!/usr/bin/env python3
"""
NGVT Braid Offline Log Analyzer
================================
Applies NGVT manifold projection and Compounding Braid attention analysis
to recorded openpilot log files (.rlog / .bz2).

OFFLINE ONLY — reads existing log files, writes no messages, touches no
control path. Safe to run on any openpilot log segment.

Usage:
  # Single segment
  python tools/ngvt_analysis.py /path/to/segment/rlog.bz2

  # Route (all segments)
  python tools/ngvt_analysis.py --route <route_id>

  # Save JSON output for the visualizer
  python tools/ngvt_analysis.py /path/to/rlog.bz2 --out ngvt_results.json

Requirements (install in your openpilot venv):
  pip install maturin
  maturin develop --manifest-path selfdrive/controls/lib/ngvt_braid/Cargo.toml
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Ensure the tools/ directory is on the path so backend modules resolve
_TOOLS_DIR = str(Path(__file__).parent)
if _TOOLS_DIR not in sys.path:
  sys.path.insert(0, _TOOLS_DIR)

# openpilot log reading utilities — present in any openpilot checkout
try:
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.tools.lib.route import Route
except ImportError:
  print("ERROR: Run this script from within an openpilot checkout with its venv active.")
  sys.exit(1)

# Compiled Rust extension — build with:
#   maturin develop --manifest-path selfdrive/controls/lib/ngvt_braid/Cargo.toml
# Falls back to the pure-Python tinygrad or PyTorch implementation if not built.
try:
  from ngvt_braid import NgvtBraidEngine
  _BACKEND = "rust"
except ImportError:
  try:
    from ngvt_braid_tinygrad import NgvtBraidEngineTinygrad as NgvtBraidEngine  # type: ignore[assignment]
    _BACKEND = "tinygrad"
  except ImportError:
    try:
      from ngvt_braid_torch import NgvtBraidEngineTorch as NgvtBraidEngine  # type: ignore[assignment]
      _BACKEND = "torch"
    except ImportError:
      print(
        "ERROR: No NgvtBraidEngine backend available.\n"
        + "  Build the Rust crate:  maturin develop --manifest-path selfdrive/controls/lib/ngvt_braid/Cargo.toml\n"
        + "  Or install tinygrad:   pip install tinygrad\n"
        + "  Or install PyTorch:    pip install torch"
      )
      sys.exit(1)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AnalyzedNode:
  frame_id: int
  lead_index: int
  raw_x: float          # longitudinal distance from modelV2 (meters)
  raw_y: float          # lateral offset from modelV2 (meters)
  raw_prob: float       # existence probability [0, 1] from modelV2
  torus_coords: list[float]   # [X, Y, Z] on the NGVT manifold
  adjusted_score: float       # Braid-boosted score, clamped to [0, 1]
  flagged_unstable: bool      # True if this node was added to the failure cache

@dataclass
class FrameResult:
  frame_id: int
  log_mono_time: int
  nodes: list[AnalyzedNode] = field(default_factory=list)
  active_failure_zones: list[list[float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis loop
# ---------------------------------------------------------------------------

# A lead is flagged as unstable when its Braid-adjusted score is below this
# threshold AND the vehicle is within this longitudinal distance (meters).
INSTABILITY_SCORE_THRESHOLD = 0.4
INSTABILITY_DISTANCE_THRESHOLD = 40.0   # meters


def analyze_log(log_path: str, engine: NgvtBraidEngine) -> list[FrameResult]:
  """
  Iterate over every modelV2 message in *log_path* and apply the NGVT Braid
  analysis to each detected lead.  Returns one FrameResult per frame.
  """
  results: list[FrameResult] = []
  frame_id = 0

  lr = LogReader(log_path)
  # lr.filter() returns the already-unwrapped struct (no .which() needed)
  for model in lr.filter("modelV2"):
    frame_result = FrameResult(frame_id=frame_id, log_mono_time=0)
    unstable_this_frame: list[list[float]] = []

    # leadsV3 is current (LeadDataV3): up to 3 leads at different time horizons.
    # x[0] = forward distance (m), y[0] = lateral offset (m), prob = existence probability.
    for idx, lead in enumerate(model.leadsV3):
      raw_x = float(lead.x[0]) if len(lead.x) > 0 else 0.0
      raw_y = float(lead.y[0]) if len(lead.y) > 0 else 0.0
      raw_prob = float(lead.prob)

      torus_coords, adjusted_score = engine.process_node(raw_x, raw_y, raw_prob)

      unstable = (
        adjusted_score < INSTABILITY_SCORE_THRESHOLD
        and raw_x < INSTABILITY_DISTANCE_THRESHOLD
      )
      if unstable:
        unstable_this_frame.append(torus_coords)

      frame_result.nodes.append(AnalyzedNode(
        frame_id=frame_id,
        lead_index=idx,
        raw_x=raw_x,
        raw_y=raw_y,
        raw_prob=raw_prob,
        torus_coords=torus_coords,
        adjusted_score=adjusted_score,
        flagged_unstable=unstable,
      ))

    engine.register_verification_results(unstable_this_frame)
    frame_result.active_failure_zones = engine.get_active_failure_zones()
    results.append(frame_result)
    frame_id += 1

  return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results: list[FrameResult]) -> None:
  total_nodes = sum(len(f.nodes) for f in results)
  unstable_nodes = sum(
    sum(1 for n in f.nodes if n.flagged_unstable) for f in results
  )
  frames_with_zones = sum(1 for f in results if f.active_failure_zones)

  print(f"\n{'='*60}")
  print("  NGVT Braid Analysis Summary")
  print(f"{'='*60}")
  print(f"  Frames analyzed       : {len(results)}")
  print(f"  Total lead nodes      : {total_nodes}")
  print(f"  Unstable nodes flagged: {unstable_nodes} ({100*unstable_nodes/max(total_nodes,1):.1f}%)")
  print("  Frames with active")
  print(f"    failure zones       : {frames_with_zones}")
  print(f"{'='*60}\n")

  if unstable_nodes > 0:
    print("  Sample unstable nodes (first 5):")
    shown = 0
    for frame in results:
      for node in frame.nodes:
        if node.flagged_unstable:
          print(
            f"    frame={node.frame_id} lead={node.lead_index} "
            + f"x={node.raw_x:.1f}m prob={node.raw_prob:.3f} "
            + f"→ score={node.adjusted_score:.3f} "
            + f"torus={[round(c,2) for c in node.torus_coords]}"
          )
          shown += 1
          if shown >= 5:
            break
      if shown >= 5:
        break


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Offline NGVT Braid analysis of openpilot log files."
  )
  p.add_argument("log", nargs="?", help="Path to an rlog or rlog.bz2 file")
  p.add_argument("--route", help="Route ID (e.g. a2a0ccea32023010|2019-12-19--15-32-17)")
  p.add_argument("--out", help="Write JSON results to this file")
  p.add_argument("--major-radius", type=float, default=10.0)
  p.add_argument("--minor-radius", type=float, default=3.0)
  p.add_argument("--boost-factor", type=float, default=3.0)
  return p


def main() -> None:
  args = build_arg_parser().parse_args()
  engine = NgvtBraidEngine(  # type: ignore[call-arg]
    major_radius=args.major_radius,
    minor_radius=args.minor_radius,
    boost_factor=args.boost_factor,
  )
  print(f"Backend: {_BACKEND}")

  log_paths: list[str] = []

  if args.route:
    route = Route(args.route)
    log_paths = [str(p) for p in route.log_paths() if p]
  elif args.log:
    log_paths = [args.log]
  else:
    print("Provide a log path or --route. Use --help for usage.")
    sys.exit(1)

  all_results: list[FrameResult] = []
  for lp in log_paths:
    print(f"Analyzing: {lp}")
    all_results.extend(analyze_log(lp, engine))

  print_summary(all_results)

  if args.out:
    out_data = [asdict(r) for r in all_results]
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
  main()
