import io
import os
import time
import base64
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


def report(args, logs, fp):
  output_path = Path(__file__).resolve().parent / "longitudinal_reports"
  output_fn = args.output or output_path / f"{fp}_{time.strftime('%Y%m%d-%H_%M_%S')}.html"
  output_path.mkdir(exist_ok=True)
  with open(output_fn, "w") as f:
    f.write("<h1>Longitudinal maneuver report</h1>\n")
    f.write(f"<h3>{fp}</h3>\n")
    if args.desc:
      f.write(f"<h3>{args.desc}</h3>")
    for description, runs in logs.items():
      f.write("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
      f.write(f"<h2>{description}</h2>\n")
      for run, log in runs.items():
        f.write(f"<h3>Run #{int(run)+1}</h3>\n")
        plt.rcParams['font.size'] = 40
        fig = plt.figure(figsize=(30, 25))
        ax = fig.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [5, 3, 1, 1]})

        ax[0].grid(linewidth=4)
        ax[0].plot(log["t"], log["carControl.actuators.accel"], label='accel command', linewidth=6)
        ax[0].plot(log["t"], log["carState.aEgo"], label='aEgo', linewidth=6)
        ax[0].set_ylabel('Acceleration (m/s^2)')
        #ax[0].set_ylim(-6.5, 6.5)
        ax[0].legend()

        ax[1].grid(linewidth=4)
        ax[1].plot(log["t"], log["carState.vEgo"], 'g', label='vEgo', linewidth=6)
        ax[1].set_ylabel('Velocity (m/s)')
        ax[1].legend()

        ax[2].plot(log["t"], log["carControl.enabled"], label='enabled', linewidth=6)
        ax[3].plot(log["t"], log["carState.gasPressed"], label='gasPressed', linewidth=6)
        ax[3].plot(log["t"], log["carState.brakePressed"], label='brakePressed', linewidth=6)
        for i in (2, 3):
          ax[i].set_yticks([0, 1], minor=False)
          ax[i].set_ylim(-1, 2)
          ax[i].legend()

        ax[-1].set_xlabel("Time (s)")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        f.write(f"<img src='data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}' style='width:100%; max-width:800px;'>\n")

    import json
    f.write(f"<p style='display: none'>{json.dumps(logs)}</p>")
  print(f"\nReport written to {output_fn}\n")
