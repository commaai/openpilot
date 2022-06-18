#!/usr/bin/env python3
import os
import filecmp
import tempfile
from opendbc.generator.generator import create_all, opendbc_root


def test_generator():
  with tempfile.TemporaryDirectory() as d:
    create_all(d)

    ignore = [f for f in os.listdir(opendbc_root) if not f.endswith('_generated.dbc')]
    comp = filecmp.dircmp(opendbc_root, d, ignore=ignore)
    assert len(comp.diff_files) == 0, f"Different files: {comp.diff_files}"


if __name__ == "__main__":
  test_generator()
