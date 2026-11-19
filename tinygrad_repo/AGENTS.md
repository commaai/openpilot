# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.
