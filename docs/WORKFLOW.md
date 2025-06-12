# openpilot development workflow

Aside from the ML models, most tools used for openpilot development are in this repo.

Most development happens on normal Ubuntu workstations, and not in cars or directly on comma devices. See the [setup guide](../tools) for getting your PC setup for openpilot development.

## Quick start

```bash
# get the latest stuff
git pull
git lfs pull
git submodule update --init --recursive

# update dependencies
tools/ubuntu_setup.sh

# build everything
scons -j$(nproc)

# build just the ui with either of these
scons -j8 selfdrive/ui/
cd selfdrive/ui/ && scons -u -j8

# test everything
pytest

# test just logging services
cd system/loggerd && pytest .

# run the linter
op lint
```
