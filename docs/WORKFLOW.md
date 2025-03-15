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

## Testing

### Automated Testing

All PRs and commits are automatically checked by GitHub Actions. Check out `.github/workflows/` for what GitHub Actions runs. Any new tests should be added to GitHub Actions.

### Code Style and Linting

Code is automatically checked for style by GitHub Actions as part of the automated tests. You can also run these tests yourself by running `pre-commit run --all`.
