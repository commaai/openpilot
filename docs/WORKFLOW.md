# openpilot development workflow

Aside from the ML models, most tools used for openpilot development are in this repo.

Most development happens on normal Ubuntu workstations, and not in cars or directly on comma devices.

## Quick start

**1. Setup your machine**

Follow the [setup guide](../tools/README.md) for getting your PC setup for openpilot development.

**2. Start using openpilot**
```bash
# build everything
scons -j$(nproc)

# build just the ui with either of these
scons -j$(nproc) selfdrive/ui/
cd selfdrive/ui/ && scons -u -j$(nproc)

# test everything
pytest .

# test just logging services
cd system/loggerd && pytest .

# run the linter
pre-commit run --all
```

## Testing

### Automated Testing

All PRs and commits are automatically checked by GitHub Actions. Check out `.github/workflows/` for what GitHub Actions runs. Any new tests should be added to GitHub Actions.

### Code Style and Linting

Code is automatically checked for style by GitHub Actions as part of the automated tests. You can also run these tests yourself by running `pre-commit run --all`.
