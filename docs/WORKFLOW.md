# the openpilot development workflow

Aside from the ML models, most tools used for openpilot development are in this repo.

## Quick start

```
# git stuff
git pull

# update dependencies
tools/ubuntu_setup.sh

# build everything
cd ~/openpilot/
scons -j$(nproc)

# test everything
pytest .

# build just the ui with either of these
scons -j8 selfdrive/ui/
cd selfdrive/ui/ && scons -u -j8

# test just logging services
cd system/loggerd && pytest .
```


## Testing

### Automated Testing

All PRs and commits are automatically checked by GitHub Actions. Check out `.github/workflows/` for what GitHub Actions runs. Any new tests should be added to GitHub Actions.

### Code Style and Linting

Code is automatically checked for style by GitHub Actions as part of the automated tests. You can also run these tests yourself by running `pre-commit run --all`.
