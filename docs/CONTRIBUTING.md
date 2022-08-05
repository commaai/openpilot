# How to contribute

Our software is open source so you can solve your own problems without needing help from others. And if you solve a problem and are so kind, you can upstream it for the rest of the world to use. Check out our [post about externalization](https://blog.comma.ai/a-2020-theme-externalization/).

Most open source development activity is coordinated through our [GitHub Discussions](https://github.com/commaai/openpilot/discussions) and [Discord](https://discord.comma.ai). A lot of documentation is available at https://docs.comma.ai and on our [blog](https://blog.comma.ai/).

### Getting Started

 * Setup your [development environment](../tools/)
 * Join our [Discord](https://discord.comma.ai)
 * Make sure you have a [GitHub account](https://github.com/signup/free)
 * Fork [our repositories](https://github.com/commaai) on GitHub

### First contribution
Try out some of these first pull requests ideas to dive into the codebase:

* Increase our [mypy](http://mypy-lang.org/) coverage
* Write some documentation
* Tackle an open [good first issue](https://github.com/commaai/openpilot/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

## Pull Requests

Pull requests should be against the master branch. Welcomed contributions include bug reports, car ports, and any [open issue](https://github.com/commaai/openpilot/issues). If you're unsure about a contribution, feel free to open a discussion, issue, or draft PR to discuss the problem you're trying to solve.

A good pull request has all of the following:
* a clearly stated purpose
* every line changed directly contributes to the stated purpose
* verification, i.e. how did you test your PR?
* justification
  * if you've optimized something, post benchmarks to prove it's better
  * if you've improved your car's tuning, post before and after plots
* passes the CI tests

### Car Ports

We've released a [Model Port guide](https://blog.comma.ai/openpilot-port-guide-for-toyota-models/) for porting to Toyota/Lexus models.

If you port openpilot to a substantially new car brand, see this more generic [Brand Port guide](https://blog.comma.ai/how-to-write-a-car-port-for-openpilot/).

## Testing

### Automated Testing

All PRs and commits are automatically checked by GitHub Actions. Check out `.github/workflows/` for what GitHub Actions runs. Any new tests should be added to GitHub Actions.

### Code Style and Linting

Code is automatically checked for style by GitHub Actions as part of the automated tests. You can also run these tests yourself by running `pre-commit run --all`.
