# How to contribute

Our software is open source so you can solve your own problems without needing help from others. And if you solve a problem and are so kind, you can upstream it for the rest of the world to use. Check out our [post about externalization](https://blog.comma.ai/a-2020-theme-externalization/).

Most open source development activity is coordinated through our [GitHub Discussions](https://github.com/commaai/openpilot/discussions) and [Discord](https://discord.comma.ai). A lot of documentation is available at https://docs.comma.ai and on our [blog](https://blog.comma.ai/).

### Getting Started

 * Setup your [development environment](../tools/)
 * Join our [Discord](https://discord.comma.ai)
 * Make sure you have a [GitHub account](https://github.com/signup/free)
 * Fork [our repositories](https://github.com/commaai) on GitHub

### What contributions are we looking for?

openpilot's priorities are safety (as defined by [SAFETY.md](docs/SAFETY.md)), stability, quality, and features, in that order.

Simple, well-tested bug fixes are the easiest to merge.
Features are the hardest to get merged.

The probability of a pull request being merged is a fucntion of its value to the project and the time it will take us to get it merged.
If a PR offers *some* value but will take lots of time to get merged, it will be closed.

### What doesn't get merged?

* style changes
* new controllers
* 500+ line PRs: it should be either cleaned up, broken up into smaller PRs, or both
* new features

### First contribution

Tackle an open [good first issue](https://github.com/commaai/openpilot/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.

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

