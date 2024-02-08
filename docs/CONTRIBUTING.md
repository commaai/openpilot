# How to contribute

Our software is open source so you can solve your own problems without needing help from others. And if you solve a problem and are so kind, you can upstream it for the rest of the world to use. Check out our [post about externalization](https://blog.comma.ai/a-2020-theme-externalization/). Development activity is coordinated through our GitHub Issues, [GitHub Discussions](https://github.com/commaai/openpilot/discussions), and [Discord](https://discord.comma.ai).

### Getting Started

* Setup your [development environment](../tools/)
* Read about the [development workflow](WORKFLOW.md)
* Join our [Discord](https://discord.comma.ai)
* Docs are at https://docs.comma.ai and https://blog.comma.ai

## What contributions are we looking for?

**openpilot's priorities are [safety](SAFETY.md), stability, quality, and features, in that order.** openpilot is part of comma's mission to *solve self-driving cars while delivering shippable intermediaries*, and **all** development is towards that goal. 

### What gets merged?

The probability of a pull request being merged is a function of its value to the project and the effort it will take us to get it merged.
If a PR offers *some* value but will take lots of time to get merged, it will be closed.
Simple, well-tested bug fixes are the easiest to merge, and new features are the hardest to get merged. 

All of these are examples of good PRs:
* typo fix: https://github.com/commaai/openpilot/pull/30678
* removing unused code: https://github.com/commaai/openpilot/pull/30573
* simple car model port: https://github.com/commaai/openpilot/pull/30245
* car brand port: https://github.com/commaai/openpilot/pull/23331

### What doesn't get merged?

* **arbitrary style changes**: code is art, and it's up to the author to make it beautiful
* **500+ line PRs**: clean it up, break it up into smaller PRs, or both
* **PRs without a clear goal**: every PR must have a singular and clear goal
* **UI design changes**: we do not have a good review process for this yet
* **New features**: We believe openpilot is mostly feature-complete, and the rest is a matter of refinement and fixing bugs. As a result of this, most feature PRs will be immediately closed, however the beauty of open source is that forks can and do offer features that upstream openpilot doesn't.

### First contribution

Check out any [good first issue](https://github.com/commaai/openpilot/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.

### What do I need to contribute?

A lot of openpilot work requires only a PC, and some requires a comma device.
Most car-related contributions require access to that car, plus a comma device installed in the car.

## Pull Requests

Pull requests should be against the master branch. If you're unsure about a contribution, feel free to open a discussion, issue, or draft PR to discuss the problem you're trying to solve.

A good pull request has all of the following:
* a clearly stated purpose
* every line changed directly contributes to the stated purpose
* verification, i.e. how did you test your PR?
* justification
  * if you've optimized something, post benchmarks to prove it's better
  * if you've improved your car's tuning, post before and after plots
* passes the CI tests

## Contributing without Code

* Report bugs in GitHub issues.
* Report driving issues in the `#driving-feedback` Discord channel.
* Consider opting into driver camera uploads to improve the driver monitoring model.
* Connect your device to Wi-Fi regularly, so that we can pull data for training better driving models.
* Run the `nightly` branch and report issues. This branch is like `master` but it's built just like a release.
* Annotate images in the [comma10k dateset](https://github.com/commaai/comma10k).
