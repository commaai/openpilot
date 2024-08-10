# Roadmap

This is the roadmap for the next major openpilot releases. Also check out

* [Milestones](https://github.com/commaai/openpilot/milestones) for minor releases
* [Projects](https://github.com/commaai/openpilot/projects?query=is%3Aopen) for shorter-term projects not tied to releases
* [Bounties](https://comma.ai/bounties) for paid individual issues
* [#current-projects](https://discord.com/channels/469524606043160576/1249579909739708446) in Discord for discussion on work-in-progress projects

## openpilot 0.10

openpilot 0.10 will be the first release with a driving policy trained in
a [learned simulator](https://youtu.be/EqQNZXqzFSI).

* Driving model trained in a learned simlator
* Always-on driver monitoring (behind a toggle)
* GPS removed from the driving stack
* 100KB qlogs
* `master-ci` pushed after 1000 hours of hardware-in-the-loop testing
* Car interface code moved into [opendbc](https://github.com/commaai/opendbc)
* openpilot on PC for Linux x86, Linux arm64, and Mac (Apple Silicon)

## openpilot 1.0

openpilot 1.0 will feature a fully end-to-end driving policy.

* End-to-end longitudinal control in Chill mode
* Automatic Emergency Braking (AEB)
* Driver monitoring with sleep detection
* Rolling updates/releases pushed out by CI
* [panda safety 1.0](https://github.com/orgs/commaai/projects/27)
