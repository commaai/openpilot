# How to contribute

Our software is open source so you can solve your own problems without needing help from others. And if you solve a problem and are so kind, you can upstream it for the rest of the world to use.

Most open source development activity is coordinated through our [Discord](https://discord.comma.ai). A lot of documentation is available on our [medium](https://medium.com/@comma_ai/)

## Getting Started

 * Join our [Discord](https://discord.comma.ai)
 * Make sure you have a [GitHub account](https://github.com/signup/free)
 * Fork [our repositories](https://github.com/commaai) on GitHub

## Testing

### Local Testing

You can test your changes on your machine by running `run_docker_tests.sh`. This will run some automated tests in docker against your code. 

### Automated Testing

All PRs are automatically checked by travis. Check out `.travis.yml` for what travis runs. Any new tests sould be added to travis.

### Code Style and Linting

Code is automatically check for style by travis as part of the automated tests. You can also run these yourself by running `check_code_quality.sh`. 

## Car Ports (openpilot)

We've released a [Model Port guide](https://medium.com/@comma_ai/openpilot-port-guide-for-toyota-models-e5467f4b5fe6) for porting to Toyota/Lexus models.

If you port openpilot to a substantially new car brand, see this more generic [Brand Port guide](https://medium.com/@comma_ai/how-to-write-a-car-port-for-openpilot-7ce0785eda84). You might also be eligible for a bounty. See our bounties at [comma.ai/bounties.html](https://comma.ai/bounties.html)

## Pull Requests

Pull requests should be against the master branch. Before running master on in-car hardware, you'll need to run
```
git submodule init
git submodule update
```
in order to pull down the submodules, such as `panda` and `opendbc`.
